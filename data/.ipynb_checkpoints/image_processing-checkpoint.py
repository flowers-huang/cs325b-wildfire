import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from einops import rearrange, pack

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from pprint import pprint
from torch.utils.data import DataLoader

import dask
import rioxarray
import xarray as xr
import pandas as pd

import warnings
from itertools import combinations
from dask.diagnostics import ProgressBar

ROBUST_PERCENTILE = 2.0

def rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True, axis=-1):
    assert robust or vmin is not None or vmax is not None

    ndim = len(darray.shape)
    if axis < 0:
        axis = ndim + axis

    reduce_dim = list(range(ndim))
    reduce_dim.remove(axis)

    # Calculate vmin and vmax automatically for `robust=True`
    # Assume that the last dimension of the array represents color channels
    # Make sure to apply np.nanpercentile over this dimension by specifying axis=-1
    if robust:
        if vmax is None:
            vmax = np.nanpercentile(darray, 100 - ROBUST_PERCENTILE, axis=reduce_dim, keepdims=True)
        if vmin is None:
            vmin = np.nanpercentile(darray, ROBUST_PERCENTILE, axis=reduce_dim, keepdims=True)
    # If not robust and one bound is None, calculate the default other bound
    # and check that an interval between them exists.
    elif vmax is None:
        vmax = 255 if np.issubdtype(darray.dtype, np.integer) else 1
        if np.any(vmax < vmin):
            raise ValueError(
                f"vmin={vmin!r} less than the default vmax ({vmax!r}) - you must supply "
                "a vmax > vmin in this case."
            )
    elif vmin is None:
        vmin = 0
        if np.any(vmin > vmax):
            raise ValueError(
                f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                "a vmin < vmax in this case."
            )
    # Compute a mask for where vmax equals vmin
    vmax_equals_vmin = np.isclose(vmax, vmin)

    # Avoid division by zero by replacing zero divisors with 1
    divisor = np.where(vmax_equals_vmin, vmax, vmax - vmin)

    # Scale interval [vmin .. vmax] to [0 .. 1], using darray as 64-bit float
    darray = ((darray.astype("f8") - vmin) / divisor).astype("f4")
    
    # return np.nan_to_num(np.minimum(np.maximum(darray, 0), 1) * 255).astype(np.uint8)
    return np.nan_to_num(darray)



def apply_bitmask(arr) -> xr.DataArray or np.array:
    """Apply QA pixel bit mask for each array depending on platform"""

    unique_platform = np.unique(arr.platform.to_numpy())

    if ["landsat-8", "landsat-9"] in unique_platform:
        mask_bitfields = [1, 2, 3, 4]  # dilated cloud, cirrus, cloud, cloud shadow
    elif ["landsat-4", "landsat-5", "landsat-7"] in unique_platform:
        mask_bitfields = [1, 3, 4, 5]  # dilated cloud, cirrus, cloud, cloud shadow
    elif ["landsat-4", "landsat-5"] in unique_platform:
        mask_bitfields = [1, 3, 4, 5]  # dilated cloud, cirrus, cloud, cloud shadow
    else:
        raise ValueError(f"No bit mask defined for {arr.platform.to_numpy()}")

    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    qa = arr.sel(band="qa_pixel").astype("uint16")
    bad = qa & bitmask  # just look at those 4 bits

    arr = arr.where(bad == 0)

    return arr


def calculate_nbr(data):
    """Calculate NBR in xarray DataArray

    Calculate the Normalized Burn Ratio for a DataArray with Landsat data. This
    function uses the NBR standard formula: .

    Args:
        - data (xr.DataArray): A data array with the "nir08" and the "swir16"
          bands.

    Returns:
        xr.DataArray
    """

    # Calculate NBR manually using the normal formula
    # nbr_manual = (
    #     data.sel(band="nir08").squeeze() - data.sel(band="swir16").squeeze()
    # ) / (data.sel(band="nir08").squeeze() + data.sel(band="swir16").squeeze())
    nbr_manual = (
        data.sel(band=3).squeeze() - data.sel(band=4).squeeze()
    ) / (data.sel(band=3).squeeze() + data.sel(band=4).squeeze())

    return nbr_manual


@dask.delayed
def apply_bit_mask_group(arr):
    """Apply QA mask to array across different platforms

    Apply the QA mask on xarray object using the platform variable. If the
    coordinate is not available, then an error will be raised! If the platform
    value is unique, then is only applied to the unique platform.

    Args:
        - arr (xr.Dataset or xr.DataArray)

    Returns:
        xr.Dataset or xr.DataArray
    """

    # Apply QA bitmask
    try:
        if len(arr['platform'].dims) == 0:
          arr = apply_bitmask(arr)
        else:
          arr = arr.groupby("platform").map(apply_bitmask)
    except TypeError or IndexError as e:
        print(f"Exception: {e}. Maybe platform length is zero: {arr.platform.values}")
        arr = apply_bitmask(arr)

    return arr


def calculate_dnbr(
    pre_data,
    post_data,
    apply_qa_bitmask=False
) -> xr.DataArray:
    """Calculate dNBR for a specific even across timeline

    Following Parks (2019), we calculate the dNBR using two sets of images: one
    pre- and post-event. We calculate the NBR using the mean composite of both
    sets and then subtract the pre values from the post period. Since we want to
    compare events, we calculate an offset using the buffer. Then our dNBR for
    each pixel is:

        \[
        \DeltaNBR_{i} = NBR_{i, t=1} - NBR_{i, t=0} - NBR_{offset}
        \]

    Args:
        - pre_array (str): path to pre-array path for the event. If event_id is
          passed, then only the root path will be used.
        - post_array (str): path to post-array for the same event. If event_id
          is passed, then only the root path will be used
        - geometry (gpd.GeoDataFrame): Spatial dataframe object with event data.
          The function expect that event_id is identical.
        - buffer_offset_size (int): Buffer distance to build offset value. If
          None, then no offset will be calculated.
        - apply_qa_bitmask (bool): If `True` apply the bitmask and remove all
          bad pixels following the QA flags
        - event_id (str): Event ID to find images in pre and post imagery paths
          and in geometry.
        - save (str): A path to save the restulting array. If `None`, then it
          won't save it.

    Returns:
        xr.DataArray
    """

    # Keep this format for all dates
    format_date = "%Y-%m-%d"

    if apply_qa_bitmask:
        pre_data = apply_bit_mask_group(pre_data)
        post_data = apply_bit_mask_group(post_data)

    # Calculate NBR for both periods
    pre_nbr = calculate_nbr(pre_data)
    post_nbr = calculate_nbr(post_data)

    # Calculate dNBR for all the pixels
    dnbr = pre_nbr - post_nbr

    return dnbr.compute()

def random_crop(image, mask, crop_size):
    """
    This function takes an image as a numpy array and desired crop size,
    then outputs a random crop of that size.
    """
        
    # Image dimensions
    max_y, max_x = image.shape[-2:]
    
    # Ensure that crop dimensions are not larger than the image 
    if crop_size > max_y or crop_size > max_x:
        raise ValueError(f"Crop dimensions ({crop_size}, {crop_size}) larger than image size ({max_y}, {max_x})")

    # Randomly select the starting pixel for the crop
    start_y = np.random.randint(0, max_y - crop_size)
    start_x = np.random.randint(0, max_x - crop_size)

    return image[:, start_y:start_y + crop_size, start_x:start_x + crop_size], mask[:, start_y:start_y + crop_size, start_x:start_x + crop_size]

def get_overlapping_tiles(image, tile_size, overlap):
    """
    This function takes an image as a numpy array, desired tile size,
    and overlap between tiles, then outputs a list of these tiles.
    """
    
    step_size = tile_size - overlap
    assert overlap < tile_size

    tiles = []
    max_y, max_x = image.shape[-2:]

    for x in range(0, max_x - tile_size + 1, step_size):
        for y in range(0, max_y - tile_size + 1, step_size):
            tile = image[:, y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            
    # Adjusting the position of last tiles
    if max_x - x > tile_size:
        for y in range(0, max_y - tile_size + 1, step_size):
            tile = image[:, y:y + tile_size, max_x - tile_size:max_x]
            tiles.append(tile)
    
    if max_y - y > tile_size:
        for x in range(0, max_x - tile_size + 1, step_size):
            tile = image[:, max_y - tile_size:max_y, x:x + tile_size]
            tiles.append(tile)
    
    # corner tile
    if max_x - x > tile_size and max_y - y > tile_size:
        tile = image[:, max_y - tile_size:max_y, max_x - tile_size:max_x]
        tiles.append(tile)

    return tiles


def combine_tiles(tiles, img_height, img_width, tile_size, overlap):
    """
    This function takes a list of tiles and combines them to reconstruct the original image.
    The average is taken in overlapping regions.
    tiles: List of tensors of shape (c, tile_size, tile_size)
    """

    step_size = tile_size - overlap
    img_depth = tiles[0].shape[0]
    combined_image = torch.zeros((img_depth, img_height, img_width))
    coverage_matrix = torch.zeros((img_height, img_width), dtype=int)

    idx = 0
    for x in range(0, img_width - tile_size + 1, step_size):
        for y in range(0, img_height - tile_size + 1, step_size):
            
            # Place each tile back into image and increment the coverage matrix
            combined_image[:, y:y + tile_size, x:x + tile_size] += tiles[idx]
            coverage_matrix[y:y + tile_size, x:x + tile_size] += 1
            idx += 1

    # Adjusting the position of last tiles
    if img_width - x > tile_size:
        for y in range(0, img_height - tile_size + 1, step_size):
            combined_image[:, y:y + tile_size, img_width - tile_size:img_width] += tiles[idx]
            coverage_matrix[y:y + tile_size, img_width - tile_size:img_width] += 1
            idx += 1

    if img_height - y > tile_size:
        for x in range(0, img_width - tile_size + 1, step_size):
            combined_image[:, img_height - tile_size:img_height, x:x + tile_size] += tiles[idx]
            coverage_matrix[img_height - tile_size:img_height, x:x + tile_size] += 1
            idx += 1

    # corner tile
    if img_width - x > tile_size and img_height - y > tile_size:
        combined_image[:, img_height - tile_size:img_height, 
                       img_width - tile_size:img_width] += tiles[idx]
        coverage_matrix[img_height - tile_size:img_height, 
                        img_width - tile_size:img_width] += 1
        
    # Ensure no division by zero
    coverage_matrix[coverage_matrix == 0] = 1
  
    # Calculate average by dividing combined_image by coverage_matrix
    reconstructed_image = combined_image / coverage_matrix
    
    return reconstructed_image

def resize_and_pad(img, output_size=(256,256), padding_mode='constant'):
    # Get initial dimensions
    c, h, w = img.shape

    # Calculate new dimension while maintaining the aspect ratio
    if h >= w:
        new_h, new_w = output_size[0], int(output_size[0] * w / h)
    else:
        new_h, new_w = int(output_size[1] * h / w), output_size[1]

    # Resize the image
    img = F.resize(img, (new_h, new_w))

    # This is padding, it requires padding_left, padding_right,
    # padding_top and padding_bottom respectively.
    pad_height = max(output_size[0] - img.shape[1], 0)
    pad_width = max(output_size[1] - img.shape[2], 0)

    # Center padding
    pad_height1 = pad_height // 2
    pad_height2 = pad_height - pad_height1
    pad_width1 = pad_width // 2
    pad_width2 = pad_width - pad_width1

    # Pad the image
    img = F.pad(img, (pad_width1, pad_height1, pad_width2, pad_height2), padding_mode=padding_mode)

    return img