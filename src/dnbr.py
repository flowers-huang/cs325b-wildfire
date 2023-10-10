import pdb
import os
import dask
import rioxarray
import numpy as np
import xarray as xr
import pandas as pd

import warnings
from itertools import combinations
from dask.diagnostics import ProgressBar


def apply_bitmask(arr) -> xr.DataArray or np.array:
    """Apply QA pixel bit mask for each array depending on platform"""

    unique_platform = np.unique(arr.platform.to_numpy())

    if ["landsat-8", "landsat-9"] in unique_platform:
        mask_bitfields = [1, 2, 3, 4]  # dilated cloud, cirrus, cloud, cloud shadow
    elif ["landsat-4", "landsat-5", "landsat-7"] in unique_platform:
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


@dask.delayed
def calculate_nbr(data):
    """Calculate NBR in xarray DataArray

    Calculate the Normalized Burn Ratio for a DataArray with Landsat data. This
    function uses the NBR standard formula: (NIR-SWIR)/(NIR+SWIR).

    Args:
        - data (xr.DataArray): A data array with the "nir08" and the "swir16"
          bands.

    Returns:
        xr.DataArray
    """

    # Calculate NBR manually using the normal formula
    nbr_manual = (
        data.sel(band="nir08").squeeze() - data.sel(band="swir16").squeeze()
    ) / (data.sel(band="nir08").squeeze() + data.sel(band="swir16").squeeze())

    return nbr_manual


@dask.delayed
def apply_bit_mask_group(arr, event):
    """Apply QA mask to array across different platforms

    Apply the QA mask on xarray object using the platform variable. If the
    coordinate is not available, then an error will be raised! If the platform
    value is unique, then is only applied to the unique platform.

    Args:
        - arr (xr.Dataset or xr.DataArray)

    Returns:
        xr.Dataset or xr.DataArray
    """

    print(event)
    # Apply QA bitmask
    try:
        arr = arr.groupby("platform").map(apply_bitmask)
    except TypeError or IndexError as e:
        print(f"Exception: {e}. Maybe platform length is zero: {arr.platform.values}")
        arr = apply_bitmask(arr)

    return arr


def calculate_dnbr(
    pre_array,
    post_array,
    geometry,
    buffer_offset_size=180,
    apply_qa_bitmask=True,
    event_id=None,
    save=None,
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

    # Build paths
    if event_id is not None:
        pre_array_path = os.path.join(pre_array, f"{event_id}.nc4")
        post_array_path = os.path.join(post_array, f"{event_id}.nc4")
    else:
        pre_array_path, post_array_path = (pre_array, post_array)

    # Open files and prepare for QA cleaning
    pre_data = xr.open_mfdataset(pre_array_path)
    post_data = xr.open_mfdataset(post_array_path)

    if apply_qa_bitmask:
        pre_data = apply_bit_mask_group(pre_data, event_id)
        post_data = apply_bit_mask_group(post_data, event_id)

    # Open datasets and calculate mean
    pre_data = pre_data.mean(dim="time").rio.write_crs("4326").to_array().squeeze()
    # Notice that we use interp here to account to possible differences in the
    # coordinates within pre and post. This is similar to a bilinear
    # interpolation where
    post_data = (
        post_data.mean(dim="time")
        .rio.write_crs("4326")
        .to_array()
        .squeeze()
        .interp_like(pre_data, method="linear")
    )

    # Calculate offset
    if buffer_offset_size is not None:
        # Check projection
        if not geometry.crs.is_projected:
            raise TypeError(f"{geometry.crs} no good for these calculations")

        try:
            geom_event = geometry[geometry.Event_ID == event_id]
            buffer = geom_event.buffer(buffer_offset_size)

            # Calculate difference and project back to lat/lon
            offset = buffer.difference(geom_event)
            offset_planar = offset.to_crs(4326).geometry.values

        except KeyError as e:
            print(
                f"Exception {e}: Cannot find event id columnd maybe? {geometry.columns}"
            )

        # Caclulate off-set in data
        pre_data_ring = pre_data.rio.clip(
            offset_planar, crs=4326, drop=False, invert=False
        )
        post_data_ring = post_data.rio.clip(
            offset_planar, crs=4326, drop=False, invert=False
        )

        # Calculate NBR
        pre_nbr_ring = calculate_nbr(pre_data_ring)
        post_nbr_ring = calculate_nbr(post_data_ring)

        dnbr_ring = (pre_nbr_ring - post_nbr_ring).mean().squeeze()

    else:
        dnbr_ring = 0

    # Calculate NBR for both periods
    pre_nbr = calculate_nbr(pre_data)
    post_nbr = calculate_nbr(post_data)

    # Calculate dNBR for all the pixels
    dnbr = (pre_nbr - post_nbr) - dnbr_ring

    if save is not None:
        dnbr.rio.to_raster(save, tiled=True, windowed=True)

    return dnbr
