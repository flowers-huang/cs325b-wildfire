import pdb
import os
import dask
import rioxarray
import numpy as np
import xarray as xr
import pandas as pd

import warnings
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
        arr = arr.groupby("platform").map(apply_bitmask)
    except TypeError or IndexError as e:
        print(f"Exception: {e}. Maybe platform length is zero: {arr.platform.values}")
        arr = apply_bitmask(arr)

    return arr


def calculate_composite(
    path_to_array,
    apply_qa_bitmask=True,
    geometry=None,
    event_id=None,
    save=None
) -> xr.DataArray:
    """Clean imagery array and calculate mean composite for each geometry

    Transform each of the multidimensional arrays (i.e. NetCDFs) into a median
    composite. This will effectively take the mean for each of the bands in the
    array, but the QA band, and return a georeferenced image (GeoTIFF). This
    function can also mask to the geometry. 

    Args:
        - path_to_array (str): Path to array imagery file
        - geometry (gpd.GeoDataFrame): Spatial dataframe object with event data.
          The function expect that event_id is identical.
        - apply_qa_bitmask (bool): If `True` apply the bitmask and remove all
          bad pixels following the QA flags
        - event_id (str): Name of event. Default is `None`. 
        - mask (bool): If `True`, mask the array to the geometry boundary.
        - save (str): A path to save the restulting array. If `None`, then it
          won't save it.

    Returns:
        Delayed xr.Dataset. To get the array in-memory, use the .compute() method
        or .gather(). 
    """

    # Build paths
    if event_id is not None:
        array_path = os.path.join(path_to_array, f"{event_id}.nc4")
    else:
        array_path = path_to_array

    # Open files and prepare for QA cleaning
    ds = xr.open_mfdataset(array_path)

    # Get metadata from files
    var_name = list(ds.data_vars.keys())[0]
    no_meta_vars = ["time", "x", "y", var_name]

    raw_meta = {k: v.to_numpy().tolist() for k,v in dict(ds.variables).items()
                if k not in no_meta_vars}
    
    if apply_qa_bitmask:
        ds = apply_bit_mask_group(ds)

    # Open datasets and calculate mean
    ds = (
            ds
            .mean(dim="time")
            .rio.write_crs("4326")
            .to_array()
            .squeeze()
            )

    if save is not None:
        ds = ds.rio.to_raster(save, tags=raw_meta)

    return ds
