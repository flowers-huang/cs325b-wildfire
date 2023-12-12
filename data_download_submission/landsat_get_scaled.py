import xarray as xr


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


def calculate_composite(
        ds,
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

    # # Build paths
    # if event_id is not None:
    #     array_path = os.path.join(path_to_array, f"{event_id}.nc4")
    # else:
    #     array_path = path_to_array

    # # Open files and prepare for QA cleaning
    # ds = xr.open_mfdataset(array_path)

    # Get metadata from files
    ds = ds.to_dataset()
    var_name = list(ds.data_vars.keys())[0]
    no_meta_vars = ["time", "x", "y", var_name]

    raw_meta = {k: v.to_numpy().tolist() for k, v in dict(ds.variables).items()
                if k not in no_meta_vars}

    if apply_qa_bitmask:
        ds = apply_bit_mask_group(ds)

    # Open datasets and calculate median
    ds = (
        ds
        .median(dim="time")
        .rio.write_crs("4326")
        .to_array()
        .squeeze()
    )

    if save is not None:
        ds = ds.rio.to_raster(save, tags=raw_meta)

    return ds

from math import sqrt
from typing import Any, Dict

import geopandas
import numpy as np
import pandas as pd
import pyproj
import pystac
from shapely import geometry, wkb
from shapely.geometry import Point
from shapely.ops import transform
from datetime import datetime, date, timedelta

def num_day_to_datetime(day_num, year) -> datetime:
    """Transformd from day of the year to datetime object

    Take a day number and return it as a datetime object for an specific year

    Args:
        - day_num (int): An integer representing the day of the year
        - year (int): year for date

    Returns:
        datetime
    """

    start_date = date(year, 1, 1)

    resolved_time = start_date + timedelta(days=int(day_num) - 1)

    return resolved_time

def intersection_percent(item: pystac.Item, aoi: Dict[str, Any]) -> float:
    """The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    """
    geom_item = geometry.shape(item.geometry)
    geom_aoi = geometry.shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)
    # item_difference = geom_item.difference(geom_aoi)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area
    # intersection_percent = ((geom_item.area - item_difference.area) * 100) / geom_item.area

    return intersection_percent

from functools import cached_property
import gzip
import os
import shutil
from urllib.parse import urlparse
import warnings

from datetime import datetime
from rasterio.enums import Resampling
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
import planetary_computer
import pystac_client
import requests
from shapely import geometry
from tqdm import tqdm

import adlfs
import stackstac


class Landsat:
    """Search and download Landsat imagery

    This class searchs and download Landsat scenes corresponding to overlapping
    geometries and metadata.
    """

    def __init__(
            self,
            aoi_path,
            cloud_coverage,
            save_path,
            date_window,
            buffer=10000,
            date_window_type="delta",
            collection="landsat-c2-l2",
            force_update=False,
    ) -> None:
        self.cloud_coverage: float = cloud_coverage
        self.aoi_path: str = aoi_path
        self.save_path: str = save_path
        self.buffer: int = buffer
        self.aoi_date = "Ig_Date"
        self.collection: str = collection
        self.date_window_type = date_window_type

        if not isinstance(date_window, tuple):
            self.date_window: tuple = tuple([date_window, date_window])
        else:
            self.date_window: tuple = date_window

        # Subset bands of interest in Landsat collection
        self.bands: list[str] = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]
        self.platform: list[str] = ["landsat-4", "landsat-5", "landsat-8", "landsat-9"]

    @cached_property
    def catalog(self) -> pystac_client.Client:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1/",
            modifier=planetary_computer.sign_inplace,
        )

        return catalog

    @cached_property
    def aoi(self) -> gpd.GeoDataFrame:
        """Property store for area of interests"""

        if isinstance(self.aoi_path, str):
            aoi: gpd.GeoDataFrame = gpd.read_file(self.aoi_path)

        elif isinstance(self.aoi_path, gpd.GeoDataFrame):
            aoi: gpd.GeoDataFrame = self.aoi_path
        else:
            raise RuntimeError(f"{self.aoi_path} is not a valid format")

        # Check projection and reproject to vanilla mercator
        if aoi.crs.to_epsg() != "4326":
            aoi: gpd.GeoDataFrame = aoi.to_crs("EPSG:4326")

        # Create time search variables
        aoi[self.aoi_date] = pd.to_datetime(aoi[self.aoi_date])

        left, right = self.date_window

        # Manage date ranges depending on variable type

        if self.date_window_type == "delta":
            # Do time range using a time window with the ignition date in the middle
            aoi = aoi.assign(
                pre_date=aoi[self.aoi_date] - pd.Timedelta(days=left),
                post_date=aoi[self.aoi_date] + pd.Timedelta(days=right),
            )
        elif self.date_window_type == "pre_year":
            # Do a time range using left and right!
            aoi = aoi.assign(
                pre_date=aoi[self.aoi_date].dt.year.apply(
                    lambda x: num_day_to_datetime(left, x - 1)
                ),
                post_date=aoi[self.aoi_date].dt.year.apply(
                    lambda x: num_day_to_datetime(right, x - 1)
                ),
            )

        elif self.date_window_type == "range_ignition":
            aoi = aoi.assign(
                pre_date=aoi[self.aoi_date].dt.year.apply(
                    lambda x: num_day_to_datetime(left, x)
                ),
                post_date=aoi[self.aoi_date],
            )
            # Remove all fires out of the fire season range
            aoi = aoi[aoi.pre_date <= aoi.Ig_Date]

        elif self.date_window_type == "post_year":
            # Do a time range using left and right but for the next year
            aoi = aoi.assign(
                pre_date=aoi[self.aoi_date].dt.year.apply(
                    lambda x: num_day_to_datetime(left, x + 1)
                ),
                post_date=aoi[self.aoi_date].dt.year.apply(
                    lambda x: num_day_to_datetime(right, x + 1)
                ),
            )

        return aoi

    def request_items_stac(self, start_date, end_date, geometry_obj, bbox_size=(0.1, 0.1)):
        """Request landsat metadata using the PC catalog seach

        Args:
            - start_date pd.DateTime: Start time for search
            - end_date pd.DateTime: End of time for search
            - geometry_obj dict: A GeoJSON object with the bounding box of each
              AOI.

        Returns:
            pystac.Collection
        """

        # Build datetime string
        start_time_str: str = start_date.strftime("%Y-%m-%d")
        end_time_str: str = end_date.strftime("%Y-%m-%d")
        timerange: str = f"{start_time_str}/{end_time_str}"

        # if hasattr(geometry_obj, "__geo_interface__"):
        #     geometry_bounds = geometry_obj.bounds
        # else:
        #     geometry_bounds = geometry_obj

        geom_centroid = geometry_obj.centroid
        bounds = geometry_obj.bounds
        bbox_size = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
        r = max(*bbox_size, 0.2)
        bbox_size = [r, r]

        minx, miny = geom_centroid.x - bbox_size[0] / 2, geom_centroid.y - bbox_size[1] / 2
        maxx, maxy = geom_centroid.x + bbox_size[0] / 2, geom_centroid.y + bbox_size[1] / 2

        search = self.catalog.search(
            collections=self.collection,
            bbox=(minx, miny, maxx, maxy),
            datetime=timerange,
            query={
                "eo:cloud_cover": {"lt": self.cloud_coverage},
                "platform": {"in": self.platform},
            },
        )

        # Make collection
        items_search = search.get_all_items()

        # Filter out all the items that do not cover the geometry 100%. This
        # might not be what we want in all applications (i.e. building a mosaic
        # or a composite), but for now is convenient to do it here.
        item_aoi_coverage: list[str] = [
            intersection_percent(item, geometry_obj) for item in items_search.items
        ]

        print(item_aoi_coverage)

        filter_items = [
            item
            for item, coverage in zip(items_search, item_aoi_coverage)
            if float(coverage) > 99
        ]

        return filter_items

    def execute_search_aoi(self, bbox_size=(0.1, 0.1)):
        """Execute search in all the elements of the AOI"""

        dict_aoi: list = self.aoi.to_dict(orient="records")

        for aoi_element in tqdm(dict_aoi, desc="Downloading from PC..."):
            # Create path to later check for file existence
            # path_to_save = os.path.join(
            #     self.save_path, f"{aoi_element['Event_ID']}.nc4"
            # )
            path_to_save = os.path.join(
                self.save_path, f"{aoi_element['Event_ID']}.tif"
            )

            if not os.path.exists(path_to_save):
                try:
                    items_aoi = self.request_items_stac(
                        start_date=aoi_element['pre_date'],
                        end_date=aoi_element['post_date'],
                        geometry_obj=aoi_element['geometry'],
                        bbox_size=bbox_size
                    )

                    if len(items_aoi) > 0:
                        geom_centroid = aoi_element["geometry"].centroid
                        bounds = aoi_element["geometry"].bounds
                        bbox_size = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
                        r = max(*bbox_size, 0.2)
                        bbox_size = [r, r]

                        minx, miny = geom_centroid.x - bbox_size[0] / 2, geom_centroid.y - bbox_size[1] / 2
                        maxx, maxy = geom_centroid.x + bbox_size[0] / 2, geom_centroid.y + bbox_size[1] / 2

                        data = stackstac.stack(
                            items_aoi,
                            bounds=(minx, miny, maxx, maxy),
                            assets=self.bands,
                            epsg=4326,
                            resampling=Resampling.bilinear,
                            snap_bounds=True,
                        )

                        data.attrs = {}
                        data = data.drop_vars(
                            [
                                "instruments",
                                "raster:bands",
                                "center_wavelength",
                                "proj:epsg",
                                "proj:shape",
                                "gsd",
                                "proj:transform",
                                "landsat:collection_category",
                                "landsat:collection_number",
                                "landsat:correction",
                                "landsat:wrs_type",
                                "description",
                                "view:off_nadir",
                                "landsat:wrs_path",
                                "landsat:wrs_row",
                                "sci:doi",
                                "classification:bitfields",
                            ],
                            errors="ignore",
                        )

                        # file_name = f"{raw_event}.tif"
                        # save_path = os.path.join(path_save, file_name)

                        composite = calculate_composite(
                            ds=data,
                            apply_qa_bitmask=True,
                            save=path_to_save
                        )
                        # composite.compute()
                        # data.to_netcdf(path_to_save)
                    else:
                        print(f"{aoi_element['Event_ID']} has no items!")

                except RuntimeError as e:
                    print(f"{aoi_element['Event_ID']} failed with: {e}")
                    pass

                except ValueError as e:
                    print(f"{aoi_element['Event_ID']} failed with: {e}")
                    pass

        return None