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


def buffer_bounding_box(bounds, buf=10000) -> tuple:
    """Create a buffer around a bounding box object

    Taking the coordinates of a polygon boundary, this function calculates a
    buffer in meters and return it in the same CRS that the input boundaries.

    Args:
        - bounds (tuple): A bounds coordinates object. In a geometry you can
          use: geometry.bounds
        - buf (int): buffer size in meters
    """

    buffer_width_m = (distance_km * 1000) / sqrt(2)
    (p_lat, p_long) = point_lat_long

    # create Shapely Point object, coodrinates as x,y
    wgs84_pt = Point(p_long, p_lat)
    # set up projections WGS84 (lat/long coordinates) for input and
    # UTM to measure distance
    # https://epsg.io/4326
    wgs84 = pyproj.CRS("EPSG:4326")
    # sample point in France - UTM zone 31N
    # Between 0°E and 6°E, northern hemisphere between equator and 84°N
    # https://epsg.io/32631
    utm = pyproj.CRS("EPSG:32631")

    # transformers:
    project_wgs84_to_utm = pyproj.Transformer.from_crs(
        wgs84, utm, always_xy=True
    ).transform
    project_utm_to_wgs84 = pyproj.Transformer.from_crs(
        utm, wgs84, always_xy=True
    ).transform

    # tranform Point to UTM
    utm_pt = transform(project_wgs84_to_utm, wgs84_pt)
    # create square buffer (cap_style = 3) around the Point
    utm_buffer = utm_pt.buffer(buffer_width_m, cap_style=3)

    return wgs84_bounds


def intersection_percent(item: pystac.Item, aoi: Dict[str, Any]) -> float:
    """The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    """
    geom_item = geometry.shape(item.geometry)
    geom_aoi = geometry.shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent
