import argparse
import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio
import requests
import rioxarray
import xarray as xr
import zipfile
from dask.diagnostics import ProgressBar
from itertools import combinations
from pathlib import Path
from rasterio.features import rasterize
from tqdm import tqdm
import warnings


def download_and_extract_zip(url, directory):
    filename = url.split("/")[-1]
    os.makedirs(directory, exist_ok=True)

    # Download the zip file
    response = requests.get(url)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(directory)

    # Remove the zip file if you want
    os.remove(filepath)


def compute_mask(geometry, ds):
    # Convert selected_event to shape, value format
    shapes = [(geom, 1) for geom in geometry]

    # Get the transforms
    transform = ds.rio.transform()

    x, y = len(ds.coords['y']), len(ds.coords['x'])

    # Rasterize the geometry using rasterio.features.rasterize
    mask = rasterize(shapes=shapes, transform=transform, out_shape=(x, y))

    return mask


def process_shared_files(shared_files, files_post, wildfires_ca, save_dir):
    for region in tqdm(shared_files):
        ds = (
            xr.open_dataset(files_post[region])
            .to_array()
            .squeeze()
        )
        ds.rio.set_crs(4326)
        selected_event = wildfires_ca[wildfires_ca.Event_ID == region]

        if len(selected_event) == 0:
            continue

        mask = compute_mask(selected_event.geometry, ds)

        np.save(os.path.join(save_dir, f"{region}.npy"), mask)


def reshape_pre_fire_images_to_same_size_as_post_fire(regions, files_post, files_pre, files_pre_scaled):
    for region in tqdm(regions):
        ds_post = (
            xr.open_dataset(files_post[region])
            .to_array()
            .squeeze()
        )
        ds_post.rio.set_crs(4326)

        ds_pre = (
            xr.open_dataset(files_pre[region])
            .to_array()
            .squeeze()
        )
        ds_pre.rio.set_crs(4326)

        ds_pre = ds_pre.interp_like(ds_post, method="linear", kwargs={"fill_value": "extrapolate"})

        ds_pre.rio.to_raster(files_pre_scaled[region])

        ds_pre.compute()


def main(root_dir):
    url = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip'
    geometry_dir = root_dir / "geometry"
    download_and_extract_zip(url, geometry_dir)

    pre_root_dir = root_dir / "pre-fire-images"
    post_root_dir = root_dir / "post-fire-images"
    mask_root_dir = root_dir / "masks"

    files_pre = {Path(f).stem: pre_root_dir / f for f in os.listdir(pre_root_dir)}
    files_post = {Path(f).stem: post_root_dir / f for f in os.listdir(post_root_dir)}

    shared_files = list(files_pre.keys() | files_post.keys())

    # Open the MTBS data, this is the same data you get from the MTBS webpage
    mtbs = gpd.read_file(geometry_dir)
    wildfires_ca = mtbs[mtbs.Incid_Type == "Wildfire"]
    wildfires_ca = wildfires_ca.to_crs("EPSG:4326")

    process_shared_files(shared_files, files_post, wildfires_ca, mask_root_dir)
    reshape_pre_fire_images_to_same_size_as_post_fire(shared_files, files_post, files_pre, files_pre)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Args for Paths')
    parser.add_argument('--root_dir', required=True, help='Root directory containing pre and post fire images and to save geometry/masks')
    args = parser.parse_args()

    main(root_dir=Path(args.root_dir))