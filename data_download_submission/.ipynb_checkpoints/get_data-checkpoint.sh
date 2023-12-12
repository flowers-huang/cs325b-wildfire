# define the path to a root data directory
path="path/to/data"

# replace all paths with variable in name
python data_download_submission/landsat_download.py --path_to_shape https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip --path_to_save "${path}/raw/post-fire-images" --cloud_coverage 40 --buffer 180 --post
python data_download_submission/landsat_download.py --path_to_shape https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip --path_to_save "${path}/raw/pre-fire-images" --cloud_coverage 40 --buffer 180 --no-post
python data_download_submission/create_composites.py --path_to_data "${path}/raw/post-fire-images" --path_to_save "${path}/processed/post-fire-images"
python data_download_submission/create_composites.py --path_to_data "${path}/raw/pre-fire-images" --path_to_save "${path}/processed/pre-fire-images"
python data_download_submission/generate_dataset.py --path_to_data "${path}/processed"