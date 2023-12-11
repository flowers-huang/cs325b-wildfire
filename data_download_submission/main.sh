# replace all paths with sarah in name 
python landsat_download.py --path_to_shape https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip --path_to_save /Users/sarah/Desktop/CS325B/post_nc4 --cloud_coverage 40 --buffer 180 --post
python landsat_download.py --path_to_shape https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip --path_to_save /Users/sarah/Desktop/CS325B/pre_nc4 --cloud_coverage 40 --buffer 180 --no-post
python create_composites.py --path_to_data /Users/sarah/Desktop/CS325B/post_nc4 --path_to_save /Users/sarah/Desktop/CS325B/processed_post
python create_composites.py --path_to_data /Users/sarah/Desktop/CS325B/pre_nc4 --path_to_save /Users/sarah/Desktop/CS325B/processed_pre
python get_mask.py --tif_dir /Users/sarah/Desktop/CS325B/processed_post --save_folder /Users/sarah/Desktop/CS325B/masks
