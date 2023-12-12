import os
import argparse
import geopandas as gpd

from landsat_get_scaled import Landsat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_shape", type=str, help="Path to extration shape")
    parser.add_argument("--path_to_save", type=str, help="Path to save data")
    parser.add_argument(
        "--cloud_coverage", type=int, default=40, help="Cloud coverage treshold"
    )
    parser.add_argument(
        "--buffer", type=int, default=180, help="Cloud coverage treshold"
    )
    parser.add_argument(
        "--post",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Get post-fire data",
    )

    args = parser.parse_args()

    # Load geometry and subset to wildfires only
    mtbs = gpd.read_file(args.path_to_shape)
    wildfires = mtbs[
        (mtbs.Event_ID.str.contains("CA")) & (mtbs.Incid_Type == "Wildfire")
    ]

    # Create buffers for extraction! remember we need to calculate and offset for each
    # fire
    wildfires_meters = wildfires.to_crs(3310)  # NAD83 for CA

    wildfire_buffers = wildfires.copy()
    wildfire_buffers["geometry"] = wildfires_meters.buffer(args.buffer).to_crs(
        4326
    )  # Go back to mercator

    print("Start class and start extraction")
    # Start class and start retrieving files to path

    if args.post:
        type_extract = "post_year"
    else:
        type_extract = "pre_year"

    ls = Landsat(
        aoi_path=wildfire_buffers,
        cloud_coverage=args.cloud_coverage,
        save_path=args.path_to_save,
        date_window=tuple([152, 273]),  # CA fire season period
        date_window_type=type_extract,
    )

    print("Start downloading process -- tqdm.tqdm should start soon")
    ls.execute_search_aoi(bbox_size=(0.75, 0.75))
