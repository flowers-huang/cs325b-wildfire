import os
import dask
import argparse
import rioxarray
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from src.dnbr import calculate_dnbr
from dask.distributed import Client


@dask.delayed
def lazy_saving(data, save_path):
    """Lazy array saving using the rio.xarray accessor"""

    data.rio.to_raster(save_path, tiled=True, windowed=True)

    return None


def batch_dnbr(
    path_pre, path_post, path_geom, path_save, buffer=True, buffer_size=None
):
    """Run the whole enchilada."""

    # Create saving dir if not exists
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # Define buffer size
    if not buffer:
        buffer_size = None

    geom = gpd.read_file(path_geom)
    wildfires = geom[
        (geom.Event_ID.str.contains("CA")) & (geom.Incid_Type == "Wildfire")
    ]

    wildfires_meters = wildfires.to_crs(3310)  # Meters in CA [NAD83]

    # List processed files to avoid repetition
    paths_pre_list = list(Path(path_pre).rglob("*.nc4"))
    paths_post_list = list(Path(path_post).rglob("*.nc4"))

    # Subset to events that are on both pre and post
    set_events = set([p.name for p in paths_pre_list]) & set(
        [p.name for p in paths_post_list]
    )

    lazy_arrays = []
    for event in tqdm(list(set_events), desc="Processing dnbr for events in MTBS"):
        raw_event = event.split(".")[0]
        file_name = f"{raw_event}.tif"
        save_path = os.path.join(path_save, file_name)

        if not os.path.exists(save_path):
            try:
                dnbr = calculate_dnbr(
                    pre_array=path_pre,
                    post_array=path_post,
                    geometry=wildfires_meters,
                    buffer_offset_size=buffer_size,
                    apply_qa_bitmask=True,
                    event_id=raw_event,
                )

                dnbr = lazy_saving(dnbr, save_path)
                lazy_arrays.append(dnbr)

            except OSError as e:
                print(f"File for {raw_event} not found!: {e}")
                pass

            except IndexError as e:
                print(f"{e}: {raw_event} it out of the collection date range")
                pass

            except ValueError as e:
                print("{e}: {raw_event} does not have data (?)")
                pass

    # Compute lazy objects via Dask. If client is available, computations will
    # be done with the Client.
    dask.compute([d.compute() for d in lazy_arrays])

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_pre", type=str, help="Path to pre-event images")
    parser.add_argument("--path_to_post", type=str, help="Path to post-event images")
    parser.add_argument(
        "--path_to_geom", type=str, help="Path to shapefile with events"
    )
    parser.add_argument("--path_to_save", type=str, help="Path to save dNBR rasters")
    parser.add_argument(
        "--buffer",
        type=int,
        default=180,
        help="Buffer size in meters for offset. Default is 180m",
    )
    parser.add_argument(
        "--offset",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Calculate dNBR with offset",
    )
    args = parser.parse_args()

    client = Client()
    print(f"Dask dashboard link: {client.dashboard_link}")
    batch_dnbr(
        path_pre=args.path_to_pre,
        path_post=args.path_to_post,
        path_geom=args.path_to_geom,
        path_save=args.path_to_save,
        buffer=args.offset,
        buffer_size=args.buffer,
    )
    client.close()
