import os
import dask
import argparse
import rioxarray
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from dask.distributed import Client
from composite import calculate_composite
import sys
sys.path.append('/lfs/turing3/0/kaif/GitHub/burned')

def batch_composite(path_to_data, path_save):
    """Run the whole enchilada."""

    # Create saving dir if not exists
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    path_list = list(Path(path_to_data).rglob("*.nc4"))

    lazy_arrays = []
    for event in tqdm(path_list, desc="Processing dnbr for events in MTBS"):
        raw_event = event.stem
        file_name = f"{raw_event}.tif"
        save_path = os.path.join(path_save, file_name)

        if not os.path.exists(save_path):
            try:
                composite = calculate_composite(
                    path_to_array=event,
                    apply_qa_bitmask=True,
                    save=save_path
                )

                lazy_arrays.append(composite)

            except OSError as e:
                print(f"File for {raw_event} not found!: {e}")
                pass

            except IndexError as e:
                print(f"{e}: {raw_event} it out of the collection date range")
                pass

            # except ValueError as e:
            #     print("{e}: {raw_event} does not have data (?)")
            #     pass

    # Compute lazy objects via Dask. If client is available, computations will
    # be done with the Client.
    dask.compute([d.compute() for d in lazy_arrays])

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, help="Path to post-event images")
    parser.add_argument("--path_to_save", type=str, help="Path to save dNBR rasters")
    args = parser.parse_args()

    client = Client()
    print(f"Dask dashboard link: {client.dashboard_link}")
    batch_composite(
        path_to_data=args.path_to_data,
        path_save=args.path_to_save,
    )
    client.close()
