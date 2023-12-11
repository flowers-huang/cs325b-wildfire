import os
import numpy as np
import rasterio
import glob
import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some paths.")

    # Add arguments
    parser.add_argument("--tif_dir", type=str, help="Directory containing TIFF files")
    parser.add_argument("--save_folder", type=str, help="Directory to save the output")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your script
    tif_dir = args.tif_dir
    save_folder = args.save_folder


    os.makedirs(save_folder, exist_ok=True)

    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # List all .tif files in the directory
    tif_files = glob.glob(os.path.join(tif_dir, "*.tif"))

    # Iterate over each .tif file
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            print(src.meta)
            affine = src.transform

            # Extract the base name of the file without the extension
            base_name = os.path.basename(tif_file).split('.')[0]

            # Save the affine matrix as a numpy array
            np.save(os.path.join(save_folder, f"{base_name}.npy"), affine)

            print(f"Affine matrix saved for {base_name}")

if __name__ == "__main__":
    main()
