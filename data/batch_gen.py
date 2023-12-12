import rasterio
import rioxarray
import xarray as xr
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import os
import dask
import rioxarray
import numpy as np
import pandas as pd
import warnings
from itertools import combinations
import pystac
import planetary_computer
import torch
import torchdata
import zen3geo
import requests
import zipfile
from torch.utils.data import Dataset, DataLoader

'''
NUMBERS TO ADJUST: 
'''
BATCH_SIZE = 64
NUM_EPOCHS = 3
root_dir = Path("/burned_area/")



'''
FILE ACCESS 
'''

pre_root_dir = root_dir / "pre-images"
post_root_dir = root_dir / "post-images"
impost_root_dir = root_dir / "inmediate-post-fire-images"
processed_root_dir = root_dir / "processed_geotiff"
# root_dir = Path("/content/data/post-images")
files_pre = set(os.listdir(pre_root_dir))
files_post = set(os.listdir(post_root_dir))
files_impost = set(os.listdir(impost_root_dir))
files_proc = set(os.listdir(processed_root_dir))

files_pre = {Path(f).stem: pre_root_dir / f for f in files_pre}
files_post = {Path(f).stem: post_root_dir / f for f in files_post}
files_impost = {Path(f).stem: impost_root_dir / f for f in files_impost}
files_proc = {Path(f).stem: processed_root_dir / f for f in files_proc}

shared_files = list(files_pre.keys() & files_post.keys() & files_impost.keys() & files_proc.keys())

print("FOUND", len(shared_files), "SHARED FILES...")


'''
SETTING UP CUSTOM DATASET FOR IMAGE PROCESSING
'''

class CustomDataset(Dataset):
    def __init__(self, files_pre, files_post, labels):
      self.data = []
      self.labels = labels
      self.label_count = [0, 0]

      for path_name in files_pre:
          #pre_root_dir = root_dir / "pre-images"
          self.data.append([pre_root_dir / (path_name + ".nc4"), "pre"])
          self.label_count[0] += 1

      for path_name in files_post:
          self.data.append([post_root_dir / (path_name + ".nc4"), "post"])
          self.label_count[1] += 1

      print("Collected", self.label_count, "images")


    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      data = self.data[idx][0]
      label = self.data[idx][1]

      #print(label, idx, data)

      ds = (
          xr.open_dataset(data)
          .to_array() # transform to an array rather than the xr.Dataset
          .squeeze()  # just like in numpy, remove all singletons
      )

      #print(ds)

      fixed_banding = ds.median(dim="time").sel(band=["red", "green", "blue", 'nir08', 'swir16', 'qa_pixel']).to_numpy()



      #sample = {"image": ds.to_numpy(), "class": label}
      return fixed_banding, label


'''
Generate image dataset obj
'''

labels = ["pre", "post"]
image_dataset = CustomDataset(files_pre, files_post, labels)

'''
New collate function -- stacks dimensions
'''

def collate_fn(batch):
    zipped = zip(batch)
    return list(zipped)


# generates a new dataloader object
image_dl = DataLoader(image_dataset, collate_fn=collate_fn, batch_size = BATCH_SIZE, shuffle=True)


for epoch in range(NUM_EPOCHS):
    print("\n==============================\n")
    print("Epoch = " + str(epoch))
    for (idx, batch) in enumerate(image_dl):
      print("\nBatch = " + str(idx))