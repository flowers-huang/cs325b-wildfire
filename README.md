# Segmenting Wildfire Burned Area in the :earth_africa:

## Data Sources

Our data is from cropped [Landsat Collection 2][1] imagery for all
available fires in California. These labels are extracted from the [MTBS][2],
a project that measures severity and burned areas across the US. Notice that
each file name corresponds to the `Event_ID` in the MTBS dataset. The folder
contains `pre-image` and `post-images`, this is corresponds to data before and
after the fire event. Thus, you should see fire scars in all the `post-images`. 


Both `pre-` and `post-` imagery is captures during the fire-season (between day
152 and 273 of the year in California). Since Landsat has an average revisit day
between 8 and 16 days, many of the fires have several images. Notice that we
have visible bands and infrared bands to calculate different vegetation and
burning indexed, this might be relevant if you want to use this data for
pre-training a model from scratch, but feel free to ignore it if you prefer
fine-tune something. Native resolution for Landsat is 30 meters, which works
pretty well for some of the purposes ecologists are interested in, but not for
segmentation (a lot of the papers use Sentinel-2 data, 3 times higher
resolution, for example), we should test this.


[1]: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2#Example-Notebook
[2]: https://www.mtbs.gov/

## Environment
Results were obtained using python 3.9. The required packages can be installed with
```bash
pip install -r requirements.txt
```

## Data
Raw data can be downloaded by setting the root path defined in the bash script and running it
```bash
bash data_dowload_submission/get_data.sh
```
To download data for the scaled expiraments run
```bash
bash data_dowload_submission/get_data_scaled.sh
```

## Training a segmentation model
A unet model with resnet34 backbone can be trained by running
```bash
python train.py \
    --model_type "unet" \
    --backbone "resnet34" \
    --pretrained True \
    --device 0 \
    --num_workers 8 \
    --log_dir "~/logs" \
    --data_dir "~/data/processed" \
    --split 0.8 \
    --batch_size 32 \
    --crop_method "scale" \
    --crop_size 256 \
    --overlap 64 \
    --post_only False \
    --dnbr True
```
For a full set of available models and backbones see https://smp.readthedocs.io/en/latest/.
Results are saved to log_dir and can be visualized with tensorboard
```bash
tensorboard --logdir ~/logs
```
