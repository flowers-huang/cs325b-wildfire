# Segmenting Wildfire Burned Area in the :earth_africa:

## Tentative data sources

Data in our GCP bucket has cropped [Landsat Collection 2][1] imagery for all
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

Data is in NetCDF format and notice that before using you should use the quality
assurance data to clean some pixels that aren't informative (i.e. snow or
clouds). If you feel you need more data, which I assume you will because is good
for out-of distribution (OOD) issues, you can use the code I added under
`src/landsat.py`. Remember California vegetation is pretty particular, and the
Mediterranean weather has climatic conditions that make the vegetation greener
and is dominated by conifers, so is not exactly the tropical Amazon. 



[1]: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2#Example-Notebook
[2]: https://www.mtbs.gov/
