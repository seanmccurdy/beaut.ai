import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from beau_functions import *

img_rows = 121
img_cols = 91
channels = 3
seed = 42


########################
### Image Processing ###
########################

###Convert scrape and process image and target data for ML analysis

labels,images = convert_images_to_nparray(	directory = "../thumbnails",
											img_rows = img_rows,
											img_cols = img_cols,
											channels = 3)

target = collect_target(target="price_per_liter_in_cents")


### save numpy arrays for future usage
np.save("../processed_data/images_RGB_%d_%d_expanded.npy" % (img_rows,img_cols),images)
np.save("../processed_data/target_expanded.npy",target)
np.save("../processed_data/labels_expanded.npy",labels)
