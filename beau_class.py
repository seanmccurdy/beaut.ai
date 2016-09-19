
import subprocess
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score,KFold,train_test_split
from sklearn.pipeline import Pipeline
from beau_cred import *
from beau_functions import *

#activates tensorflow backend for convnets
subprocess.call('KERAS_BACKEND=tensorflow python -c "from keras import backend; print (backend._BACKEND)"', shell=True)

os.chdir("/Users/seanmccurdy/Dropbox/insight_project/code")

img_rows = 121
img_cols = 91
channels = 3
image_shape = (channels,img_rows,img_cols)
seed = 42
target = sys.argv[1] ##alternatives are'prime_cat'(categorization)or'price' 
comments = ""
if len(sys.argv)>1:
	comments = sys.argv[2]

### ^^^ performs regression ##takes from command line

######################
### ML Preparation ###
######################

if target == "prime_cat":
	target_import = "../processed_data/prime_categories_expanded.npy"
elif target == "price":
	target_import = "../processed_data/target_expanded.npy"

X, Y_prime, L = load_processed_data(	X_filename = "../processed_data/images_RGB_121_91_expanded.npy",
										Y_filename = target_import,
										labels_filename = "../processed_data/labels_expanded.npy",
										randomize = False,
										target=target)

if target == "prime_cat":
	Y = np_utils.to_categorical(Y_prime[0],len(Y_prime[1]))
elif target =="price":
	Y = Y_prime

#create train/test/valid splits
x_train, x_test, y_train, y_test = train_test_split(	X,
														Y,
														test_size=0.4,
														random_state=seed)

x_test, x_valid, y_test, y_valid = train_test_split(	x_test,
														y_test,
														test_size=0.50,
														random_state=seed)

### format and normalize images and check shapes prior to ML
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_valid = x_valid.astype("float32")
x_train /=255; x_test/=255; x_valid/=255
print("xtrain shape:",x_train.shape,"ytrain shape:",y_train.shape);print("xtest shape:",x_test.shape,"ytest shape:",y_test.shape);print("xvalid shape:",x_valid.shape,"yvalid shape:",y_valid.shape);

### Perform learning curve, collect model, and store performance

learn_sets = [100,250,500,750,1000,1500,2000,4000,y_train.shape[0]]

learning_curve(	x_train = x_train,
				y_train = y_train,
				x_test = x_test,
				y_test = y_test,
				learn_sets = learn_sets,
				save_directory = "../model_data/",
				target = target,
				comments = comments,
				hot_one_keys = Y_prime[1])


