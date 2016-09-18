
import subprocess
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sqlalchemy import create_engine
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
seed = 42

######################
### ML Preparation ###
######################

X, Y, L = load_processed_data(	X_filename = "../processed_data/images_RGB_121_91_expanded.npy",
								Y_filename= "../processed_data/prime_categories_expanded.npy",
								labels_filename = "../processed_data/labels_expanded.npy",
								randomize = False,
								target="prime_cat")
# print("imported")
# sys.exit()


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

x_train = x_train.astype("float32"); x_test = x_test.astype("float32"); x_valid = x_valid.astype("float32")
x_train=x_train/255; x_test=x_test/255; x_valid=x_valid/255
print("xtrain shape:",x_train.shape,"ytrain shape:",y_train.shape);print("xtest shape:",x_test.shape,"ytest shape:",y_test.shape);print("xvalid shape:",x_valid.shape,"yvalid shape:",y_valid.shape);

### Perform learning curve, collect model, and store performance

learn_sets = [100,250,500,750,1000,1500,2000,4000,y_train.shape[0]]

class_model = class_deep_net_model_1(	image_shape=(channels,img_rows,img_cols),
										loss="categorical_crossentropy")

trained_model, performance = learning_curve(model = class_model,
											x_train = x_train,
											y_train = y_train,
											x_test = x_test,
											y_test = y_test,
											learn_sets = learn_sets,
											save_directory = "../model_data/",
											target="prime_cat")

updateCSV(	data_to_update = performance,
			filename = "../model_data/classification_performance.csv")




	    

