from scipy.misc import imread,imresize
from beau_cred import *
import pandas as pd
import os
import pandas as pd
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from keras import backend as K
from pandas import DataFrame,Series
from sqlalchemy import create_engine
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score,KFold,train_test_split
from sklearn.pipeline import Pipeline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils import np_utils

img_rows = 121
img_cols = 91
channels = 3

def convert_images_to_nparray(directory,img_rows,img_cols,channels=3):
	"""returns numpy arrays of labels and images """
	filelist = list(filter((".DS_Store").__ne__, os.listdir(directory)))
	if channels == 3:
		mode = "RGB"
	elif channels ==1:
		mode = "L"
	else:
		print("invalid channel #")
	labels = np.array([str.split(filelist[0],sep=".")[0]])	
	images = imread(name=directory+filelist[0],mode=mode)
	
	if images.shape[0] is not height or images.shape[1] is not width:
	        images = [imresize(images,(height,width,channels))]
	
	i=0

	for f in filelist[1:]:
	    temp_label = np.array([str.split(f,sep=".")[0]])
	    labels = np.concatenate((labels,temp_label),axis=0)
	    img = imread(name=dir+f,mode=mode)
	    if img.shape[0] is not height or img.shape[1] is not width:
	        img = imresize(img,(height,width,channels))
	    images = np.concatenate((images,[img]),axis=0)
	    i+=1
	    print (round(i/len(filelist)*100,2),"%",f,"completed")
	target = []
	for l in labels:
	    try:
	        result = int(data.loc[int(l)])
	    except:
	        result = 0
	    finally:
	        target.extend([result])
	print(completed)
	return (labels,images)
	print(images.shape)
	print(labels.shape)

# mines target from SQL product_info table

def collect_target(sql_user=sql_user,sql_pw=sql_pw,sql_host=sql_host,sql_db=sql_db,target="price_per_liter_in_cents"):
	engine = create_engine('postgresql://%s:%s@%s/%s' % (sql_user,sql_pw,sql_host,sql_db))
	data = pd.read_sql("SELECT DISTINCT product_no,%s FROM product_info" %target,con=engine)
	data.set_index("product_no",inplace=True)
	data = data.dropna()
	target = []
	for l in labels:
	    try:
	        result = int(data.loc[int(l)])
	    except:
	        result = 0
	    finally:
	        target.extend([result])
	        print(l,result)
	return target

def load_processed_data(X_filename,Y_filename,labels_filename,randomize=True):
	X = np.load(X_filename)
	Y = np.load(Y_filename)
	L = np.load(labels_filename)
	X = np.rollaxis(np.rollaxis(X,3),1) #converts axis to (n,channel,row,height)
	### remove null values that don't have an image
	
	L = L[Y!=0]
	X = X[Y!=0]
	Y = Y[Y!=0]*0.76/100 ### corrects for USD and cents
	###randomization
	smp = np.random.randint(low=0,high=Y.shape[0],size=Y.shape[0]).tolist()
	if randomize == True:
		L = L[smp]
		X = X[smp]
		Y = Y[smp]
	return (X,Y,L)


def reg_deep_net_model_1(image_shape,loss="mse",optimizer="adam"):	
	model = Sequential()
	model.add(Convolution2D(8,3,3,input_shape=image_shape,border_mode="same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.compile(loss=loss, optimizer='adam')
	return model

def mae_percentage(y_pred,y_true):
    diff = abs((y_true - y_pred) / y_true)
    return 100 * diff.mean()

def learning_curve(model,x_train,y_train,x_test,y_test,learn_sets,save_directory="../model_data"):
	rand_perf = []
	train_perf = []
	test_perf = []
	for i in learn_sets:
	    smp = np.random.randint(low=0,high=y_train.shape[0],size=i).tolist()
	    model.fit(	x_train[smp],
	    			y_train[smp],
	    			nb_epoch=20,
	    			batch_size=100,
	    			verbose=1,
	    			validation_data = (x_test,y_test))
	    train_perf.append(model.evaluate(x_train[smp],y_train[smp]))
	    test_perf.append(model.evaluate(x_test,y_test))
	    rand_perf.append(mae_percentage(bootstrap_resample(y_train),y_train))
	    learning_curve_performance = DataFrame([rand_perf,train_perf,test_perf,learn_sets],index=["random","train","test","random_samples"]).T
	    print(learning_curve_performance)
	    print(mae_percentage(model.predict(x_train),y_train))
	time_stamp = timenow()
	learning_curve_performance.insert(0,"time_stamp",timenow())
	model.save(save_directory+'beautai_algorithm_reg_expanded_ver_%s.h5' % time_stamp)
	return (model,learning_curve_performance)

def timenow(tz="America/Los_Angeles"):
    """get current time"""
    return datetime.datetime.now().replace(tzinfo=pytz.timezone(tz))

def updateCSV(data_to_update,filename):
    """creates or updates(ifexists) a file in csv format"""
    if os.path.isfile(filename)==True:
        with open(filename,'a') as f:
            data_to_update.to_csv(f,header=False,index=False)
    else:
        data_to_update.to_csv(filename,index=False)


def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
    	n = len(X)
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample