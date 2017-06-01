#This looks through all the data and finds the number of craters in each image, as well as the max craters in a given image.
#When running this, probably a good idea to port (>) to a txt file, as it prints once per file.

import cv2
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

#####################
#load/read functions#
########################################################################
def get_im_cv2(path, img_width, img_height):
    img = cv2.imread(path)
    #resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR)
    return img

def load_data(path, data_type, img_width, img_height):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    max_N_craters = 0
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl,img_width,img_height)
        X.append(img)
        X_id.append(fl)
        N_craters = get_csv_len(fl)
        print "%d craters in file %s"%(N_craters[0],fl)
        y.append(N_craters)
        max_N_craters = np.max((max_N_craters, N_craters[0]))
    return  X, y, X_id, max_N_craters

def get_csv_len(file_):                        #returns # craters in each image (target)
    file2_ = file_.split('.png')[0] + '.csv'
    df = pd.read_csv(file2_ , header=0)
    return [len(df.index)]

img_width = 256              #image width
img_height = 256             #image height
#dir = '/scratch/k/kristen/malidib/moon/'
dir = ''
train_path, valid_path, test_path = '%sTrain_ds4_msk/'%dir, '%sDev_ds4_msk/'%dir, '%sTest_ds4_msk/'%dir
train_data, train_target, id, max_train_craters = load_data(train_path, 'train', img_width, img_height)
test_data, test_target, id, max_test_craters = load_data(valid_path, 'valid', img_width, img_height)
test_data, test_target, id, max_test_craters = load_data(test_path, 'test', img_width, img_height)
print "max train craters = %d, max test craters = %d"%(max_train_craters,max_test_craters)
