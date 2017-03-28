import cv2
import os
import glob
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

learning_rate = 0.0001
img_width = 224
img_height = 224

#####################
#load/read functions#
########################################################################
#load/read files
def get_im_cv2(path):
    img = cv2.imread(path)
    #resized = cv2.resize(img, (32, 32))#, cv2.INTER_LINEAR)
    resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR)
    return resized

def load_data(path, data_type):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X.append(img)
        X_id.append(fl)
        y.append(get_csv_len(fl))
    return  X, y, X_id

def read_and_normalize_data(path):
    if 'train' in path:
        data_type = 'train'
    elif 'test' in path:
        data_type = 'test'
    data, target, id = load_data(path, data_type)
    data = np.array(data, dtype=np.uint8)      #convert to numpy
    target = np.array(target, dtype=np.uint8)
    data = data.astype('float32')              #convert to float
    data = data / 255
    print('%s shape:'%data_type, data.shape)
    return data, target, id

def get_csv_len(file_):                        #previously y_trainn2
    file2_ = file_.split('.png')[0] + '.csv'
    df = pd.read_csv(file2_ , header=0)
    return [len(df.index)]

#######
#vgg16#
########################################################################
#Following https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
def vgg16(n_classes,im_width,im_height):
    model = Sequential()
    n_filters = 32          #vgg16 uses 64
    n_blocks = 3            #vgg16 uses 5
    n_dense = 512           #vgg16 uses 4096

    #first block
    model.add(Conv2D(n_filters, (3, 3), activation='relu', padding='same',input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))          #maximum of 512 filters in vgg16
        model.add(Conv2D(n_filters_, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(n_filters_, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

