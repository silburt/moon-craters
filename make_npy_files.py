"""
Charles's attempt at creating a convnet that translates an image
into a density map, which can then be postprocessesed to give a
crater count.
"""

################ IMPORTS ################

# Past-proofing
from __future__ import absolute_import, division, print_function

# System modules
import os
import sys
import glob
#import cv2
import datetime

# I/O and math stuff
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append("/home/m/mhvk/czhu/moon_craters")
import make_density_map as densmap

# NN and CV stuff
from sklearn.model_selection import KFold, train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras
#from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as kpimg

################ DATA READ-IN FUNCTIONS (FROM moon4.py and moon_vgg16_1.2.2.py) ################

def get_im_csv(path, imgshp, pix_cut=0):
    """Grabs image (greyscale) using PIL.Image, converts it to
    np.array, then grabs craters as pd.DataFrame
    """
    img = Image.open(path).convert('L')
    img = np.asanyarray(img.resize(imgshp))
    craters = pd.read_csv(path.split(".png")[0] + ".csv")
    if pix_cut:
        craters.drop( np.where(craters["Diameter (pix)"] < pix_cut)[0], 
                            inplace=True )
        craters.reset_index(drop=True, inplace=True)
    return img, craters


def load_data(path, imgshp, pix_cut=0):
    """Chain-loads data.
    """
    X = []
    X_id = []
    ctrs = []
    files = glob.glob('%s*.png'%path)
    print("number of files: %d"%(len(files)))
    for fl in files:
        flbase = os.path.basename(fl)
        img, craters  = get_im_csv(fl, imgshp, pix_cut=pix_cut)
        X.append(img)
        X_id.append(fl)
        ctrs.append(craters)
    return X, ctrs, X_id


def read_and_normalize_data(path, imgshp, pix_cut=0, data_flag="this"):
    """Reads and normalizes input data.  Removes craters below some
    minimum size.
    """
    print("For {0:s} data".format(data_flag))
    X, ctrs, X_id = load_data(path, imgshp, pix_cut=pix_cut)
    # Convert to np.array and normalize
    X = np.array(X, dtype=np.float32) / 255.
    print('Shape:', X.shape)
    return X, ctrs, X_id


def 


    # Try to load data from working directory
    # The .npy stuff doesn't work for pandas dataframes
#    try:
#        train_data = np.load(args["path"] + '/training_set/train_data.npy')
#        train_ctrs = np.load(args["path"] + '/training_set/train_ctrs.npy')
#        test_data = np.load(args["path"] + '/test_set/test_data.npy')
#        test_ctrs = np.load(args["path"] + '/test_set/test_ctrs.npy')
#        print("Successfully loaded .npy files from working directory.")
#    except:
    print("Can't find .npy files locally; reading in from args.path.")
    train_data, train_ctrs, train_id = \
                read_and_normalize_data(args["path"] + "/training_set/", 
                                        args, "train")
    test_data, test_ctrs, test_id = \
                read_and_normalize_data(args["path"] + "/test_set/", 
                                        args, "test")
#        np.save(args["path"] + '/training_set/train_data.npy', train_data)
#        np.save(args["path"] + '/training_set/train_ctrs.npy', train_ctrs)
#        np.save(args["path"] + '/test_set/test_data.npy', test_data)
#        np.save(args["path"] + '/test_set/test_ctrs.npy', test_ctrs)

    # Calculate next largest multiple of batchsize to N_train*f_samp
    # Then use to obtain subset
    N_sub = int(args["batchsize"] * np.ceil( train_data.shape[0] * \
                                        args["f_samp"] / args["batchsize"] ))
    subset = np.random.choice(train_data.shape[0], size=N_sub)
    train_data = train_data[subset]
    train_target = train_target[subset]












