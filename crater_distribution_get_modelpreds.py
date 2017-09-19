# This code is typically run on p8t03/04 on scinet.
# Just for extracting the predictions.

import numpy as np
import cv2
import glob
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

from utils.rescale_invcolor import *
from utils.template_match_target import *

def load_data(path):
    X = []
    X_id = []
    files = glob.glob('%s/*.png'%path)
    print "number of files are: %d"%len(files)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255.
        X.append(img)
        X_id.append(int(os.path.basename(f).split('_')[1].split('.png')[0]))
    return  X, X_id

def read_and_normalize_data(path, dim):
    data, id_ = load_data(path)
    data = np.array(data).astype('float32')             #convert to numpy, convert to float
    data = data.reshape(len(data),dim, dim, 1)          #add dummy third dimension, required for keras
    print 'shape:', data.shape
    return data, id_

def get_modelpreds(data_dir,data_prefix,csv_prefix,model_loc,n_imgs,inv_color,rescale):
    # get data
    try:
        data=np.load('%s/%s.npy'%(data_dir,data_prefix))/255.   #for Charles' highilen dataset
        #data=np.load('%s/%s_data.npy'%(data_dir,data_prefix))
        data = data[:n_imgs]
        print "Successfully loaded %s files locally."%data_dir
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%data_dir
        data, id = read_and_normalize_data(data_dir, dim)
        np.save('%s/%s_data.npy'%(data_dir,data_prefix),data)
        np.save('%s/%s_id.npy'%(data_dir,data_prefix),id)
        data, id = data[:n_imgs], id[:n_imgs]

    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        data = rescale_and_invcolor(data, inv_color, rescale)
    
    # generate model predictions
    model = load_model(model_loc)
    pred = model.predict(data.astype('float32'))
    np.save('%s/%s_modelpreds_n%d_new.npy'%(data_dir,data_prefix,n_imgs),pred)
    print "generated and saved predictions"

if __name__ == '__main__':
    #args
    # ilen_1500_to_2500 settings - from Mohamad
#    data_dir = 'datasets/ilen_1500_to_2500/ilen_2000'       #location of data to predict on. Exclude final '/' in path.
#    data_prefix = ''                                        #prefix of e.g. *_data.npy files.
#    csv_prefix = ''                                         #prefix of e.g. *_0001.csv files.
#    model_loc = 'models/unet_s256_rings_nFL96.h5'
#    n_imgs = 1000

    # Charles highilen augmented dataset
    data_dir = 'datasets/highilen'                          #location of data to predict on. Exclude final '/' in path.
    data_prefix = 'highilen_train_input_n5000'              #prefix of e.g. *_data.npy files.
    csv_prefix = ''                                         #prefix of e.g. *_0001.csv files.
    model_loc = 'models/unet_s256_rings_nFL96.h5'
    n_imgs = 5000

#    data_dir = 'datasets/rings/Dev_rings'                  #location of data to predict on. Exclude final '/' in path.
#    data_prefix = 'dev'                                    #prefix of e.g. *_data.npy files.
#    csv_prefix = 'lola'                                     #prefix of e.g. *_0001.csv files.
#    model_loc = 'models/unet_s256_rings_nFL96.h5'
#    n_imgs = 30016          #number of images to use for getting crater distribution.

    inv_color = 1           #**must be same setting as what model was trained on**
    rescale = 1             #**must be same setting as what model was trained on**
    get_modelpreds(data_dir,data_prefix,csv_prefix,model_loc,n_imgs,inv_color,rescale)

    print "Script completed successfully"
