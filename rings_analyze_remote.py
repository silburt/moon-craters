import glob
import cv2
import os
import glob
import numpy as np
import pandas as pd
import sys
from PIL import Image
from keras.models import load_model
from keras import backend as K

def load_data(path, data_type):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255.      #(im_width,im_height)
        X.append(img)
        y.append(np.array(Image.open('%smask.tiff'%f.split('.png')[0])))
    return  X, y

def read_and_normalize_data(path, im_width, im_height, data_type):
    data, target = load_data(path, data_type)
    data = np.array(data).astype('float32')             #convert to numpy, convert to float
    data = data.reshape(len(data),im_width,im_height,1) #add dummy third dimension, required for keras
    target = np.array(target).astype('float32')         #convert to numpy, convert to float
    print('%s shape:'%data_type, data.shape)
    return data, target

#experimenting with bigger contrast
#https://www.mathworks.com/help/vision/ref/contrastadjustment.html
#Since maxpooling is used, we want the interesting stuff (craters) to be 1, not 0.
#But ignore null background pixels, keep them at 0.
def rescale_and_invcolor(data, inv_color, rescale):
    for img in data:
        if inv_color == 1:
            img[img > 0.] = 1. - img[img > 0.]
        if rescale == 1:
            minn, maxx = np.min(img[img>0]), np.max(img[img>0])
            low, hi = 0.1, 1                                                #low, hi rescaling values
            img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    return data

#load data
dim, inv_color, rescale = 256, 1, 1
test_data, test_target = read_and_normalize_data('datasets/rings/Test_rings_sample/', dim, dim, 'test')

#reshape
test_data = test_data[:,:,:,0].reshape(len(test_data),dim,dim,1)

#invcolor and rescale
if inv_color==1 or rescale==1:
    test_data = rescale_and_invcolor(test_data, inv_color, rescale)

#list of models you want predictions on
models = ['unet_s256_rings_FL5_he_uniform.h5','unet_s256_rings_FL5_glorot_normal.h5','unet_s256_rings_FL3_he_normal.h5','unet_s256_rings_FL3_he_uniform.h5','unet_s256_rings_FL3_glorot_normal.h5']

n,off=20,0
print "begin generating predictions"
for m in models:
    model = load_model(m)
    target = model.predict(test_data[off:(n+off)].astype('float32'))
    name = os.path.basename(filename).split('.h5')[0]
    
    #dimensions go data, ground_truth targets, predicted targets
    arr = np.concatenate((test_data[off:(n+off)],test_target[off:(n+off)].reshape(n,dim,dim,1),target.reshape(n,dim,dim,1)),axis=3)
    np.save('models/%s_pred.npy'%name,target)
    print "successfully generated predictions at models/%s_pred.npy for model %s"%(name,m)
