import glob
import cv2
import os
import glob
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

def get_im_cv2(path, img_width, img_height):
    img = cv2.imread(path)
    return img

def load_data(path, data_type, img_width, img_height):
    X, X_id, y = [], [], []
    minpix = 2                                  #minimum number of pixels for crater to count
    files = glob.glob('%s*.png'%path)
    minpix, maxpix = 2, 100                          #minimum pixels required for a crater to register in an image
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        flbase = os.path.basename(f)
        img = get_im_cv2(f,img_width,img_height) / 255.
        X.append(img)
        y.append(np.array(Image.open('%smask.tiff'%f.split('.png')[0])))
    return  X, y, X_id

def read_and_normalize_data(path, img_width, img_height, data_flag):
    if data_flag == 0:
        data_type = 'train'
    elif data_flag == 1:
        data_type = 'test'
    data, target, id = load_data(path, data_type, img_width, img_height)
    data = np.array(data).astype('float32')     #convert to numpy, convert to float
    target = np.array(target).astype('float32')
    print('%s shape:'%data_type, data.shape)
    return data, target, id

#save target no border for Hough circles
def save_image(data, cm, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cm)
    plt.savefig(fn, dpi = height)

def rescale_and_invcolor(data, inv_color, rescale):
    for img in data:
        if inv_color == 1:
            img[img > 0.] = 1. - img[img > 0.]
        if rescale == 1:
            minn, maxx = np.min(img[img>0]), np.max(img[img>0])
            low, hi = 0.1, 1                                                #low, hi rescaling values
            img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    return data

#dice coefficient
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#load data
dim, inv_color, rescale = 256, 1, 1
test_data, test_target, test_id = read_and_normalize_data('datasets/rings/Test_rings_sample/', dim, dim, 1)

#reshape
test_data = test_data[:,:,:,0].reshape(len(test_data),256,256,1)

#invcolor and rescale
if inv_color==1 or rescale==1:
    test_data = rescale_and_invcolor(test_data, inv_color, rescale)

filename = 'models/unet_s256_rings_copy_glorot_normal.h5'
model = load_model(filename, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

print "loaded everything successfully, generating predictions"
n,off=32,0
target = model.predict(test_data[off:(n+off)].astype('float32'))

name = os.path.basename(filename).split('h5')[0]
np.save('datasets/rings/Test_rings_sample/%s_pred.npy'%name,target)
print "successfully generated predictions at datasets/rings/Test_rings_sample/%s_pred.npy"%name
