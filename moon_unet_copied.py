#This is the unet model architechture applied on binary rings. 
#keras version 1.2.2.

import cv2
import os
import glob
import numpy as np
import pandas as pd
import random
from PIL import Image

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import AveragePooling2D, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.regularizers import l2
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

import utils.make_density_map_charles as mdm

#############################
#load/read/process functions#
########################################################################
def load_data(path, data_type):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        img = cv2.imread(f)/255.
        X.append(img)
        y.append(np.array(Image.open('%smask.tiff'%f.split('.png')[0])))
    return  X, y

def read_and_normalize_data(path, data_type):
    data, target = load_data(path, data_type)
    data = np.array(data).astype('float32')     #convert to numpy, convert to float
    target = np.array(target).astype('float32') #convert to numpy, convert to float
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

########################
#custom image generator#
########################################################################
#Following https://github.com/fchollet/keras/issues/2708
def custom_image_generator(data, target, batch_size=32):
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i+batch_size].copy(), target[i:i+batch_size].copy() #most efficient for memory?
            
            #horizontal/vertical flips
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])               #left/right
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])               #up/down
            
            #random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix,npix+1,batch_size)                  #horizontal shift
            v = np.random.randint(-npix,npix+1,batch_size)                  #vertical shift
            r = np.random.randint(0,4,batch_size)                           #90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix,npix),(npix,npix),(0,0)), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix,:] #RGB
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix]
                d[j], t[j] = np.rot90(d[j],r[j]), np.rot90(t[j],r[j])
            yield (d, t)

#############################
#unet model (keras 1.2.2)#
########################################################################
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Following https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def unet_model(im_width,im_height,learn_rate,init):
    print('Making unet model...')
    n_filters = 64      #vgg16 uses 64
    img_input = Input(batch_shape=(None, im_width, im_height, 1))

    inputs = Input(batch_shape=(None, im_width, im_height, 1))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=init)(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init=init)(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init=init)(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init=init)(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init=init)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init=init)(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init=init)(conv5)

    up6 = Deconvolution2D(256, 2, 2, output_shape=(None, 32, 32, 256), subsample=(2, 2), border_mode='same', init=init)(conv5)
    up6 = merge((up6, conv4), mode='concat', concat_axis=3)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init=init)(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init=init)(conv6)
    
    up7 = Deconvolution2D(128, 2, 2, output_shape=(None, 64, 64, 128), subsample=(2, 2), border_mode='same', init=init)(conv6)
    up7 = merge((up7, conv3), mode='concat', concat_axis=3)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init=init)(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init=init)(conv7)
    
    up8 = Deconvolution2D(64, 2, 2, output_shape=(None, 128, 128, 64), subsample=(2, 2), border_mode='same', init=init)(conv7)
    up8 = merge((up8, conv2), mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init=init)(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init=init)(conv8)
    
    up9 = Deconvolution2D(32, 2, 2, output_shape=(None, 256, 256, 32), subsample=(2, 2), border_mode='same', init=init)(conv8)
    up9 = merge((up9, conv1), mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=init)(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=init)(conv9)
    
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init=init)(conv9)
    model = Model(input=inputs, output=conv10)
    
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice_coef])  #binary cross-entropy severely penalizes opposite predictions.
    print model.summary()

    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_and_test_model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,n_train_samples,learn_rate,batch_size,nb_epoch,im_width,im_height,save_model,init):
    
    model = unet_model(im_width,im_height,learn_rate,init)
    
    model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=len(X_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
        
    if save_model == 1:
        model.save('models/unet_s256_rings_copy_%s.h5'%init)

    return model.evaluate(X_test.astype('float32'), Y_test.astype('float32'))

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,nb_epoch,n_train_samples,save_models,inv_color,rescale):
    #Static arguments
    im_width = 256              #image width
    im_height = 256             #image height
    
    #Load data
    dir = 'datasets/rings'
    try:
        train_data=np.load('%s/Train_rings/train_data.npy'%dir)
        train_target=np.load('%s/Train_rings/train_target.npy'%dir)
        valid_data=np.load('%s/Dev_rings/valid_data.npy'%dir)
        valid_target=np.load('%s/Dev_rings/valid_target.npy'%dir)
        test_data=np.load('%s/Test_rings/test_data.npy'%dir)
        test_target=np.load('%s/Test_rings/test_target.npy'%dir)
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        train_path, valid_path, test_path = '%s/Train_rings/'%dir, '%s/Dev_rings/'%dir, '%s/Test_rings/'%dir
        train_data, train_target = read_and_normalize_data(train_path, 'train')
        valid_data, valid_target = read_and_normalize_data(valid_path, 'validation')
        test_data, test_target = read_and_normalize_data(test_path, 'test')
        np.save('%s/Train_rings/train_data.npy'%dir,train_data)
        np.save('%s/Train_rings/train_target.npy'%dir,train_target)
        np.save('%s/Dev_rings/valid_data.npy'%dir,valid_data)
        np.save('%s/Dev_rings/valid_target.npy'%dir,valid_target)
        np.save('%s/Test_rings/test_data.npy'%dir,test_data)
        np.save('%s/Test_rings/test_target.npy'%dir,test_target)
    #Select desired subset number of samples, take first slice (saves memory) but keep data 3D.
    train_data  = train_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)
    train_target = train_target[:n_train_samples]
    valid_data = valid_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)
    valid_target = valid_target[:n_train_samples]
    test_data = test_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)
    test_target = test_target[:n_train_samples]

    #Invert image colors and rescale pixel values to increase contrast
    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        train_data = rescale_and_invcolor(train_data, inv_color, rescale)
        valid_data = rescale_and_invcolor(valid_data, inv_color, rescale)
        test_data = rescale_and_invcolor(test_data, inv_color, rescale)

    #Iterate
    N_runs = 5
    init = ['glorot_normal', 'he_uniform', 'he_normal', 'orthogonal', 'identity']
    #epochs = [15,20,25]
    for i in range(N_runs):
        I = init[i]
        score = train_and_test_model(train_data,train_target,valid_data,valid_target,test_data,test_target,n_train_samples,learn_rate,batch_size,nb_epoch,im_width,im_height,save_models,I)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, n_epoch=%d, n_train_samples=%d, im_width=%d, im_height=%d, inv_color=%d, rescale=%d, init=%s'%(learn_rate,batch_size,nb_epoch,n_train_samples,im_width,im_height,inv_color,rescale,I)
        print '###################################'
        print '###################################'

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    
    #args
    lr = 0.0001         #learning rate
    bs = 32             #batch size: smaller values = less memory but less accurate gradient estimate
    epochs = 5          #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 10080     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models
    inv_color = 1       #use inverse color
    rescale = 1         #rescale images to increase contrast (still 0-1 normalized)
    
    #run models
    run_cross_validation_create_models(lr,bs,epochs,n_train,save_models,inv_color,rescale)
