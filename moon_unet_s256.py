#s256: small images - using the new 256x256 images put together by Kristen.
#Fork: From my pure skip connection model I'm noticing that the small craters are being captured nicely, but the large craters are not being recognized. So, I need a separate fork on the onset with a large receptive field to capture the large craters as well.
#Skip: This model uses skip connections to merge the where with the what, and have scale aware analysis.
#See "Residual connection on a convolution layer" in https://jtymes.github.io/keras_docs/1.2.2/getting-started/functional-api-guide/#multi-input-and-multi-output-models

#This python script is adapted from moon2.py and uses the vgg16 convnet structure.
#The number of blocks, and other aspects of the vgg16 model can be modified.
#This has the keras 1.2.2. architechture

import os
import glob
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import AveragePooling2D, merge, Input, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, AtrousConvolution2D
from keras.regularizers import l2
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

import utils.make_density_map as mdm

###########
#load data#
########################################################################
def load_data(n_train_samples,im_width,im_height,inv_color_flag):
    train_data=np.load('training_set/lola_0_input.npy')[:n_train_samples].astype('float32')
    train_target=np.load('training_set/lola_0_targets.npy')[:n_train_samples].astype('float32')
    test_data=np.load('test_set/lola_1_input.npy')[:n_train_samples].astype('float32')
    test_target=np.load('test_set/lola_1_targets.npy')[:n_train_samples].astype('float32')
    print "Successfully loaded files locally."
    
    save_sample = 0
    if save_sample == 1:
        np.save('training_set/lola_0_input_sample.npy',train_data[0:50])
        np.save('training_set/lola_0_targets_sample.npy',train_target[0:50])
        np.save('test_set/lola_1_input_sample.npy',test_data[0:50])
        np.save('test_set/lola_1_targets_sample.npy',test_target[0:50])
    
    #norm data
    train_data /= 255
    test_data /= 255
    
    #increase contrast of data, inverse color if flag=1
    low, hi = 0.1, 1                                                    #low, hi rescaling values
    for img in train_data:
        if inv_color_flag == 1:
            img[img > 0.] = 1. - img[img > 0.]
        minn, maxx = np.min(img[img>0]), np.max(img[img>0])
        img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    
    for img in test_data:
        if inv_color_flag == 1:
            img[img > 0.] = 1. - img[img > 0.]
        minn, maxx = np.min(img[img>0]), np.max(img[img>0])
        img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    
    #reshape data to 3D array
    train_data = np.reshape(train_data, (len(train_data),im_width,im_height,1))
    test_data = np.reshape(test_data, (len(test_data),im_width,im_height,1))
    
    #norm train targets
    for i in range(len(train_target)):
        maxx = np.max(train_target[i])
        if maxx > 0:
            train_target[i] /= maxx

    #norm test targets
    for i in range(len(test_target)):
        maxx = np.max(test_target[i])
        if maxx > 0:
            test_target[i] /= maxx

    print "Successfully normalized data and target."
    return train_data, train_target, test_data, test_target


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
            npix = 10
            h = np.random.randint(-npix,npix+1,batch_size)                  #horizontal shift
            v = np.random.randint(-npix,npix+1,batch_size)                  #vertical shift
            r = np.random.randint(0,4,batch_size)                           #90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix,npix),(npix,npix),(0,0)), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix,:]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix]
                d[j], t[j] = np.rot90(d[j],r[j]), np.rot90(t[j],r[j])
            
            yield (d, t)

#############################
#FCN vgg model (keras 1.2.2)#
########################################################################
#Following https://github.com/aurora95/Keras-FCN/blob/master/models.py
#and also loosely following: https://blog.keras.io/building-autoencoders-in-keras.html
#and maybe: https://github.com/nicolov/segmentation_keras
#and this!: https://gist.github.com/Neltherion/f070913fd6284c4a0b60abb86a0cd642
#DC: https://arxiv.org/pdf/1511.07122.pdf
def unet_model(im_width,im_height,learn_rate,lmbda,FL):
    print('Making VGG16-style Fully Convolutional Network model...')
    n_filters = 32      #vgg16 uses 64
    #FL = 12            #Receptive Field
    img_input = Input(batch_shape=(None, im_width, im_height, 1))
    
    #model a - small receptive field for small craters
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv1_a1', border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv1_a2', border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_a1')(a1)
    
    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv2_a1', border_mode='same')(a1P)
    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv2_a2', border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_a1')(a2)
    
    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv3_a1', border_mode='same')(a2P)
    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv3_a2', border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_a1')(a3)
    
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv4_a1', border_mode='same')(a3P)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv4_a2', border_mode='same')(u)
    
    u = UpSampling2D((2,2), name='up4->3')(u)
    u = merge((a3, u), mode='concat', name='merge3')
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge3_1', border_mode='same')(u)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge3_2', border_mode='same')(u)
    
    u = UpSampling2D((2,2), name='up3->2')(u)
    u = merge((a2, u), mode='concat', name='merge2')
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge2_1', border_mode='same')(u)
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge2_2', border_mode='same')(u)
    
    u = UpSampling2D((2,2), name='up2->1')(u)
    u = merge((a1, u), mode='concat', name='merge1')
    u = Convolution2D(n_filters, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge1_1', border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', W_regularizer=l2(lmbda), name='conv_merge1_2', border_mode='same')(u)
    
    #final output
    #final conv layer used to be 3x3, but now 1x1.
    u = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((im_width, im_height))(u)
    model = Model(input=img_input, output=u)
    
    #optimizer/compile
    #optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(loss='mse', optimizer=optimizer)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print model.summary()
    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_and_test_model(train_data,train_target,test_data,test_target,n_train_samples,learn_rate,batch_size,lmbda,FL,nb_epoch,im_width,im_height,rs,save_model,inv_color):
    
    #Main Routine - Build/Train/Test model
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.20, random_state=rs)
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    
    model = unet_model(im_width,im_height,learn_rate,lmbda,FL)
    
    model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=len(X_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
              
    if save_model == 1:
        model.save('models/unet_s256_FL%d_invc%d.h5'%(FL,inv_color))
     
    test_pred = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    npix = test_target.shape[0]*test_target.shape[1]*test_target.shape[2]
    return np.sum((test_pred - test_target)**2)/npix    #calculate test score

'''
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
'''

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,lmbda,nb_epoch,n_train_samples,save_models,inv_color):
    #Static arguments
    im_width = 256              #image width
    im_height = 256             #image height
    rs = 42                     #random_state for train/test split

    #Load data
    train_data, train_target, test_data, test_target = load_data(n_train_samples,im_width,im_height,inv_color)

    #Iterate
    N_runs = 3
    #lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1); lmbda.append(0)
    filter_length = [3,10,15]
    for i in range(N_runs):
        FL = filter_length[i]
        l = 0
        score = train_and_test_model(train_data,train_target,test_data,test_target,n_train_samples,learn_rate,batch_size,l,FL,nb_epoch,im_width,im_height,rs,save_models,inv_color)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d, n_train_samples=%d, inv_color=%d, random_state=%d, im_width=%d, im_height=%d'%(learn_rate,batch_size,FL,nb_epoch,n_train_samples,inv_color,rs,im_width,im_height)
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
    lmbda = 0           #L2 regularization strength (lambda)
    epochs = 8          #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 10080      #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models
    inv_color = 0

    #run models
    run_cross_validation_create_models(lr,bs,lmbda,epochs,n_train,save_models,inv_color)


