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
#Following https://arxiv.org/pdf/1505.04597.pdf
#and this for merging specifics: https://gist.github.com/Neltherion/f070913fd6284c4a0b60abb86a0cd642
def unet_model(im_width,im_height,learn_rate,lmbda,FL):
    print('Making VGG16-style Fully Convolutional Network model...')
    n_filters = 32      #vgg16 uses 64
    img_input = Input(batch_shape=(None, im_width, im_height, 1))
    
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
    u = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((im_width, im_height))(u)
    model = Model(input=img_input, output=u)
    
    #optimizer/compile
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print model.summary()

    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_and_test_model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,n_train_samples,learn_rate,batch_size,lmbda,FL,nb_epoch,im_width,im_height,save_model):
    
    model = unet_model(im_width,im_height,learn_rate,lmbda,FL)
    
    model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=len(X_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
        
    if save_model == 1:
        model.save('models/unet_s256_rings_FL%d.h5'%FL)

    return model.evaluate(X_test.astype('float32'), Y_test.astype('float32'))

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,lmbda,nb_epoch,n_train_samples,save_models,inv_color,rescale):
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
    train_data  = train_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)  #keep 3D
    train_target = train_target[:n_train_samples]
    valid_data = valid_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)
    valid_target = valid_target[:n_train_samples]
    test_data = test_data[:n_train_samples,:,:,0].reshape(n_train_samples,im_width,im_height,1)
    test_target = test_target[:n_train_samples]

    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        train_data = rescale_and_invcolor(train_data, inv_color, rescale)
        valid_data = rescale_and_invcolor(valid_data, inv_color, rescale)
        test_data = rescale_and_invcolor(test_data, inv_color, rescale)

    #Iterate
    N_runs = 1
    filter_length = [10]
    #lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1); lmbda.append(0)
    #epochs = [15,20,25]
    for i in range(N_runs):
        FL = filter_length[i]
        l=0
        score = train_and_test_model(train_data,train_target,valid_data,valid_target,test_data,test_target,n_train_samples,learn_rate,batch_size,l,FL,nb_epoch,im_width,im_height,save_models)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d, n_train_samples=%d, im_width=%d, im_height=%d, inv_color=%d, rescale=%d'%(learn_rate,batch_size,FL,nb_epoch,n_train_samples,im_width,im_height,inv_color,rescale)
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
    epochs = 2          #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 10080     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models
    inv_color = 1       #use inverse color
    rescale = 1         #rescale images to increase contrast (still 0-1 normalized)
    
    #run models
    run_cross_validation_create_models(lr,bs,lmbda,epochs,n_train,save_models,inv_color,rescale)
