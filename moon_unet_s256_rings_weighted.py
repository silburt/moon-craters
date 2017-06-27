######################
#MOON_UNET_S256_RINGS#
############################################
#This model:
#a) uses a custom loss (separately, i.e. *not* differentiable and guiding backpropagation) to assess how well our algorithm is doing, by connecting the predicted circles to the "ground truth" circles
#b) trained using the original LU78287GT.csv values as the ground truth,
#c) uses the Unet model architechture applied on binary rings.

#This model uses keras version 1.2.2.
############################################

import cv2
import os
import glob
import numpy as np
import pandas as pd
import random
from PIL import Image
from skimage.feature import match_template

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import AveragePooling2D, merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

import utils.make_density_map_charles as mdm
from utils.rescale_invcolor import *
from utils.template_match_target import *

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
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255.
        X.append(img)
        y.append(np.array(Image.open('%smask.tiff'%f.split('.png')[0])))
    return  X, y

def read_and_normalize_data(path, dim, data_type):
    data, target = load_data(path, data_type)
    data = np.array(data).astype('float32')             #convert to numpy, convert to float
    data = data.reshape(len(data),dim, dim, 1)          #add dummy third dimension, required for keras
    target = np.array(target).astype('float32')         #convert to numpy, convert to float
    print('%s shape:'%data_type, data.shape)
    return data, target

########################
#custom image generator#
########################################################################
#Following https://github.com/fchollet/keras/issues/2708
def custom_image_generator(data, target, batch_size=32):
    L, W = data[0].shape[0], data[0].shape[1]
    target = target.reshape(len(target),L,W)
    
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
            yield (d, t.reshape(batch_size, L*W))

#######################
#custom loss functions#
########################################################################
def prepare_custom_loss(path, dim):
    # hyperparameters - should not change
    minrad, maxrad = 2, 75    #min/max radius (in pixels) required to include crater in target
    cutrad = 0.5              #0-1 range, if x+cutrad*r > img_width, remove, i.e. exclude craters ~half gone from image
    min_craters = 5           #minimum craters in the image required for processing (make it worth your while)
    
    # load data
    try:
        imgs = np.load("%s/custom_loss_images.npy"%path)
        csvs = np.load("%s/custom_loss_csvs.npy"%path)
        N_perfect_matches = len(imgs)
        print "Successfully loaded files locally for custom_loss."
    except:
        print "Couldn't load files for custom_loss, making now"
        imgs, targets, csvs = [], [], []
        csvs_ = glob.glob('%s/*.csv'%path)
        N_perfect_matches = 0
        for c in csvs_:
            print "processing file %s"%c
            csv = pd.read_csv(c)
            img = cv2.imread('%s.png'%c.split('.csv')[0], cv2.IMREAD_GRAYSCALE)/255.
            
            # prune csv list for small/large/half craters
            csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
            csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
            csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
            if len(csv) < min_craters:
                print "only %d craters in image, skipping"%len(csv)
                continue
            
            # make target and csv array, ensure template matching algorithm is working - need Charles' ring routine
            target = mdm.make_mask(csv, img, binary=True, rings=True, ringwidth=2, truncate=True)
            csv_coords = np.asarray((csv['x'],csv['y'],csv['Diameter (pix)']/2)).T
            N_match, N_csv, N_templ, csv_duplicate_flag = template_match_target_to_csv(target, csv_coords, minrad, maxrad)
            if N_match == N_csv and csv_duplicate_flag == 0:
                imgs.append(img)
                targets.append(target)
                csvs.append(csv_coords)
                N_perfect_matches += 1
        imgs = np.array(imgs).astype('float32').reshape(len(imgs),dim,dim,1)
        targets = np.array(targets).astype('float32')
        np.save("%s/custom_loss_images.npy"%path,imgs)
        np.save("%s/custom_loss_csvs.npy"%path,csvs)
        print "out of %d files there are %d perfect matches"%(len(csvs_),N_perfect_matches)
    return imgs, csvs, N_perfect_matches

#for weighted binary cross-entropy - already reshaped
def get_class_weights(target):
    ones = float(len(np.where(target == 1)[0]))
    total = target.shape[0]*target.shape[1]
    
    weights = np.zeros((target.shape[0],target.shape[1]))
    weights[:,0] = ones*1./total
    weights[:,1] = (total-ones)*1./total
    return weights

##########################
#unet model (keras 1.2.2)#
########################################################################
#Following https://arxiv.org/pdf/1505.04597.pdf
#and this for merging specifics: https://gist.github.com/Neltherion/f070913fd6284c4a0b60abb86a0cd642
def unet_model(dim,learn_rate,lmbda,FL,init,n_filters):
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)

    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters*4, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters*2, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2,2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init, W_regularizer=l2(lmbda), border_mode='same')(u)

    #final output
    final_activation = 'sigmoid'       #sigmoid, relu
    u = Convolution2D(2, 1, 1, activation=final_activation, init=init, W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((dim*dim,2))(u)
    model = Model(input=img_input, output=u)
    
    #optimizer/compile
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  #binary cross-entropy severely penalizes opposite predictions.
    print model.summary()

    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_and_test_model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,loss_data,loss_csvs,dim,learn_rate,nb_epoch,batch_size,save_models,lmbda,FL,init,n_filters):
    model = unet_model(dim,learn_rate,lmbda,FL,init,n_filters)
    
    dim = Y_train.shape[1]
    Y_train = Y_train.reshape(len(Y_train),dim*dim)
    Y_valid = Y_valid.reshape(len(Y_valid),dim*dim)
    Y_test = Y_test.reshape(len(Y_test),dim*dim)
    
    n_samples = len(X_train)
    for nb in range(nb_epoch):
        model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                            samples_per_epoch=n_samples,nb_epoch=1,verbose=1,
                            #validation_data=(X_valid, Y_valid), #no generator for validation data
                            validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                            nb_val_samples=n_samples,
                            class_weight=get_class_weights(Y_train),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
                            
        # calcualte custom loss
        print ""
        print "custom loss for epoch %d/%d:"%(nb,nb_epoch)
        match_csv_arr, templ_csv_arr, templ_new_arr = [], [], []
        loss_target = model.predict(loss_data.astype('float32'))
        loss_target = loss_target.reshape(len(loss_target),dim,dim)
        
        for i in range(len(loss_data)):
            N_match, N_csv, N_templ, csv_duplicate_flag = template_match_target_to_csv(loss_target[i], loss_csvs[i])
            match_csv, templ_csv, templ_new = 0, 0, 0
            if N_csv > 0:
                match_csv = float(N_match)/float(N_csv)             #recall
                templ_csv = float(N_templ)/float(N_csv)             #craters detected/craters in csv
            if N_templ > 0:
                templ_new = float(N_templ - N_match)/float(N_templ) #fraction of craters that are new
            match_csv_arr.append(match_csv); templ_csv_arr.append(templ_csv); templ_new_arr.append(templ_new)
        print "mean and std of N_match/N_csv (recall) = %f, %f"%(np.mean(match_csv_arr), np.std(match_csv_arr))
        print "mean and std of N_template/N_csv = %f, %f"%(np.mean(templ_csv_arr), np.std(templ_csv_arr))
        print "mean and std of (N_template - N_match)/N_template (fraction of craters that are new) = %f, %f"%(np.mean(templ_new_arr), np.std(templ_new_arr))
        print ""

    if save_models == 1:
        model.save('models/unet_s256_rings_weighted.h5'%lmbda)

    return model.evaluate(X_test.astype('float32'), Y_test.astype('float32'))

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(dir,learn_rate,batch_size,nb_epoch,n_train_samples,save_models,inv_color,rescale):
    #Static arguments
    dim = 256              #image width/height, assuming square images. Shouldn't change
    
    #Load data
    try:
        train_data=np.load('%s/Train_rings/train_data.npy'%dir)
        train_target=np.load('%s/Train_rings/train_target.npy'%dir)
        valid_data=np.load('%s/Dev_rings/dev_data.npy'%dir)
        valid_target=np.load('%s/Dev_rings/dev_target.npy'%dir)
        test_data=np.load('%s/Test_rings/test_data.npy'%dir)
        test_target=np.load('%s/Test_rings/test_target.npy'%dir)
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        train_path, valid_path, test_path = '%s/Train_rings/'%dir, '%s/Dev_rings/'%dir, '%s/Test_rings/'%dir
        train_data, train_target = read_and_normalize_data(train_path, dim, 'train')
        valid_data, valid_target = read_and_normalize_data(valid_path, dim, 'dev')
        test_data, test_target = read_and_normalize_data(test_path, dim, 'test')
        np.save('%s/Train_rings/train_data.npy'%dir,train_data)
        np.save('%s/Train_rings/train_target.npy'%dir,train_target)
        np.save('%s/Dev_rings/dev_data.npy'%dir,valid_data)
        np.save('%s/Dev_rings/dev_target.npy'%dir,valid_target)
        np.save('%s/Test_rings/test_data.npy'%dir,test_data)
        np.save('%s/Test_rings/test_target.npy'%dir,test_target)
    #take desired subset of data
    train_data, train_target = train_data[:n_train_samples], train_target[:n_train_samples]
    valid_data, valid_target = valid_data[:n_train_samples], valid_target[:n_train_samples]
    test_data, test_target = test_data[:n_train_samples], test_target[:n_train_samples]

    #prepare custom loss
    custom_loss_path = '%s/Dev_rings_for_loss'%dir
    loss_data, loss_csvs, N_loss = prepare_custom_loss(custom_loss_path, dim)

    #Invert image colors and rescale pixel values to increase contrast
    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        train_data = rescale_and_invcolor(train_data, inv_color, rescale)
        valid_data = rescale_and_invcolor(valid_data, inv_color, rescale)
        test_data = rescale_and_invcolor(test_data, inv_color, rescale)
        loss_data = rescale_and_invcolor(loss_data, inv_color, rescale)

    ########## Parameters to Iterate Over ##########
#    N_runs = 6
#    filter_length = [3,3,3,3,3,3]   #See unet model. Filter length used.
#    n_filters = [64,64,64,64,64,64]     #See unet model. Arranging this so that total number of model parameters <~ 10M, otherwise OOM problems
#    lmbda = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5]           #See unet model. L2 Weight regularization strength (lambda).
    N_runs = 1
    filter_length=[3]
    n_filters=[64]
    lmbda=[0]
    I = 'he_normal'      #See unet model. Initialization of weights.

    #Iterate
    for i in range(N_runs):
        #I = init[i]
        NF = n_filters[i]
        FL = filter_length[i]
        L = lmbda[i]
        score = train_and_test_model(train_data,train_target,valid_data,valid_target,test_data,test_target,loss_data,loss_csvs,dim,learn_rate,nb_epoch,batch_size,save_models,L,FL,I,NF)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d, n_train_samples=%d, img_dimensions=%d, inv_color=%d, rescale=%d, init=%s, n_filters=%d, lambda=%e'%(learn_rate,batch_size,FL,nb_epoch,n_train_samples,dim,inv_color,rescale,I,NF,L)
        print '###################################'
        print '###################################'

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    
    #args
    dir = 'datasets/rings'  #location of Train_rings/, Dev_rings/, Test_rings/, Dev_rings_for_loss/ folders. Don't include final '/' in path
    lr = 0.0001             #learning rate
    bs = 32                 #batch size: smaller values = less memory but less accurate gradient estimate
    epochs = 4              #number of epochs. 1 epoch = forward/back pass through all train data
    n_train = 20000         #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1         #save models
    inv_color = 1           #use inverse color
    rescale = 1             #rescale images to increase contrast (still 0-1 normalized)
    
    #run models
    run_cross_validation_create_models(dir,lr,bs,epochs,n_train,save_models,inv_color,rescale)
