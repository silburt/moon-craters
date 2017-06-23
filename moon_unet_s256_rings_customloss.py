#This uses a custom loss (separately, i.e. *not* differentiable and guiding backpropagation) to assess how well our algorithm is doing, by connecting the predicted circles to the "ground truth" circles
#This model is trained using the original LU78287GT.csv values as the ground truth,
#and then predictions from this model are used as the new ground truth for moon_unet_s256_rings_pred.py

#This is the unet model architechture applied on binary rings.
#keras version 1.2.2.

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

#############
#custom loss#
########################################################################
def template_match_target_to_csv(target, csv_coords, minrad=2, maxrad=75):
    # hyperparameters, probably don't need to change
    ring_thickness = 2       #thickness of rings for the templates. 2 seems to work well.
    template_thresh = 0.5    #0-1 range, if template matching probability > template_thresh, count as detection
    target_thresh = 0.1      #0-1 range, pixel values > target_thresh -> 1, pixel values < target_thresh -> 0
    
    #Match Threshold (squared)
    # for template matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, remove (x2,y2,r2) circle (it is a duplicate).
    # for predicted target -> csv matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, positive detection
    match_thresh2 = 20
    
    # target - can be predicted or ground truth
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0
    
    radii = np.linspace(minrad,maxrad,maxrad-minrad,dtype=int)
    templ_coords = []        #coordinates extracted from template matching
    for r in radii:
        # template
        n = 2*(r+ring_thickness+1)
        template = np.zeros((n,n))
        cv2.circle(template, (r+ring_thickness+1,r+ring_thickness+1), r, 1, ring_thickness)
        
        # template match
        result = match_template(target, template, pad_input=True)   #skimage
        coords_r = np.asarray(zip(*np.where(result > template_thresh)))
        
        # store x,y,r
        for c in coords_r:
            templ_coords.append([c[1],c[0],r])

    # remove duplicates from template matching at neighboring radii/locations
    templ_coords = np.asarray(templ_coords)
    i, N = 0, len(templ_coords)
    while i < N:
        diff = (templ_coords - templ_coords[i])**2
        diffsum = np.asarray([sum(x) for x in diff])
        index = (diffsum == 0)|(diffsum > match_thresh2)
        templ_coords = templ_coords[index]
        N, i = len(templ_coords), i+1

    # compare template-matched results to "ground truth" csv input data
    N_match = 0
    N_csv, N_templ = len(csv_coords), len(templ_coords)
    for tc in templ_coords:
        diff = (csv_coords - tc)**2
        diffsum = np.asarray([sum(x) for x in diff])
        index = (diffsum == 0)|(diffsum > match_thresh2)
        N = len(np.where(index==False)[0])
        if N > 1:
            print "multiple matches found in csv file for template matched crater ", tc, " :"
            print csv_coords[np.where(index==False)]
        N_match += N
        csv_coords = csv_coords[index]
        if len(csv_coords) == 0:
            #print "all matches found"
            break
    return N_match, N_csv, N_templ


def prepare_custom_loss(path, dim):
    # hyperparameters - should not change
    minrad, maxrad = 2, 75    #min/max radius (in pixels) required to include crater in target
    cutrad = 0.5              #0-1 range, if x+cutrad*r > img_width, remove, i.e. exclude craters ~half gone from image
    min_craters = 5           #minimum craters in the image required for processing (make it worth your while)
    
    # load data
    try:
        imgs = np.load("%scustom_loss_images.npy"%path)
        csvs = np.load("%scustom_loss_csvs.npy"%path)
        N_perfect_matches = len(imgs)
        print "Successfully loaded files locally for custom_loss."
    except:
        print "Couldn't load files for custom_loss, making now"
        path = 'datasets/rings/Test_rings_for_custom_loss/'
        imgs, targets, csvs = [], [], []
        csvs_ = glob.glob('%s*.csv'%path)
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
            N_match, N_csv, N_templ = template_match_target_to_csv(target, csv_coords, minrad, maxrad)
            if N_match == N_csv:
                imgs.append(img)
                targets.append(target)
                csvs.append(csv_coords)
                N_perfect_matches += 1
        imgs = np.array(imgs).astype('float32').reshape(len(imgs),dim,dim,1)
        targets = np.array(targets).astype('float32')
        np.save("%scustom_loss_images.npy"%path,imgs)
        np.save("%scustom_loss_csvs.npy"%path,csvs)
        print "out of %d files there are %d perfect matches"%(len(csvs_),N_perfect_matches)
    return imgs, csvs, N_perfect_matches

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
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init, W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((dim, dim))(u)
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
def train_and_test_model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,loss_data,loss_csvs,learn_rate,batch_size,lmbda,FL,nb_epoch,dim,save_model,init,n_filters):
    
    model = unet_model(dim,learn_rate,lmbda,FL,init,n_filters)
    
    n_samples = len(X_train)
    for nb in range(nb_epoch):
        model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_samples,nb_epoch=1,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=n_samples,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
                        
        # calcualte custom loss
        loss = []
        print "custom loss for epoch %d is (N_csv-N_match, N_templ-N_match, N_templ-N_csv):"%nb
        loss_target = model.predict(loss_data.astype('float32'))
        for i in range(len(loss_data)):
            N_match, N_csv, N_templ = template_match_target_to_csv(loss_target[i], loss_csvs[i])
            loss.append((N_csv-N_match, N_templ-N_match, N_templ-N_csv))
        print loss
    
    if save_model == 1:
        model.save('models/unet_s256_rings_FL%d_%s_customloss.h5'%(FL,init))

    return model.evaluate(X_test.astype('float32'), Y_test.astype('float32'))

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,lmbda,nb_epoch,n_train_samples,save_models,inv_color,rescale):
    #Static arguments
    dim = 256              #image width/height, assuming square images
    
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
        train_data, train_target = read_and_normalize_data(train_path, dim, 'train')
        valid_data, valid_target = read_and_normalize_data(valid_path, dim, 'validation')
        test_data, test_target = read_and_normalize_data(test_path, dim, 'test')
        np.save('%s/Train_rings/train_data.npy'%dir,train_data)
        np.save('%s/Train_rings/train_target.npy'%dir,train_target)
        np.save('%s/Dev_rings/valid_data.npy'%dir,valid_data)
        np.save('%s/Dev_rings/valid_target.npy'%dir,valid_target)
        np.save('%s/Test_rings/test_data.npy'%dir,test_data)
        np.save('%s/Test_rings/test_target.npy'%dir,test_target)
    #take desired subset of data
    train_data, train_target = train_data[:n_train_samples], train_target[:n_train_samples]
    valid_data, valid_target = valid_data[:n_train_samples], valid_target[:n_train_samples]
    test_data, test_target = test_data[:n_train_samples], test_target[:n_train_samples]

    #prepare custom loss
    custom_loss_path = '%sTest_rings_for_custom_loss/'%dir
    loss_data, loss_csvs, N_loss = prepare_custom_loss(custom_loss_path, dim)

    #Invert image colors and rescale pixel values to increase contrast
    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        train_data = rescale_and_invcolor(train_data, inv_color, rescale)
        valid_data = rescale_and_invcolor(valid_data, inv_color, rescale)
        test_data = rescale_and_invcolor(test_data, inv_color, rescale)
        loss_data = rescale_and_invcolor(loss_data, inv_color, rescale)

    #Iterate
    N_runs = 1
    filter_length = [3]
    n_filters = [64]  #arranging this so that total number of model parameters <~ 10M.
    init = ['he_normal']
    for i in range(N_runs):
        I = init[i]
        NF = n_filters[i]
        FL = filter_length[i]
        score = train_and_test_model(train_data,train_target,valid_data,valid_target,test_data,test_target,loss_data,loss_csvs,learn_rate,batch_size,lmbda,FL,nb_epoch,dim,save_models,I,NF)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
        print 'learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d, n_train_samples=%d, img_dimensions=%d, inv_color=%d, rescale=%d, init=%s, n_filters=%d'%(learn_rate,batch_size,FL,nb_epoch,n_train_samples,dim,inv_color,rescale,I,NF)
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
    epochs = 10         #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 10016     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models
    inv_color = 1       #use inverse color
    rescale = 1         #rescale images to increase contrast (still 0-1 normalized)
    
    #run models
    run_cross_validation_create_models(lr,bs,lmbda,epochs,n_train,save_models,inv_color,rescale)
