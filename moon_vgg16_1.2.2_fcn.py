#This script constructs a fully connected network (FCN) following the same procedures as the paper "Fully Convolutional Networks for Semantic Segmentation" - https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
#The goal is for the FCN to output crater masks with the same dimensions as the input images, where 1=crater, 0=no crater.
#This has the keras 1.2.2. architechture

import cv2
import os
import glob
import numpy as np
import pandas as pd
import random
import utils.make_density_map as mdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D, AtrousConvolution2D
from keras.regularizers import l2

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

#####################
#load/read functions#
########################################################################
def get_im_cv2(path, img_width, img_height):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR) #downsampler.
    return resized

def load_data(path, data_type, img_width, img_height):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    minpix = 3      #minimum pixels required for a crater to register in an image
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        #fbase = os.path.basename(f)
        img = get_im_cv2(f,img_width,img_height)
        X.append(img)
        X_id.append(f)
    
        #make mask as target
        csv = pd.read_csv('%s.csv'%f.split('.png')[0])
        csv.drop(np.where(csv['Diameter (pix)'] < minpix)[0], inplace=True)
        y.append(mdm.make_mask(csv, img, binary=False, truncate=True))
    return  X, y, X_id

def read_and_normalize_data(path, img_width, img_height, data_flag):
    if data_flag == 0:
        data_type = 'train'
    elif data_flag == 1:
        data_type = 'test'
    data, target, id = load_data(path, data_type, img_width, img_height)
    data = np.array(data, dtype=np.uint8)       #convert to numpy
    target = np.array(target, dtype=np.uint8)
    data = data.astype('float32')               #convert to float
    data = data / 255                           #normalize color
    print('%s shape:'%data_type, data.shape)
    return data, target, id

########################
#custom image generator#
########################################################################
#Following https://github.com/fchollet/keras/issues/2708
def custom_image_generator(data, target, batch_size=32):
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i+batch_size].copy(), target[i:i+batch_size].copy() #is this the most memory efficient way?
            
            #horizontal/vertical flips
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])   #left/right flips
            for j in np.where(np.random.randint(0,2,batch_size)==1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])   #up/down flips
            
            #random up/down and left/right pixel shifts
            npix = 1
            h = np.random.randint(-npix,npix+1,batch_size)      #horizontal shift
            v = np.random.randint(-npix,npix+1,batch_size)      #vertical shift
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix,npix),(npix,npix),(0,0)), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix,:]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix+h[j]:L+h[j]+npix,npix+v[j]:W+v[j]+npix]
            yield (d, t)

#############################
#FCN vgg model (keras 1.2.2)#
########################################################################
#Following https://github.com/aurora95/Keras-FCN/blob/master/models.py
#and also loosely following https://blog.keras.io/building-autoencoders-in-keras.html
#and maybe https://github.com/nicolov/segmentation_keras
def FCN_model(im_width,im_height,learn_rate,lmbda):
    print('Making VGG16-style Fully Convolutional Network model...')
    n_filters = 32          #vgg16 uses 64
    n_blocks = 4            #vgg16 uses 5
    n_dense = 256          #vgg16 uses 4096
    upsample = im_height    #upsample scale - factor to get back to img_height, im_width

    #first block
    model = Sequential()
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda), input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        if i==1:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(3, 3)))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #FC->CONV layers - http://cs231n.github.io/convolutional-networks/#convert
    model.add(Conv2D(n_dense, nb_row=12, nb_col=12, activation='relu', border_mode='valid', W_regularizer=l2(lmbda), name='fc1')) #filter dim = dim of previous layer
    model.add(Conv2D(n_dense, nb_row=1, nb_col=1, activation='relu', border_mode='valid', W_regularizer=l2(lmbda), name='fc2'))

    #Upsample and create mask
    model.add(UpSampling2D(size=(upsample, upsample)))
    model.add(Conv2D(1, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda), name='output'))
    #model.add(Reshape((im_width,im_height)))

    #Alternative layers
    #model.add(AtrousConvolution2D(n_dense, nb_row=7, nb_col=7, activation='relu', border_mode='same', atrous_rate=(2, 2), W_regularizer=l2(lmbda), name='fc1'))
    #model.add(Deconvolution2D(20, nb_row=4, nb_col=4, output_shape=(None,im_width,im_height,1), subsample=(4, 4), border_mode='same', activation='relu', name='Deconv'))

    #optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=optimizer)
    print model.summary()
    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,lmbda,nb_epoch,n_train_samples,im_width,im_height,rs):
    
    #Main Routine - Build/Train/Test model
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.20, random_state=rs)
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    model = FCN_model(im_width,im_height,learn_rate,lmbda)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
    '''
    model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                        validation_data=(X_valid, Y_valid), #no generator for validation data
                        #validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=len(X_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
    '''
    #model_name = ''
    #model.save_weights(model_name)     #save weights of the model
    
    test_predictions = model.predict(test_data, batch_size=batch_size, verbose=2)
    return mean_absolute_error(test_target, test_predictions)  #calculate test score

##############
#Main Routine#
########################################################################
def run_models(learn_rate,batch_size,nb_epoch,n_train_samples,lmbda):
    #Static arguments
    im_width = 300              #image width
    im_height = 300             #image height
    rs = 43                     #random_state for train/test split
    
    #Load data
    dir = '/scratch/k/kristen/malidib/moon/'
    try:
        train_data=np.load('training_set/train_data_mask.npy')
        train_target=np.load('training_set/train_target_mask.npy')
        test_data=np.load('test_set/test_data_mask.npy')
        test_target=np.load('test_set/test_target_mask.npy')
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        train_path, test_path = '%straining_set/'%dir, '%stest_set/'%dir
        train_data, train_target, train_id = read_and_normalize_data(train_path, im_width, im_height, 0)
        test_data, test_target, test_id = read_and_normalize_data(test_path, im_width, im_height, 1)
        np.save('training_set/train_data_mask.npy',train_data)
        np.save('training_set/train_target_mask.npy',train_target)
        np.save('test_set/test_data_mask.npy',test_data)
        np.save('test_set/test_target_mask.npy',test_target)
    train_data = train_data[:n_train_samples]
    train_target = train_target[:n_train_samples]

    #Iterate
    for i in range(len(lmbda)):
        l = lmbda[i]
        score = train_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,l,nb_epoch,n_train_samples,im_width,im_height,rs)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f.\n'%score)
        print 'learning_rate=%e, batch_size=%d, lambda=%e, n_epoch=%d, n_train_samples=%d, random_state=%d, im_width=%d, im_height=%d'%(learn_rate,batch_size,l,nb_epoch,n_train_samples,rs,im_width,im_height)
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
    epochs = 30         #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 8000      #number of training samples, needs to be a multiple of batch size. Big memory hog.
    
    #iterables
    N_runs = 1
    lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1)           #L2 regularization strength (lambda)
    lmbda.append(0)
    
    #run models
    run_models(lr,bs,epochs,n_train,lmbda)
