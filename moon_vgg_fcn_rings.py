#rings: This makes use of binary rings as the target. 
#Fork: From my pure skip connection model I'm noticing that the small craters are being captured nicely, but the large craters are not being recognized. So, I need a separate fork on the onset with a large receptive field to capture the large craters as well.
#Skip: This model uses skip connections to merge the where with the what, and have scale aware analysis.
#See "Residual connection on a convolution layer" in https://jtymes.github.io/keras_docs/1.2.2/getting-started/functional-api-guide/#multi-input-and-multi-output-models

#This python script is adapted from moon2.py and uses the vgg16 convnet structure.
#The number of blocks, and other aspects of the vgg16 model can be modified.
#This has the keras 1.2.2. architechture

import cv2
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
    minpix = 3                          #minimum pixels required for a crater to register in an image
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        #fbase = os.path.basename(f)
        img = get_im_cv2(f,img_width,img_height)
        X.append(img)
        X_id.append(f)
        
        #make mask as target
        csv = pd.read_csv('%s.csv'%f.split('.png')[0])
        csv.drop(np.where(csv['Diameter (pix)'] < minpix)[0], inplace=True)
        target = mdm.make_mask(csv, img, binary=True, truncate=True, rings=True)
        maxx = target.max()
        if maxx > 0:
            target /= maxx            #normalizing between 0-1
        y.append(target)
    return  X, y, X_id

def read_and_normalize_data(path, img_width, img_height, data_flag):
    if data_flag == 0:
        data_type = 'train'
    elif data_flag == 1:
        data_type = 'test'
    data, target, id = load_data(path, data_type, img_width, img_height)
    data = np.array(data).astype('float32')     #convert to numpy, convert to float
    target = np.array(target).astype('float32')
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
def FCN_skip_model(im_width,im_height,learn_rate,lmbda):
    print('Making VGG16-style Fully Convolutional Network model...')
    n_filters = 32          #vgg16 uses 64
    img_input = Input(batch_shape=(None, im_width, im_height, 3))

    #model a - small receptive field for small craters
    FL_a = 3            #Receptive Field
    a1 = Convolution2D(n_filters, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv1_a1', border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv1_a2', border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_a1')(a1)

    a2 = Convolution2D(n_filters*2, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv2_a1', border_mode='same')(a1P)
    a2 = Convolution2D(n_filters*2, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv2_a2', border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_a1')(a2)

    a3 = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv3_a1', border_mode='same')(a2P)
    a3 = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv3_a2', border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(3, 3), name='pool3_a1')(a3)

    a4 = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv4_a1', border_mode='same')(a3P)
    a4 = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv4_a2', border_mode='same')(a4)

    #model b - large receptive field for large craters via dilated conv
    #RFL = receptive field length = (DF - 1)(FL - 1) + FL
    DF = 4              #Dilation Factor - previous success = 4
    FL_b = 10           #Filter Length of model b - previous success = 10
    b1 = AtrousConvolution2D(n_filters, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='conv1_b1', border_mode='same')(img_input)
    b1 = AtrousConvolution2D(n_filters, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv1_b2', border_mode='same')(b1)
    b1P = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_b1')(b1)

    b2 = AtrousConvolution2D(n_filters*2, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv2_b1', border_mode='same')(b1P)
    b2 = AtrousConvolution2D(n_filters*2, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv2_b2', border_mode='same')(b2)
    b2P = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_b1')(b2)

    b3 = AtrousConvolution2D(n_filters*4, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv3_b1', border_mode='same')(b2P)
    b3 = AtrousConvolution2D(n_filters*4, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv3_b2', border_mode='same')(b3)
    b3P = MaxPooling2D((2, 2), strides=(3, 3), name='pool3_b1')(b3)

    b4 = AtrousConvolution2D(n_filters*4, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv4_b1', border_mode='same')(b3P)
    b4 = AtrousConvolution2D(n_filters*4, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv4_b2', border_mode='same')(b4)

    #merge models 1 and 2
    u = merge((a4, b4), mode='concat', name='merge4')
    u = Convolution2D(n_filters*8, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge4_1', border_mode='same')(u)
    u = Convolution2D(n_filters*8, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge4_2', border_mode='same')(u)
    #u = AtrousConvolution2D(n_filters*4, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv_merge4_1', border_mode='same')(u)
    u = UpSampling2D((3,3), name='up4->3')(u)

    u = merge((a3, b3, u), mode='concat', name='merge3')
    u = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge3_1', border_mode='same')(u)
    u = Convolution2D(n_filters*4, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge3_2', border_mode='same')(u)
    #u = AtrousConvolution2D(n_filters*2, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv_merge3_1', border_mode='same')(u)
    u = UpSampling2D((2,2), name='up3->2')(u)

    u = merge((a2, b2, u), mode='concat', name='merge2')
    u = Convolution2D(n_filters*2, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge2_1', border_mode='same')(u)
    u = Convolution2D(n_filters*2, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge2_2', border_mode='same')(u)
    #u = AtrousConvolution2D(n_filters, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv_merge2_1', border_mode='same')(u)
    u = UpSampling2D((2,2), name='up2->1')(u)

    u = merge((a1, b1, u), mode='concat', name='merge1')
    u = Convolution2D(n_filters, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge1_1', border_mode='same')(u)
    u = Convolution2D(n_filters, FL_a, FL_a, activation='relu', W_regularizer=l2(lmbda), name='conv_merge1_2', border_mode='same')(u)
    #u = AtrousConvolution2D(n_filters, FL_b, FL_b, atrous_rate=(DF,DF), W_regularizer=l2(lmbda), activation='relu', name='aconv_merge1_1', border_mode='same')(u)

    #final output - I think this should be a sigmoid but how to get it to work - or maybe it is working and just all the values are small and get squashed to 0.
    u = BatchNormalization()(u)
    u = Convolution2D(1, 3, 3, activation='sigmoid', W_regularizer=l2(lmbda), name='output', border_mode='same')(u)
    u = Reshape((im_width, im_height))(u)
    model = Model(input=img_input, output=u)
    
    #optimizer/compile
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
def train_and_test_model(train_data,train_target,test_data,test_target,n_train_samples,learn_rate,batch_size,lmbda,nb_epoch,im_width,im_height,rs,save_model):
    
    #Main Routine - Build/Train/Test model
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.20, random_state=rs)
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    
    model = FCN_skip_model(im_width,im_height,learn_rate,lmbda)
    '''
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
    '''
    model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                        samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                        #validation_data=(X_valid, Y_valid), #no generator for validation data
                        validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                        nb_val_samples=len(X_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
                        
    if save_model == 1:
        model.save('models/FCNforkskip_rings.h5')
     
    test_pred = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    npix = test_target.shape[0]*test_target.shape[1]*test_target.shape[2]
    return np.sum((test_pred - test_target)**2)/npix    #calculate test score

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,lmbda,nb_epoch,n_train_samples,save_models):
    #Static arguments
    im_width = 300              #image width
    im_height = 300             #image height
    rs = 42                     #random_state for train/test split

    #Load data
    dir = '/scratch/k/kristen/malidib/moon/'
    try:
        train_data=np.load('training_set/train_data_rings.npy')
        train_target=np.load('training_set/train_target_rings.npy')
        test_data=np.load('test_set/test_data_rings.npy')
        test_target=np.load('test_set/test_target_rings.npy')
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        train_path, test_path = '%straining_set/'%dir, '%stest_set/'%dir
        train_data, train_target, train_id = read_and_normalize_data(train_path, im_width, im_height, 0)
        test_data, test_target, test_id = read_and_normalize_data(test_path, im_width, im_height, 1)
        np.save('training_set/train_data_rings.npy',train_data)
        np.save('training_set/train_target_rings.npy',train_target)
        np.save('test_set/test_data_rings.npy',test_data)
        np.save('test_set/test_target_rings.npy',test_target)
    train_data = train_data[:n_train_samples]
    train_target = train_target[:n_train_samples]

    #Iterate
    N_runs = 1
    #lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1); lmbda.append(0)
    lmbda = [0]
    for i in range(len(lmbda)):
        l = lmbda[i]
        score = train_and_test_model(train_data,train_target,test_data,test_target,n_train_samples,learn_rate,batch_size,l,nb_epoch,im_width,im_height,rs,save_models)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f \n'%score)
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
    lmbda = 0           #L2 regularization strength (lambda)
    epochs = 2          #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 10080     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models

    #run models
    run_cross_validation_create_models(lr,bs,lmbda,epochs,n_train,save_models)


