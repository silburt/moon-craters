#This model uses skip connections to merge the where with the what, and have scale aware analysis.
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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.regularizers import l2
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

import utils.make_density_map2d as mdm

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

#############################
#FCN vgg model (keras 1.2.2)#
########################################################################
#Following https://github.com/aurora95/Keras-FCN/blob/master/models.py
#and also loosely following: https://blog.keras.io/building-autoencoders-in-keras.html
#and maybe: https://github.com/nicolov/segmentation_keras
#and this!: https://gist.github.com/Neltherion/f070913fd6284c4a0b60abb86a0cd642
def FCN_skip_model(im_width,im_height,learn_rate,lmbda):
    print('Making VGG16-style Fully Convolutional Network model...')
    n_filters = 32          #vgg16 uses 64
    n_blocks = 4            #vgg16 uses 5
    n_dense = 256           #vgg16 uses 4096
    
    img_input = Input(batch_shape=(None, im_width, im_height, 3))
    l1 = Convolution2D(n_filters, 3, 3, activation='relu', name='conv1_1', border_mode='same')(img_input)
    l1 = Convolution2D(n_filters, 3, 3, activation='relu', name='conv1_2', border_mode='same')(l1)
    l1P = MaxPooling2D((2, 2), name='maxpool_1')(l1)

    l2 = Convolution2D(n_filters*2, 3, 3, activation='relu', name='conv2_1', border_mode='same')(l1P)
    l2 = Convolution2D(n_filters*2, 3, 3, activation='relu', name='conv2_2', border_mode='same')(l2)
    l2P = MaxPooling2D((2, 2), name='maxpool_2')(l2)

    l3 = Convolution2D(n_filters*4, 3, 3, activation='relu', name='conv3_1', border_mode='same')(l2P)
    l3 = Convolution2D(n_filters*4, 3, 3, activation='relu', name='conv3_2', border_mode='same')(l3)
    l3P = MaxPooling2D((3, 3), name='maxpool_3')(l3)

    l4 = Convolution2D(n_filters*4, 3, 3, activation='relu', name='conv4_1', border_mode='same')(l3P)
    l4 = Convolution2D(n_filters*4, 3, 3, activation='relu', name='conv4_2', border_mode='same')(l4)

    u = UpSampling2D((3,3), name='up4->3')(l4)
    u = BatchNormalization(axis=1, name='normu3')(u)
    u = merge((BatchNormalization(axis=1, name='norml3')(l3), u), mode='concat', name='merge3')
    u = Convolution2D(n_filters*4, 3, 3, activation='relu', name='conv_merge3', border_mode='same')(u)
        
    u = UpSampling2D((2,2), name='up3->2')(u)
    u = BatchNormalization(axis=1, name='normu2')(u)
    u = merge((BatchNormalization(axis=1, name='norml2')(l2), u), mode='concat', name='merge2')
    u = Convolution2D(n_filters*2, 3, 3, activation='relu', name='conv_merge2', border_mode='same')(u)

    u = UpSampling2D((2,2), name='up2->1')(u)
    u = BatchNormalization(axis=1, name='normu1')(u)
    u = merge((BatchNormalization(axis=1, name='norml1')(l1), u), mode='concat', name='merge1')
    u = Convolution2D(n_filters, 3, 3, activation='relu', name='conv_merge1', border_mode='same')(u)

    #final output
    u = Convolution2D(1, 3, 3, activation='relu', name='output', border_mode='same')(u)
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
def train_and_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,lmbda,nb_epoch,im_width,im_height,rs,save_model):
    
    #Main Routine - Build/Train/Test model
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.20, random_state=rs)
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    
    model = FCN_skip_model(im_width,im_height,learn_rate,lmbda)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
    
    if save_model == 1:
        model.save('models/FCNskip_lmbda%.0e.h5'%lmbda)
     
    test_pred = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    return np.sum((test_pred - test_target)**2)/test_target.shape[0]  #calculate test score

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
        train_data=np.load('training_set/train_data_mask2d.npy')
        train_target=np.load('training_set/train_target_mask2d.npy')
        test_data=np.load('test_set/test_data_mask2d.npy')
        test_target=np.load('test_set/test_target_mask2d.npy')
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        train_path, test_path = '%straining_set/'%dir, '%stest_set/'%dir
        train_data, train_target, train_id = read_and_normalize_data(train_path, im_width, im_height, 0)
        test_data, test_target, test_id = read_and_normalize_data(test_path, im_width, im_height, 1)
        np.save('training_set/train_data_mask2d.npy',train_data)
        np.save('training_set/train_target_mask2d.npy',train_target)
        np.save('test_set/test_data_mask2d.npy',test_data)
        np.save('test_set/test_target_mask2d.npy',test_target)
    train_data = train_data[:n_train_samples]
    train_target = train_target[:n_train_samples]

    #Iterate
    N_runs = 1
    lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1)
    lmbda.append(0)
    for i in range(N_runs):
        l = lmbda[i]
        score = train_and_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,l,nb_epoch,im_width,im_height,rs,save_models)
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
    lmbda = 0           #L2 regularization strength (lambda)
    epochs = 5          #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 16000     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    save_models = 1     #save models

    #run models
    run_cross_validation_create_models(lr,bs,lmbda,epochs,n_train,save_models)


