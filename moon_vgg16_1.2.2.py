#This tries out different modules to see if they add any benefit.

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
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
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
    resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR)
    return resized

def get_csv_len(file_):                        #returns # craters in each image (target)
    file2_ = file_.split('.png')[0] + '.csv'
    df = pd.read_csv(file2_ , header=0)
    return [len(df.index)]

def load_data(path, data_type, img_width, img_height):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl,img_width,img_height)
        X.append(img)
        X_id.append(fl)
        y.append(get_csv_len(fl))
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

###########################
#vgg16 model (keras 1.2.2)#
########################################################################
#Following https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py 
def vgg16(n_classes,im_width,im_height,learn_rate,lmbda,dropout):
    n_filters = 32          #vgg16 uses 64
    n_blocks = 3            #vgg16 uses 5
    n_dense = 512           #vgg16 uses 4096

    #first block
    print('Making VGG model...')
    model = Sequential()
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda), input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))                          #maximum of 512 filters in vgg16
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same', W_regularizer=l2(lmbda)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(n_dense, activation='relu', W_regularizer=l2(lmbda)))   #biggest memory sink
    model.add(Dropout(dropout))
    model.add(Dense(n_dense, activation='relu', W_regularizer=l2(lmbda)))
    model.add(Dense(n_classes, activation='relu', name='predictions'))      #relu/regression output

    #optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    print model.summary()
    return model

##################
#Train/Test Model#
########################################################################
#Need to create this function so that memory is released every iteration (when function exits).
#Otherwise the memory used accumulates and eventually the program crashes.
def train_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,lmbda,drop,nb_epoch,n_train_samples,im_width,im_height,n_classes,rs):
    
    #Main Routine - Build/Train/Test model
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.20, random_state=rs)
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    
    #ImageDataGenerator - for manipulating images to prevent overfitting
    gen = ImageDataGenerator(#channel_shift_range=30,                    #R,G,B shifts
                             #rotation_range=180,                        #rotations
                             width_shift_range=1./im_width,
                             height_shift_range=1./im_height,
                             fill_mode='constant',
                             horizontal_flip=True,vertical_flip=True)    #flips

    model = vgg16(n_classes,im_width,im_height,learn_rate,lmbda,drop)
    model.fit_generator(gen.flow(X_train,Y_train,batch_size=batch_size,shuffle=True),
                         samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                         #validation_data=(X_valid, Y_valid), #no generator for validation data
                         validation_data=gen.flow(X_valid,Y_valid,batch_size=batch_size),nb_val_samples=len(X_valid),
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
    #model_name = ''
    #model.save_weights(model_name)     #save weights of the model
     
    test_predictions = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
    return mean_absolute_error(test_target, test_predictions)  #calculate test score

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,lmbda,dropout,nb_epoch,n_train_samples):
    #Static arguments
    n_classes = 1               #number of classes in final dense layer
    im_width = 224              #image width
    im_height = 224             #image height
    rs = 42                     #random_state for train/test split

    #Load data
    kristen_dir = '/scratch/k/kristen/malidib/moon/'
    try:
        train_data=np.load('training_set/train_data.npy')
        train_target=np.load('training_set/train_target.npy')
        test_data=np.load('test_set/test_data.npy')
        test_target=np.load('test_set/test_target.npy')
        print "Successfully loaded files locally."
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%kristen_dir
        train_path, test_path = '%straining_set/'%kristen_dir, '%stest_set/'%kristen_dir
        train_data, train_target, train_id = read_and_normalize_data(train_path, im_width, im_height, 0)
        test_data, test_target, test_id = read_and_normalize_data(test_path, im_width, im_height, 1)
        np.save('training_set/train_data.npy',train_data)
        np.save('training_set/train_target.npy',train_target)
        np.save('test_set/test_data.npy',test_data)
        np.save('test_set/test_target.npy',test_target)
    train_data = train_data[:n_train_samples]
    train_target = train_target[:n_train_samples]

    #Squash train_target (e.g. from 0-10 -> 0-1 crater counts)
    #train_target = np.log10(1+train_target)

    #Iterate
    N_runs = 6
    lmbda = random.sample(np.logspace(-3,1,5*N_runs), N_runs-1)
    dropout = random.sample(np.linspace(0,0.8,5*N_runs), N_runs-1)
    lmbda.append(0), dropout.append(0)  #ensure we have a baseline comparison
    for i in range(N_runs):
        l,d = lmbda[i], dropout[i]
        score = train_test_model(train_data,train_target,test_data,test_target,learn_rate,batch_size,l,d,nb_epoch,n_train_samples,im_width,im_height,n_classes,rs)
        print '###################################'
        print '##########END_OF_RUN_INFO##########'
        print('\nTest Score is %f.\n'%score)
        print 'learning_rate=%e, batch_size=%d, lambda=%e, dropout=%f, n_epoch=%d, n_train_samples=%d, n_classes=%d, random_state=%d, im_width=%d, im_height=%d'%(learn_rate,batch_size,l,d,nb_epoch,n_train_samples,n_classes,rs,im_width,im_height)
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
    dropout = 0         #percentage of neurons in the FC layers randomly set to 0.
    epochs = 30         #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 16000     #number of training samples, needs to be a multiple of batch size. Big memory hog.

    #run models
    run_cross_validation_create_models(lr,bs,lmbda,dropout,epochs,n_train)


