
################################# Import Stuffs #################################
import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import StratifiedKFold, KFold
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import mean_absolute_error
from keras import __version__ as keras_version
from keras import backend as K
import argparse

K.set_image_dim_ordering('tf')
#################################

learning_rate = 0.0001
img_width = 224
img_height = 224
#nbr_train_samples = 3019
#nbr_validation_samples = 758
#nbr_epochs = 25
#batch_size = 32
#################################

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", type=str)
    parser.add_argument("--run_fold", type=int, default=1)
    #parser.add_argument("--nice", dest="nice", action="store_true")
#parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args



################################# READ FILES, DONT CHANGE #################################
def get_im_cv2(path):
    img = cv2.imread(path)
    #resized = cv2.resize(img, (32, 32))#, cv2.INTER_LINEAR)
    resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR)
    return resized
def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    print('Read train images')
    folders = ['training_set']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('./', fld, '*.png')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(fl)
            y_train.append(y_trainn2(fl))
    return  X_train, y_train, X_train_id             
def load_test():
    X_test = []
    X_test_id = []
    y_test = []
    print('Read test images')
    folders = ['test_set']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('./', fld, '*.png')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_test.append(img)
            X_test_id.append(fl)
            y_test.append(y_testt2(fl))


#    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, y_test, X_test_id
    
def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
  
    #print('Reshape...') # No reshape to get tensorflow ordering
#    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
 
    #train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    
    return train_data, train_target, train_id
    
def read_and_normalize_test_data():
    test_data, test_target, test_id = load_test()

    print('Convert to numpy...')

    test_data = np.array(test_data, dtype=np.uint8)
    test_target = np.array(test_target, dtype=np.uint8)
    #print('Reshape...') # No reshape to get tensorflow ordering
#    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')

    test_data = test_data.astype('float32')
    test_data = test_data / 255    
    #train_target = np_utils.to_categorical(train_target, 8)

    return test_data, test_target, test_id


def y_testt2(file_):
    target = []    
    file2_=file_[:21]
    file2_=str(file2_)+'.csv'
    df = pd.read_csv(file2_ , header=0) 
    target.append(len(df.index))
    return target

def y_trainn2(file_):
    target = []    
    file2_=file_[:25]
    file2_=str(file2_)+'.csv'
    df = pd.read_csv(file2_ , header=0) 
    target.append(len(df.index))
    return target
###################################################################################################

################################################# Convnet Model  #############################################
def create_model_resnet():
    print('Loading ResNet50 Weights ...')
    ResNet50_notop = ResNet50(include_top=False, weights='imagenet',
                    input_tensor=None , input_shape=(224, 224,3)
                                    )
#    ResNet50_notop = InceptionV3(include_top=False, weights='imagenet',
#                    input_tensor=None , input_shape=(299, 299,3)
#                                    )
    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = ResNet50_notop.get_layer(index = -1).output 
    output = Dropout(0.50)(output)
    
     # Shape: (8, 8, 2048)    
#    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)

    output = Flatten(name='flatten')(output)
    output = Dense(1, activation='relu', name='predictions')(output)

    ResNet50_model = Model(ResNet50_notop.input, output)
#    ResNet50_model.summary()
 #   exit(0)
    optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
    ResNet50_model.compile(loss='mae', optimizer = optimizer, metrics = ['accuracy'])
    return ResNet50_model
###################################################################################################




################################################ Main Routine ############################################
def run_cross_validation_create_models(nfolds=4):
    # input image dimensions
    batch_size = 128 #16
    nb_epoch =30 #30
    random_state = 51
    args = get_args()

    models = [] 
    train_data, train_target, train_id = read_and_normalize_train_data()
    test_data, test_target, test_id = read_and_normalize_test_data()
    
#    print train_target

    y_for_folds=train_target.copy()
#    train_target = np_utils.to_categorical(train_target)
#    print train_target
    
    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
#   # kf = StratifiedKFold(y_for_folds, n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    
    

				
    for train_index, test_index in kf:
					
#        print len(train_data[train_index]), len((train_target[train_index]))
        model = create_model_resnet()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        if args.run_fold != num_fold: 
												continue
								
								
        

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
#            best_model
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)
        
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = mean_absolute_error(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Total average loss score: ", score)
    
    for model in (models):
        predictions_valid_test = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
        score = mean_absolute_error(test_target, predictions_valid_test)    
        print('Validation set Score log_loss: ', score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) +'_ep_' + str(nb_epoch)
    return info_string, models
###################################################################################################

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 4
#    path =r'./training_set' 
#    allFiles = glob.glob(path + "/*.csv")
#    allFilesimg = glob.glob(path + "/*.png")
    info_string, models = run_cross_validation_create_models(num_folds)
