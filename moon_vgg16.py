import cv2
import os
import glob
import numpy as np
import pandas as pd
import argparse

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import __version__ as keras_version
from keras import backend as K
K.set_image_dim_ordering('tf')

#####################
#load/read functions#
########################################################################
def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", type=str)
    parser.add_argument("--run_fold", type=int, default=1)
    #parser.add_argument("--nice", dest="nice", action="store_true")
    #parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

def get_im_cv2(path):
    img = cv2.imread(path)
    #resized = cv2.resize(img, (32, 32))#, cv2.INTER_LINEAR)
    resized = cv2.resize(img, (img_width, img_height))#, cv2.INTER_LINEAR)
    return resized

def load_data(path, data_type):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X.append(img)
        X_id.append(fl)
        y.append(get_csv_len(fl))
    return  X, y, X_id

def read_and_normalize_data(path):
    if 'train' in path:
        data_type = 'train'
    elif 'test' in path:
        data_type = 'test'
    data, target, id = load_data(path, data_type)
    data = np.array(data, dtype=np.uint8)      #convert to numpy
    target = np.array(target, dtype=np.uint8)
    data = data.astype('float32')              #convert to float
    data = data / 255
    print('%s shape:'%data_type, data.shape)
    return data, target, id

def get_csv_len(file_):                        #previously y_trainn2
    file2_ = file_.split('.png')[0] + '.csv'
    df = pd.read_csv(file2_ , header=0)
    return [len(df.index)]

#############
#vgg16 model#
########################################################################
#Following https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
def vgg16(n_classes,im_width,im_height,learn_rate):
    model = Sequential()
    n_filters = 32          #vgg16 uses 64
    n_blocks = 3            #vgg16 uses 5
    n_dense = 512           #vgg16 uses 4096

    #first block
    model.add(Conv2D(n_filters, (3, 3), activation='relu', padding='same', input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))          #maximum of 512 filters in vgg16
        model.add(Conv2D(n_filters_, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(n_filters_, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_classes, activation='relu'))          #if counting craters, want a relu/regression output

    optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,nfolds=4,n_classes=1,im_width=224,im_height=224):
    # input image dimensions
    batch_size = 64 #16
    nb_epoch = 1 #30
    random_state = 51
    args = get_args()

    train_path, test_path = 'training_set/', 'test_set/'
    train_data, train_target, train_id = read_and_normalize_data(train_path)
    test_data, test_target, test_id = read_and_normalize_data(test_path)

    y_for_folds=train_target.copy()
    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    # kf = StratifiedKFold(y_for_folds, n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_index,test_index in kf:
        model = vgg16(n_classes,im_width,im_height,learn_rate)
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
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0)]
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

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    
    #args
    learning_rate = 0.0001  #learning rate
    num_folds = 2           #number of cross-validation folds
    n_classes = 1           #number of classes in final dense layer
    img_width = 224         #image width
    img_height = 224        #image height
    
    #run model
    info_string, models = run_cross_validation_create_models(learning_rate,num_folds,n_classes,img_width,img_height)

