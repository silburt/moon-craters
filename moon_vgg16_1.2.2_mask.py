#This tries out different modules to see if they add any benefit.

#This python script is adapted from moon2.py and uses the vgg16 convnet structure.
#The number of blocks, and other aspects of the vgg16 model can be modified.
#This has the keras 1.2.2. architechture

import cv2
import os
import glob
import numpy as np
import pandas as pd
import make_density_map as mdm

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

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

def load_data(path, data_type, img_width, img_height):
    X, X_id, y = [], [], []
    minpix = 2                                  #minimum number of pixels for crater to count
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl,img_width,img_height)
        X.append(img)
        X_id.append(fl)
        
        #make mask as target
        csv = pd.read_csv('%s.csv'%fl.split('.png')[0])
        csv.drop(np.where(csv['Diameter (pix)'] < minpix)[0], inplace=True)
        y.append(mdm.make_mask(csv, (img_width,img_height), binary=True))
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
    data = data / 255                           #normalize
    print('%s shape:'%data_type, data.shape)
    return data, target, id

########################
#custom image generator#
########################################################################
#Following https://github.com/fchollet/keras/issues/2708
def custom_image_generator(data, target, batch_size=32):
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i+batch_size].copy(), target[i:i+batch_size].copy() #is this the most memory efficient way?
            lr, ud = np.where(np.random.randint(0,2,batch_size)==1)[0], np.where(np.random.randint(0,2,batch_size)==1)[0]
            for j in lr:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])   #left/right flips
            for j in ud:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])   #up/down flips
            yield (d, t)

###########################
#vgg16 model (keras 1.2.2)#
########################################################################
#Following https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
def vgg16(n_classes,im_width,im_height,learn_rate):
    print('Making VGG16 model...')
    model = Sequential()
    n_filters = 64          #vgg16 uses 64
    n_blocks = 4            #vgg16 uses 5
    n_dense = 2048          #vgg16 uses 4096
    
    #first block
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same', input_shape=(im_width,im_height,3)))
    model.add(Conv2D(n_filters, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #subsequent blocks
    for i in np.arange(1,n_blocks):
        n_filters_ = np.min((n_filters*2**i, 512))                      #maximum of 512 filters in vgg16
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
        model.add(Conv2D(n_filters_, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(n_classes, activation='relu',name='predictions'))   #relu/regression output

    #optimizer = SGD(lr=learn_rate, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    print model.summary()
    return model

##############
#Main Routine#
########################################################################
def run_cross_validation_create_models(learn_rate,batch_size,nb_epoch,n_train_samples):
    #static arguments
    nfolds = 4                  #number of cross-validation folds
    n_classes = 1               #number of classes in final dense layer
    im_width = 300              #image width
    im_height = 300             #image height
    random_state = 51
    
    #load data
    kristen_dir = '/scratch/k/kristen/malidib/moon/'
    try:
        train_data=np.load('training_set/train_data.npy')[:n_train_samples]
        train_target=np.load('training_set/train_target.npy')[:n_train_samples]
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

    #main routine
    kf = KFold(len(train_target), n_folds=nfolds, shuffle=True, random_state=random_state)
    sum_score, num_fold = 0, 0
    for train_index,test_index in kf:
        num_fold += 1
        model = vgg16(n_classes,im_width,im_height,learn_rate)
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        print('Start KFold number %d of %d'%(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0)]
        
        model.fit_generator(custom_image_generator(X_train,Y_train,batch_size=batch_size),
                            samples_per_epoch=n_train_samples,nb_epoch=nb_epoch,verbose=1,
                            validation_data=(X_valid, Y_valid), #no generator for validation data
                            #validation_data=custom_image_generator(X_valid,Y_valid,batch_size=batch_size),
                            nb_val_samples=len(X_valid),
                            callbacks=callbacks)
            
        #model_name = ''
        #model.save_weights(model_name)     #save weights of the model
        
        #make target predictions and unsquash
        predictions_valid = model.predict(test_data.astype('float32'), batch_size=batch_size, verbose=2)
        
        #calculate test score
        score = mean_absolute_error(test_target, predictions_valid)
        print('\nTest Score for fold %d Score is %f.\n'%(num_fold, score))
        sum_score += score

    info_string = 'avgloss_' + str(sum_score/nfolds) + '_folds_' + str(nfolds) +'_ep_' + str(nb_epoch)
    return info_string

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    
    #args
    lr = 0.0001         #learning rate
    bs = 32             #batch size: smaller values = less memory but less accurate gradient estimate
    epochs = 30         #number of epochs. 1 epoch = forward/back pass thru all train data
    n_train = 16000     #number of training samples, needs to be a multiple of batch size. Big memory hog.
    
    #run model
    info_string = run_cross_validation_create_models(lr,bs,epochs,n_train)
    print info_string
