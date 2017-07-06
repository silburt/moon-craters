import glob
import numpy as np

from keras.models import load_model
from keras import backend as K
from utils.rescale_invcolor import rescale_and_invcolor
################
#Loss Functions#
########################################################################
def weighted_binary_XE(y_true, y_pred):
    #sum total number of 1s and 0s in y_true
    total_ones = tf.reduce_sum(y_true)
    total_zeros = tf.reduce_sum(tf.to_float(tf.equal(y_true, tf.zeros_like(y_true))))
    result = K.binary_crossentropy(y_pred, y_true)
    weights = y_true * total_zeros*1.0/(total_zeros + total_ones)
    return K.mean(result*weights + result, axis=-1)

##############
#Main Routine#
########################################################################
def predict_targets(dir,inv_color,rescale,n_pred_samples,offset,models,custom_loss):
    #static arguments
    dim = 256               #image dimensions, assuming square images. Should not change
    
    #load data
    if n_pred_samples < 50:
        try:
            test_data = np.load('%s/Test_rings/test_data_50im.npy'%dir)[:n_pred_samples]
            test_target = np.load('%s/Test_rings/test_target_50im.npy'%dir)[:n_pred_samples]
            print "Loaded 50 image subset successfully."
        except:
            print "Couldn't find 50 image subset numpy arrays. Loading full data. Saving subset of 50 images for future use."
            test_data = np.load('%s/Test_rings/test_data.npy'%dir)[:n_pred_samples]
            test_target = np.load('%s/Test_rings/test_target.npy'%dir)[:n_pred_samples]
            np.save('%s/Test_rings/test_data_50im.npy'%dir,test_data[0:50])
            np.save('%s/Test_rings/test_target_50im.npy'%dir,test_target[0:50])
    else:
        test_data = np.load('%s/Test_rings/test_data.npy'%dir)[:n_pred_samples]
        test_target = np.load('%s/Test_rings/test_target.npy'%dir)[:n_pred_samples]
    test_data = rescale_and_invcolor(test_data, inv_color, rescale)

    print "Generating predictions."
    for m in models:
        if custom_loss == 1:
            model = load_model('%s'%m, custom_objects={'weighted_binary_XE': weighted_binary_XE})
        else:
            model = load_model('%s'%m)
        target_pred = model.predict(test_data[offset:(n_pred_samples+offset)].astype('float32'))
        
        #dimensions go data, ground_truth targets, predicted targets
        result = np.concatenate((test_data[offset:(n_pred_samples+offset)],
                                 test_target[offset:(n_pred_samples+offset)].reshape(n_pred_samples,dim,dim,1),
                                 target_pred.reshape(n_pred_samples,dim,dim,1)),axis=3)
                                 
        name = m.split('.h5')[0]
        np.save('%s_pred.npy'%name,result)
        print "Successfully generated predictions at %s_pred.npy for model %s."%(name,m)

################
#Arguments, Run#
########################################################################
if __name__ == '__main__':
    #arguments
    dir = 'datasets/rings'   #location of where test data is. Likely doesn't need to change
    inv_color = 1           #use inverse color (**must be same setting as what was used for the model(s)**)
    rescale = 1             #rescale images to increase contrast (**must be same setting as what was used for the model(s)**)
    n_pred_samples = 20     #number of test images to predict on
    offset = 0              #index offset to start predictions at in test array
    custom_loss = 0        #custom loss
    models = ['models/unet_s256_rings_FL3_weightedpred.h5']
    
    predict_targets(dir,inv_color,rescale,n_pred_samples,offset,models,custom_loss)
    

