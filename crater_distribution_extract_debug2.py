# This code is typically run on p8t03/04 on scinet.

import numpy as np
import cPickle
import cv2
import glob
import os
import pandas as pd
from PIL import Image
from keras.models import load_model

from utils.rescale_invcolor import *
from utils.template_match_target import *

def load_data(path):
    X = []
    X_id = []
    files = glob.glob('%s/*.png'%path)
    print "number of files are: %d"%len(files)
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255.
        X.append(img)
        X_id.append(int(os.path.basename(f).split('_')[1].split('.png')[0]))
    return  X, X_id

def read_and_normalize_data(path, dim):
    data, id_ = load_data(path)
    data = np.array(data).astype('float32')             #convert to numpy, convert to float
    data = data.reshape(len(data),dim, dim, 1)          #add dummy third dimension, required for keras
    print 'shape:', data.shape
    return data, id_

def get_crater_dist(data_dir,data_prefix,csv_prefix,pickle_loc,model_loc,n_imgs,inv_color,rescale):
    
    # properties of the dataset, shouldn't change (unless you use a different dataset)
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = 256                       #image dimension (pixels, assume dim=height=width)
    P = cPickle.load(open(pickle_loc, 'r'))
    
    # get data
    try:
        pred = np.load('%s/%s_modelpreds_n%d_new.npy'%(data_dir,data_prefix,n_imgs))
        id=np.load('%s/%s_id.npy'%(data_dir,data_prefix))
        print "Successfully loaded %s files locally."%data_dir
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%data_dir
        data=np.load('%s/%s_data.npy'%(data_dir,data_prefix))
        id=np.load('%s/%s_id.npy'%(data_dir,data_prefix))
        data, id = data[:n_imgs], id[:n_imgs]
        data, id = read_and_normalize_data(data_dir, dim)
        if inv_color==1 or rescale==1:
            print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
            data = rescale_and_invcolor(data, inv_color, rescale)
        # generate model predictions
        model = load_model(model_loc)
        pred = model.predict(data.astype('float32'))
        np.save('%s/%s_modelpreds_n%d_new.npy'%(data_dir,data_prefix,n_imgs),pred)
        print "generated and saved predictions"

    # extract crater distribution, remove duplicates live
    pred_crater_dist = np.empty([0,3])
    GT_crater_dist = np.empty([0,3])
    print "Extracting crater radius distribution of %d files."%n_imgs
    for i in range(len(pred)):
        coords = template_match_target(pred[i])
        if len(coords) >= 1:
            P_ = P[id[i]]
            img_pix_height = float(P_['box'][2] - P_['box'][0])
            pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/float(dim))*r_moon
            long_pix,lat_pix,radii_pix = coords.T
            radii_km = radii_pix*pix_to_km
            long_deg = P_['llbd'][0] + (P_['llbd'][1]-P_['llbd'][0])*(long_pix/float(dim))
            lat_deg = P_['llbd'][3] - (P_['llbd'][3]-P_['llbd'][2])*(lat_pix/float(dim))
            tuple_pred = np.column_stack((long_deg,lat_deg,radii_km))
            pred_crater_dist = np.concatenate((pred_crater_dist,tuple_pred))
    
            #corresponding csv
            minrad, maxrad = np.min(radii_pix), np.max(radii_pix)
            cutrad = 1              #0-1 range, if x+cutrad*r > dim, remove, higher cutrad = larger % of circle required
            csv = pd.read_csv('%s/%s_%s.csv'%(data_dir,csv_prefix,str(id[i]).zfill(5)))
            csv = csv[(csv['Diameter (pix)'] <= 2*maxrad) & (csv['Diameter (pix)'] >= 2*minrad)]
            csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
            csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
            if len(csv) > 1:
                tuple_csv = np.column_stack((csv['Long'].values,csv['Lat'].values,csv['Diameter (km)'].values/2.))
                GT_crater_dist = np.concatenate((GT_crater_dist,tuple_csv))
            print i, minrad, maxrad

    np.save('%s/%s_predcraterdist_debug2_n%d.npy'%(data_dir,data_prefix,n_imgs),pred_crater_dist)
    np.save('%s/%s_GTcraterdist_debug2_n%d.npy'%(data_dir,data_prefix,n_imgs),GT_crater_dist)

if __name__ == '__main__':
    #args
#    data_dir = 'datasets/ilen_1500_to_2500/ilen_1500'       #location of data to predict on. Exclude final '/' in path.
#    data_prefix = ''                                        #prefix of e.g. *_data.npy files.
#    csv_prefix = ''                                         #prefix of e.g. *_0001.csv files.
#    pickle_loc = '%s/outp_p0.p'%data_dir                    #location of corresponding pickle file
#    model_loc = 'models/unet_s256_rings_nFL96.h5'

    data_dir = 'datasets/rings/Test_rings'                  #location of data to predict on. Exclude final '/' in path.
    data_prefix = 'test'                                    #prefix of e.g. *_data.npy files.
    csv_prefix = 'lola'                                     #prefix of e.g. *_0001.csv files.
    pickle_loc = '%s/lolaout_test.p'%data_dir               #location of corresponding pickle file
    model_loc = 'models/unet_s256_rings_nFL96.h5'
    
    n_imgs = 30016          #number of images to use for getting crater distribution.
    inv_color = 1           #**must be same setting as what model was trained on**
    rescale = 1             #**must be same setting as what model was trained on**

    get_crater_dist(data_dir,data_prefix,csv_prefix,pickle_loc,model_loc,n_imgs,inv_color,rescale)
    print "Script completed successfully"
