# This code is typically run on p8t03/04 on scinet.

import numpy as np
import cPickle
import cv2
import glob
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
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

def get_crater_dist(data_dir,data_prefix,csv_prefix,pickle_loc,model_loc,n_imgs,inv_color,rescale,ground_truth_only,unique_thresh2):
    
    # properties of the dataset, shouldn't change (unless you use a different dataset)
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = 256                       #image dimension (pixels, assume dim=height=width)
    P = cPickle.load(open(pickle_loc, 'r'))
    
    #this likely needs to be fine-tuned by comparing csv dist extracted here to the alanalldata.csv list.
    #unique_thresh2 = 50
    
    # get data
    try:
        data=np.load('%s/%s_data.npy'%(data_dir,data_prefix))
        id=np.load('%s/%s_id.npy'%(data_dir,data_prefix))
        print "Successfully loaded %s files locally."%data_dir
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%data_dir
        data, id = read_and_normalize_data(data_dir, dim)
        np.save('%s/%s_data.npy'%(data_dir,data_prefix),data)
        np.save('%s/%s_id.npy'%(data_dir,data_prefix),id)
    data, id = data[:n_imgs], id[:n_imgs]

    if ground_truth_only == 0:
        if inv_color==1 or rescale==1:
            print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
            data = rescale_and_invcolor(data, inv_color, rescale)
        
        # generate model predictions
        model = load_model(model_loc)
        pred = model.predict(data.astype('float32'))

        # extract crater distribution, remove duplicates live
        print "Extracting crater radius distribution of %d files."%n_imgs
        pred_crater_dist = np.empty([0,3])
        for i in range(len(pred)):
            coords = template_match_target(pred[i])
            if len(coords) > 0:
                P_ = P[id[i]]
                img_pix_height = float(P_['box'][2] - P_['box'][0])
                pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/float(dim))*r_moon
                long_pix,lat_pix,radii_pix = coords.T
                radii_km = radii_pix*pix_to_km
                long_deg = P_['llbd'][0] + (P_['llbd'][1]-P_['llbd'][0])*(long_pix/float(dim))
                lat_deg = P_['llbd'][3] - (P_['llbd'][3]-P_['llbd'][2])*(lat_pix/float(dim))
                tuple_ = np.column_stack((long_deg,lat_deg,radii_km))

                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))
                #only add unique (non-duplicate) values to the master pred_crater_dist
                if len(pred_crater_dist) > 0:
                    for j in range(len(tuple_)):
                        diff = (pred_crater_dist - tuple_[j])**2
                        diffsum = np.asarray([sum(x) for x in diff])
                        #long, lat, rad = pred_crater_dist.T
                        #diffsum = np.asarray(25*(long-tuple_[j][0])**2 + 25*(lat-tuple_[j][1])**2 + (rad-tuple_[j][2])**2)
                        index = diffsum < unique_thresh2
                        if len(np.where(index==True)[0]) == 0: #unique value
                            pred_crater_dist = np.vstack((pred_crater_dist,tuple_[j]))
                else:
                    pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

        pred_crater_dist = np.asarray(pred_crater_dist)
        np.save('%s/%s_predcraterdist_n10000_new.npy'%(data_dir,data_prefix),pred_crater_dist)

    # Generate csv dist
    GT_crater_dist = np.empty([0,3])
    # hyperparameters
    minrad, maxrad = 3, 75  #min/max radius (in pixels) required to include crater in target
    cutrad = 1              #0-1 range, if x+cutrad*r > dim, remove, higher cutrad = larger % of circle required
    print "Getting ground truth crater distribution."
    for id_ in id:
        csv = pd.read_csv('%s/%s_%s.csv'%(data_dir,csv_prefix,str(id_).zfill(5)))
        csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
        csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
        csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
        csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
        csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
        GT_radii = csv['Diameter (km)'].values/2
        GT_lat = csv['Lat']
        GT_long = csv['Long']
        tuple_ = np.column_stack((GT_long,GT_lat,GT_radii))

        GT_crater_dist = np.concatenate((GT_crater_dist,tuple_))
        #only add unique (non-duplicate) values to the master pred_crater_dist
        if len(GT_crater_dist) > 0:
            for j in range(len(tuple_)):
                diff = (GT_crater_dist - tuple_[j])**2
                diffsum = np.asarray([sum(x) for x in diff])
                #long, lat, rad = GT_crater_dist.T
                #diffsum = np.asarray(25*(long-tuple_[j][0])**2 + 25*(lat-tuple_[j][1])**2 + (rad-tuple_[j][2])**2)
                index = diffsum < unique_thresh2
                if len(np.where(index==True)[0]) == 0: #unique value
                    GT_crater_dist = np.vstack((GT_crater_dist,tuple_[j]))
        else:
            GT_crater_dist = np.concatenate((GT_crater_dist,tuple_))

    GT_crater_dist = np.asarray(GT_crater_dist)
    np.save('%s/%s_GTcraterdist_n10000_new.npy'%(data_dir,data_prefix),GT_crater_dist)

if __name__ == '__main__':
    #args
    # ilen_1500_to_2500 settings
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
    
    n_imgs = 10016          #number of images to use for getting crater distribution.
    inv_color = 1           #**must be same setting as what model was trained on**
    rescale = 1             #**must be same setting as what model was trained on**
    ground_truth_only = 0   #get ground truth crater distribution only (from csvs), do not generate predictions
    unique_thresh2 = 1e-6   #duplicate threshold

    get_crater_dist(data_dir,data_prefix,csv_prefix,pickle_loc,model_loc,n_imgs,inv_color,rescale,ground_truth_only,unique_thresh2)

#    unique_thresh2 = [0.01,0.05,0.1,0.5,1,5,10]
#    for ut2 in unique_thresh2:
#        get_crater_dist(data_dir,data_prefix,csv_prefix,pickle_loc,model_loc,n_imgs,inv_color,rescale,ground_truth_only,ut2)
#        print ut2
    print "Script completed successfully"
