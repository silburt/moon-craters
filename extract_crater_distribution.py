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

def load_data(path, data_type):
    X = []
    X_id = []
    y = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)/255.
        X.append(img)
        y.append(np.array(Image.open('%smask.tiff'%f.split('.png')[0])))
        X_id.append(int(os.path.basename(f).split('lola_')[1].split('.png')[0]))
    return  X, y, X_id

def read_and_normalize_data(path, dim, data_type):
    data, target, id_ = load_data(path, data_type)
    data = np.array(data).astype('float32')             #convert to numpy, convert to float
    data = data.reshape(len(data),dim, dim, 1)          #add dummy third dimension, required for keras
    target = np.array(target).astype('float32')         #convert to numpy, convert to float
    print('%s shape:'%data_type, data.shape)
    return data, target, id_

def get_crater_dist(dir,type,n_imgs,modelpath,inv_color,rescale,ground_truth_dist):
    # properties of the dataset, shouldn't change (unless you use a different dataset)
    master_img_height_pix = 20000.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737                   #radius of the moon (km)
    dim = 256                       #output length (pixels)
    P = cPickle.load(open('%s/lolaout_%s.p'%(dir,type), 'r'))
    
    # get data
    path = {'train':'%s/Train_rings/'%dir, 'dev':'%s/Dev_rings/'%dir, 'test':'%s/Test_rings/'%dir}
    try:
        data=np.load('%s%s_data.npy'%(path[type],type))
        target=np.load('%s%s_target.npy'%(path[type],type))
        id=np.load('%s%s_id.npy'%(path[type],type))
        print "Successfully loaded %s files locally."%path[type]
    except:
        print "Couldnt find locally saved .npy files, loading from %s."%dir
        data, target, id = read_and_normalize_data(path[type], dim, type)
        np.save('%s%s_data.npy'%(path[type],type),data)
        np.save('%s%s_target.npy'%(path[type],type),target)
        np.save('%s%s_id.npy'%(path[type],type),id)

    data, target, id = data[:n_imgs], target[:n_imgs], id[:n_imgs]
    if inv_color==1 or rescale==1:
        print "inv_color=%d, rescale=%d, processing data"%(inv_color, rescale)
        data = rescale_and_invcolor(data, inv_color, rescale)

    # generate model predictions and fit template match
    model = load_model(modelpath)
    pred = model.predict(data.astype('float32'))

    # extract crater distribution
    pred_crater_dist = []
    print "Extracting crater radius distribution of %d %s files."%(n_imgs,type)
    for i in range(n_imgs):
        coords = template_match_target(pred[i])
        img_pix_height = P[id[i]]['box'][2] - P[id[i]]['box'][0]
        pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180)*(img_pix_height/dim)*r_moon
        _,_,radii = zip(*coords*pix_to_km)
        pred_crater_dist += list(radii)

    GT_crater_dist = []
    if ground_truth_dist == 1:
        print "Getting ground truth crater distribution."
        minrad, maxrad = 3, 75    #min/max radius (in pixels) required to include crater in target
        cutrad = 0.5              #0-1 range, if x+cutrad*r > img_width, remove, i.e. exclude craters ~half gone from image
        for id_ in id:
            csv = pd.read_csv('%slola_%s.csv'%(path[type],str(id_).zfill(5)))
            csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
            csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
            csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
            rad = csv['Diameter (pix)'].values/2
            GT_crater_dist += list(rad)

    pred_crater_dist, GT_crater_dist = np.asarray(pred_crater_dist), np.asarray(GT_crater_dist)
    np.save('%s%s_predcraterdist_n%d.npy'%(path[type],type,n_imgs),pred_crater_dist)
    np.save('%s%s_GTcraterdist_n%d.npy'%(path[type],type,n_imgs),GT_crater_dist)
    return pred_crater_dist, GT_crater_dist

if __name__ == '__main__':
    #args
    dir = 'datasets/rings'  #location of Train_rings/, Dev_rings/, Test_rings/ folders. Exclude final '/' in path.
    type = 'test'           #what to get crater distribution of: train, dev, test
    n_imgs = 30016          #number of images to use for getting crater distribution.
    ground_truth_dist = 1   #get ground truth crater distribution too (from csvs)
    
    modelpath = 'models/unet_s256_rings_nFL96.h5'
    inv_color = 1           #**must be same setting as what model was trained on**
    rescale = 1             #**must be same setting as what model was trained on**

    pred_crater_dist, GT_crater_dist = get_crater_dist(dir,type,n_imgs,modelpath,inv_color,rescale,ground_truth_dist)
