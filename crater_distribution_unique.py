#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameter to tune is unique_thresh2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.

import numpy as np
import cPickle
import pandas as pd
from utils.template_match_target import *
import time
import glob

#########################
def extract_unique_GT(dir, unique_thresh2, id):
    t1 = time.time()
    print "extracting unique ground truth craters, unique_thresh2=%.2f"%ut2
    
    # hyperparameters
    minrad, maxrad = 3, 50  #min/max radius (in pixels) required to include crater in target
    cutrad = 1              #0-1 range, if x+cutrad*r > dim, remove, higher cutrad = larger % of circle required
    dim = 256
    
    GT_crater_dist = np.empty([0,3])
    for i,id_ in enumerate(id):
        print i, len(GT_crater_dist)
        csv = pd.read_csv('%s/lola_%s.csv'%(dir,str(id_).zfill(5)))
        csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
        csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2. <= dim)]
        csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2. <= dim)]
        csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2. > 0)]
        csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2. > 0)]
        if len(csv) > 0:
            GT_radii = csv['Diameter (km)'].values/2.
            GT_lat = csv['Lat']
            GT_long = csv['Long']
            tuple_ = np.column_stack((GT_long,GT_lat,GT_radii))
            if len(GT_crater_dist) > 0:
                for j in range(len(tuple_)):
                    diff = (GT_crater_dist - tuple_[j])**2
                    diffsum = np.asarray([sum(x) for x in diff])
                    index = diffsum < unique_thresh2
                    if len(np.where(index==True)[0]) == 0: #unique value
                        GT_crater_dist = np.vstack((GT_crater_dist,tuple_[j]))
            else:
                GT_crater_dist = np.concatenate((GT_crater_dist,tuple_))

    np.save('%s/test_uniqueGT_ut%.1e_n%d.npy'%(dir,unique_thresh2,len(id)),pred_crater_dist)
    print "Elapsed time for GT with unique_thresh2=%.2f is %f"%(ut2,time.time() - t1)
    print ""

#########################
def extract_unique_pred(pred, unique_thresh2, id, P):
    t1 = time.time()
    print "extracting unique predicted craters, unique_thresh2=%.2f"%ut2
    
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = 256                       #image dimension (pixels, assume dim=height=width)
    
    pred_crater_dist = np.empty([0,3])
    N_matches_tot = 0
    for i in range(len(pred)):
        print i, len(pred_crater_dist)
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
            N_matches_tot += len(coords)
            
            #only add unique (non-duplicate) values to the master pred_crater_dist
            if len(pred_crater_dist) > 0:
                for j in range(len(tuple_)):
                    diff = (pred_crater_dist - tuple_[j])**2
                    diffsum = np.asarray([sum(x) for x in diff])
                    index = diffsum < unique_thresh2
                    if len(np.where(index==True)[0]) == 0: #unique value
                        pred_crater_dist = np.vstack((pred_crater_dist,tuple_[j]))
            else:
                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

    np.save('%s/test_uniquepred_ut%.1e_n%d.npy'%(dir,unique_thresh2,len(pred)),pred_crater_dist)
    print "Total Number of Matches for %f: %d"%(unique_thresh2,N_matches_tot)
    print "Elapsed time for pred unique_thresh2=%.2f is %f"%(ut2,time.time() - t1)
    print ""

#########################
if __name__ == '__main__':
    #arrays = (long, lat, radii)
    dir = 'datasets/rings/Test_rings'
    file = 'test_modelpreds_n10016_new.npy'
    #file = 'test_modelpreds_n1000_new.npy'

    pred = np.load('%s/%s'%(dir,file))
    id = np.load('%s/test_id.npy'%dir)[0:len(pred)]
    P = cPickle.load(open('%s/lolaout_test.p'%dir, 'r'))
    
    print "Using Preds: %s/%s"%(dir,file)
    print ""

    #unique_thresh2 = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    unique_thresh2 = [3,1,0.5,0.1,1e-2]
    for ut2 in unique_thresh2:
        extract_unique_GT(dir, ut2, id)
        #extract_unique_pred(pred, ut2, id, P)

