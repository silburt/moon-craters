#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameters to tune are thresh_longlat2 and thresh_rad2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.
#First you need to generate predictions from crater_distribution_get_modelpreds.py

import numpy as np
import cPickle
import pandas as pd
from utils.template_match_target import *
from utils.rescale_invcolor import *
import time
import glob

#########################
def add_unique_craters(tuple, crater_dist, thresh_longlat2, thresh_rad2):
    Long, Lat, Rad = crater_dist.T
    for j in range(len(tuple)):
        lo,la,r = tuple[j].T
        diff_longlat = (Long - lo)**2 + (Lat - la)**2
        Rad_ = Rad[diff_longlat < thresh_longlat2]
        if len(Rad_) > 0:
            diff_rad = ((Rad_ - r)/r)**2                #fractional radius change
            index = diff_rad < thresh_rad2
            if len(np.where(index==True)[0]) == 0:      #unique value determined from long/lat, then rad
                crater_dist = np.vstack((crater_dist,tuple[j]))
        else:                                           #unique value determined from long/lat alone
            crater_dist = np.vstack((crater_dist,tuple[j]))
    return crater_dist

#########################
def extract_unique_GT(GT_crater_dist, dir, id, thresh_longlat2, thresh_rad2, output_name):
    t1 = time.time()
    print "extracting unique ground truth craters, thresh_longlat2=%.2e, thresh_rad2=%.2e"%(thresh_longlat2, thresh_rad2)
    
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
                GT_crater_dist = add_unique_craters(tuple_, GT_crater_dist, thresh_longlat2, thresh_rad2)
            else:
                GT_crater_dist = np.concatenate((GT_crater_dist,tuple_))

    np.save(output_name,GT_crater_dist)
    print "Elapsed time for GT with thresh_longlat2=%.2e,thresh_rad2=%.2e is %f"%(thresh_longlat2,thresh_rad2,time.time() - t1)
    print ""
    return GT_crater_dist

#########################
def extract_unique_pred(pred_crater_dist, pred, id, P, thresh_longlat2, thresh_rad2, output_name):
    print "extracting unique predicted craters, thresh_longlat2=%.2e, thresh_rad2=%.2e"%(thresh_longlat2, thresh_rad2)
    
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = 256.0                     #image dimension (pixels, assume dim=height=width), needs to be float
    
    N_matches_tot = 0
    for i in range(len(pred)):
        print i, len(pred_crater_dist)
        coords = template_match_target(pred[i])
        if len(coords) > 0:
            P_ = P[id[i]]
            img_pix_height = float(P_['box'][2] - P_['box'][0])
            pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/dim)*r_moon
            long_pix,lat_pix,radii_pix = coords.T
            radii_km = radii_pix*pix_to_km
            long_deg = P_['llbd'][0] + (P_['llbd'][1]-P_['llbd'][0])*(long_pix/dim)
            lat_deg = P_['llbd'][3] - (P_['llbd'][3]-P_['llbd'][2])*(lat_pix/dim)
            tuple_ = np.column_stack((long_deg,lat_deg,radii_km))
            N_matches_tot += len(coords)
            
            #only add unique (non-duplicate) values to the master pred_crater_dist
            if len(pred_crater_dist) > 0:
                pred_crater_dist = add_unique_craters(tuple_, pred_crater_dist, thresh_longlat2, thresh_rad2)
            else:
                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

    np.save(output_name,pred_crater_dist)
    return pred_crater_dist

#########################
if __name__ == '__main__':
    datatypes = ['train','test']
    
    #primary model *predictions* paths
    dirs = ['datasets/rings/Train_rings','datasets/rings/Test_rings']
    nimgs = 30016       #1000, 10016, 30016
    
    #augmented highilen predictions (generated by Charles)
    highilen_dir = 'datasets/highilen'

    #grid search thresholds
    #thresholds = zip(thresh_longlat2*len(thresh_rad2),np.repeat(thresh_rad2,len(thresh_longlat2)))
    #thresholds = [(0.6,0.6),(0.7,0.7),(0.8,0.8),(0.9,0.9)]
    thresholds = [(0.6,0.6)]

    #########MAIN LOOP#########
    for i in range(len(datatypes)):
        datatype, dir = datatypes[i], dirs[i]
        
        #primary model preds
        file = '%s_modelpreds_n%d_new.npy'%(datatype,nimgs)
        pred = np.load('%s/%s'%(dir,file))
        id = np.load('%s/%s_id.npy'%(dir,datatype))[0:len(pred)]
        P = cPickle.load(open('%s/lolaout_%s.p'%(dir,datatype), 'r'))
        print "Using Preds: %s/%s"%(dir,file)
        print ""
        
        #augmented dataset
        highilen_file = 'highilen_%s_modelpreds_n5000_new.npy'%datatype
        highilen_pred = np.load('%s/%s'%(highilen_dir,highilen_file))
        highilen_P = cPickle.load(open('%s/%soutp_v2.p'%(highilen_dir,datatype), 'r'))
        highilen_id = range(len(highilen_P))    #preds and P have same index, dummy id array
        
        for llt2,rt2 in thresholds:
            t1 = time.time()
            print "thresh_longlat2=%.2e,thresh_rad2=%.2e"%(llt2,rt2)
            
            pred_output_name = '%s/%s_highilenpred_llt%.1e_rt%.1e_n%d.npy'%(dir,datatype,llt2,rt2,len(pred))   #save directory
            pred_crater_dist = np.empty([0,3])  #reset empty crater dist
            pred_crater_dist = extract_unique_pred(pred_crater_dist, pred, id, P, llt2, rt2, pred_output_name)

            print "finished primary dataset, analyzing augmented ilen datasets"
            pred_crater_dist = extract_unique_pred(pred_crater_dist, highilen_pred, highilen_id, highilen_P, llt2, rt2, pred_output_name)
        
            print "Elapsed time for pred with thresh_longlat2=%.2e,thresh_rad2=%.2e is %f"%(llt2,rt2,time.time()-t1)
            print ""

#    print "getting ground truth"
#    GT_output_name = '%s/%s_highilenGT_llt%.1e_rt%.1e_n%d.npy'%(dir,datatype,llt2,rt2,len(pred))   #save directory
#    GT_crater_dist = np.empty([0,3])
#    GT_crater_dist = extract_unique_GT(GT_crater_dist, dir, id, llt2, rt2, GT_output_name)
