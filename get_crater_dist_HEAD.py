#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameters to tune are thresh_longlat2 and thresh_rad2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.
#First you need to generate predictions from crater_distribution_get_modelpreds.py

import numpy as np, h5py
from utils.template_match_target import *
from utils.preprocessing import *
import glob
import sys
from keras.models import load_model

#########################
def get_id(i, zeropad=5):
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)

#########################
def get_model_preds(CP):
    dim, n_imgs, dtype = CP['dim'], CP['n_imgs'], CP['datatype']

    #data = h5py.File('%s%s_images.hdf5'%(CP['dir_data'],dtype), 'r')
    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    preprocess(Data)

    model = load_model(CP['model'])
    preds = model.predict(Data[dtype][0])
    
    #save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    print "Successfully generated and saved model predictions."
    return preds

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
def extract_crater_dist(CP, pred_crater_dist):
    
    #load/generate model preds
    try:
        preds = h5py.File(CP['dir_preds'],'r')[CP['datatype']]
        print "Loaded model predictions successfully"
    except:
        print "Couldnt load model predictions, generating"
        preds = get_model_preds(CP)
    
    # need for long/lat bounds
    #P = h5py.File('%s%s_images.hdf5'%(CP['dir_data'],CP['datatype']), 'r')
    P = h5py.File(CP['dir_data'], 'r')
    llbd, pbd = 'longlat_bounds', 'pix_bounds'
    
    master_img_height_pix = 30720.  #number of pixels for height
    master_img_height_lat = 120.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = float(CP['dim'])          #image dimension (pixels, assume dim=height=width), needs to be float

    N_matches_tot = 0
    for i in range(CP['n_imgs']):
        print i, len(pred_crater_dist)
        coords = template_match_target(preds[i])
        if len(coords) > 0:
            id = get_id(i)
            D = float(P[pbd][id][3] - P[pbd][id][1])/dim    #accounts for image downsampling by some factor D
            pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*r_moon*D
            long_pix,lat_pix,radii_pix = coords.T
            radii_km = radii_pix*pix_to_km
            long_deg = P[llbd][id][0] + (P[llbd][id][1]-P[llbd][id][0])*(long_pix/dim)
            lat_deg = P[llbd][id][3] - (P[llbd][id][3]-P[llbd][id][2])*(lat_pix/dim)
            tuple_ = np.column_stack((long_deg,lat_deg,radii_km))
            N_matches_tot += len(coords)
            
            #only add unique (non-duplicate) values to the master pred_crater_dist
            if len(pred_crater_dist) > 0:
                pred_crater_dist = add_unique_craters(tuple_, pred_crater_dist, CP['llt2'], CP['rt2'])
            else:
                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

    np.save(CP['dir_result'],pred_crater_dist)
    return pred_crater_dist

#########################
if __name__ == '__main__':
    # Arguments
    CP = {}
    CP['dir_data'] = '/scratch/m/mhvk/czhu/moondata/fullilen_uncropped/dev_wideilen_images.hdf5'
    #CP['dir_data'] = 'datasets/HEAD/'
    
    # Tuned Hyperparameters - Shouldn't really change
    CP['llt2'] = float(sys.argv[1])    #D_{L,L} from Silburt et. al (2017)
    CP['rt2'] = float(sys.argv[2])     #D_{R} from Silburt et. al (2017)
    
    CP['datatype'] = 'dev'
    CP['n_imgs'] = 30000
    CP['dir_preds'] = 'datasets/HEAD/HEADwideilen_%spreds_n%d.hdf5'%(CP['datatype'],CP['n_imgs'])
    CP['dir_result'] = 'datasets/HEAD/HEAD_%s_craterdist_llt%.2f_rt%.2f.npy'%(CP['datatype'], CP['llt2'], CP['rt2'])
    
    #Needed to generate model_preds if they don't exist yet
    CP['model'] = 'models/HEAD_FINALL.h5'
    CP['dim'] = 256

    pred_crater_dist = np.empty([0,3])
    pred_crater_dist = extract_crater_dist(CP, pred_crater_dist)
