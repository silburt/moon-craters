#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameter to tune is unique_thresh2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.

import numpy as np

def extract_unique(pred, unique_thresh2, id, P):
    master_img_height_pix = 23040.  #number of pixels for height
    master_img_height_lat = 180.    #degrees used for latitude
    r_moon = 1737.4                 #radius of the moon (km)
    dim = 256                       #image dimension (pixels, assume dim=height=width)

    pred_crater_dist = np.empty([0,3])
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

    np.save('%s/test_uniquedist_ut%.1e.npy'%(dir,unique_thresh2),pred_crater_dist)

if __name__ == '__main__':
    #arrays = (long, lat, radii)
    dir = 'datasets/rings/Test_rings'
    pred = np.load('%s/test_modelpreds_n1000_new.npy'%dir)
    id = np.load('%s/test_id.npy'%dir)
    P = cPickle.load(open('%s/lolaout_test.p'%dir, 'r'))

    #unique_thresh2 = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    unique_thresh2 = [1e-1,1e-2,1e-3]
    for ut2 in unique_thresh2:
#        print "extracting unique ground truth craters, unique_thresh2=%.2f"%ut2
#        GT = extract_unique(truth, ut2, dir, output_nametruth)

        print "extracting unique predicted craters, unique_thresh2=%.2f"%ut2
        pred = extract_unique(pred, ut2, id, P)
