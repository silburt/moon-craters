#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameter to tune is thresh_unique2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.

import numpy as np

def extract_unique(data, unique_thresh2, dir, name):
    i, N = 0, len(data)
    while i < N:
        data_cut = data[i:]                                         #compare only with higher indices, include entry i from data
        diff = (data_cut - data_cut[0])**2
        diffsum = np.asarray([sum(x) for x in diff])                #sum((lat-lat_i)^2 + (long-long_i)^2 + (rad-rad_i)^2)
        index = diffsum < unique_thresh2
        dup = np.where(index == True)[0]                            #get duplicates
        if len(dup) > 1:                                            #will be at least 1 match for 0th slot
            data[i] = np.mean(data_cut[dup],axis=0)                 #take average of duplicate coords, store in *original* data array
            keep = np.where(index==False)[0]                        #indices of non-duplicate entries
            data = np.concatenate((data[0:i+1],data_cut[keep]))     #concatenate and remove duplicates
        N, i = len(data), i+1
    
    np.save('%s/%s_ut2%.3f.npy'%(dir,name,unique_thresh2),data)
    return data

if __name__ == '__main__':
    #arrays = (long, lat, radii)
    dir = 'datasets/rings/Test_rings'
    pred = np.load('%s/test_predcraterdist_full.npy'%dir)
    truth = np.load('%s/test_GTcraterdist_full.npy'%dir)

    output_nameGT = 'unique_GTcraters'
    output_namepred = 'unique_predcraters'

    unique_thresh2 = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    for ut2 in thresh_unique2:
        print "extracting unique ground truth craters, thresh_unique=%.2f"%ut2
        GT = extract_unique(truth, ut2, dir, output_nameGT)
        
#        print "extracting unique predicted craters, thresh_unique=%.2f"%ut2
#        pred = extract_unique(pred, ut2, dir, output_namepred)
