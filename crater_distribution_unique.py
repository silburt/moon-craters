#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameter to tune is unique_thresh2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.

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
    
    np.save('%s/%s_ut2_%.1e.npy'%(dir,name,unique_thresh2),data)
    return data

if __name__ == '__main__':
    #arrays = (long, lat, radii)
    dir = 'datasets/rings/Test_rings'
    
    truth = np.load('%s/test_GTcraterdist_full.npy'%dir)
    pred = np.load('%s/test_predcraterdist_full.npy'%dir)

    output_nametruth = 'unique_GTcraters'
    output_namepred = 'unique_predcraters'

    #unique_thresh2 = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    unique_thresh2 = [1e-6]
    for ut2 in unique_thresh2:
#        print "extracting unique ground truth craters, unique_thresh2=%.2f"%ut2
#        GT = extract_unique(truth, ut2, dir, output_nametruth)

        print "extracting unique predicted craters, unique_thresh2=%.2f"%ut2
        pred = extract_unique(pred, ut2, dir, output_namepred)
