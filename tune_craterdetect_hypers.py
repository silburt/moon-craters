# The purpose of this macro is to tune the hyperparameters of the crater detection algorithm
# on the validation set, namely:
# match_thresh2, template_thresh and target_thresh.

import numpy as np
import pandas as pd
import sys
from utils.template_match_target import *

minrad, maxrad = 2, 50

def prep_csvs(dir, datatype, ids, nimgs):
    cutrad, dim = 1, 256
    try:
        csvs = np.load('%s/%s_csvs_n%d_Oct2017.npy'%(dir,datatype,nimgs))
    except:
        csvs = []
        for i in range(nimgs):
            csv_name = '%s/lola_%s.csv'%(dir,str(ids[i]).zfill(5))
            csv = pd.read_csv(csv_name)
            # prune csv list for small/large/half craters
            csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
            csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
            csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
            csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
            if len(csv) < 3:
                csvs.append([-1])
            else:
                csv_coords = np.asarray((csv['x'],csv['y'],csv['Diameter (pix)']/2)).T
                csvs.append(csv_coords)
        np.save('%s/%s_csvs_n%d_Oct2017.npy'%(dir,datatype,nimgs), csvs)
    print "successfully loaded csvs"
    return csvs

def get_recall(preds, csvs, nimgs, match_thresh2, template_thresh, target_thresh):
    recall, precision, f1 = [], [], []
    for i in range(nimgs):
        if len(csvs[i]) < 3:
            continue
        N_match, N_csv, N_templ, maxr, csv_duplicate_flag = template_match_target_to_csv(preds[i], csvs[i], minrad, maxrad, match_thresh2, template_thresh, target_thresh)
        p = float(N_match)/float(N_match + (N_templ-N_match))   #assuming all unmatched detected circles are false positives
        r = float(N_match)/float(N_csv)                         #N_csv = tp + fn, i.e. total ground truth matches
        recall.append(r); precision.append(p); f1.append(2*r*p/(r+p))

    print "match_thresh2=%f, template_thresh=%f, target_thresh=%f"%(match_thresh2, template_thresh, target_thresh)
    print "mean and std of N_match/N_csv (recall) = %f, %f"%(np.mean(recall), np.std(recall))
    print "mean and std of N_match/(N_match + (N_templ-N_match)) (precision) = %f, %f"%(np.mean(precision), np.std(precision))
    print "mean and std of 2rp/(r+p) (F1 score) = %f, %f"%(np.mean(f1), np.std(f1))

if __name__ == '__main__':
    #data parameters
    dir = 'datasets/rings/Dev_rings'    #location of model predictions. Exclude final '/' in path.
    datatype = 'dev'
    nimgs = 5000                        #1000, 10016, 30016
    
    #load hyperparams
    match_thresh2 = float(sys.argv[1])
    template_thresh = float(sys.argv[2])
    target_thresh = float(sys.argv[3])
    
    #load data
    file = '%s_modelpreds_n%d_Oct2017.npy'%(datatype,nimgs)
    preds = np.load('%s/%s'%(dir,file))
    ids = np.load('%s/%s_id.npy'%(dir,datatype))        #number for lola_X.csv

    csvs = prep_csvs(dir,datatype,ids,nimgs)

    get_recall(preds, csvs, nimgs, match_thresh2, template_thresh, target_thresh)
    print "finised successfully"

#    # Main Loop
#    for ma2,te,ta in params:
#        get_recall(preds, csvs, nimgs, ma2, te, ta)
