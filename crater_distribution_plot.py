#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle

norm = False
nbins = 50

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
#pred = np.load('datasets/ilen_1500_to_2500/ilen_1500/_predcraterdist_n1000.npy')
#truth = np.load('datasets/ilen_1500_to_2500/ilen_1500/_GTcraterdist_n1000.npy')

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_unique_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_unique_n30016.npy')
#truthcsv = pd.read_csv('utils/alanalldata.csv')
#long, lat, pred = pred.T
#longT, latT, truth = truth.T

#best
pred = np.load('datasets/rings/Test_rings/test_predcraterdist_debug_n30016.npy')
truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
rad, scale = pred.T
plt.hist(rad*scale, nbins, range=[min(truth),max(truth)], normed=norm, label='scale')
plt.hist(truth, nbins, normed=norm, alpha=0.5, label='ground truth')

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_full.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_full.npy')
#long_pred, lat_pred, rad_pred = pred.T
#long_GT, lat_GT, rad_GT = truth.T
#plt.hist(rad_pred, nbins, range=[min(rad_GT),max(rad_GT)], normed=norm, label='scale')
#plt.hist(rad_GT, nbins, normed=norm, alpha=0.5, label='ground truth')

#plt.hist(pred, nbins, range=[5,20], normed=norm, label='pred')
#plt.hist(truth, nbins, range=[5,20], normed=norm, alpha=0.5, label='ground truth')

#plt.hist(rad, nbins, range=[5,20], normed=norm, label='pred')

plt.legend()
plt.yscale('log')


#plot pickle
#P = cPickle.load(open('datasets/rings/Test_rings/lolaout_test.p', 'r'))
#scales = []
#for i in range(len(P)):
#    P_ = P[i]
#    scales.append(P_['box'][2] - P_['box'][0])
#
#plt.hist(scales, nbins)
#plt.xlabel('ilen image scales')

#final output
#plt.savefig('output_dir/images/raddist_bin30.png')
plt.show()
