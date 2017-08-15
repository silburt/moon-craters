#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
#pred = np.load('datasets/ilen_1500_to_2500/ilen_1500/_predcraterdist_n1000.npy')
#truth = np.load('datasets/ilen_1500_to_2500/ilen_1500/_GTcraterdist_n1000.npy')

pred = np.load('datasets/rings/Test_rings/test_predcraterdist_unique_n30016.npy')
truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_unique_n30016.npy')

truthcsv = pd.read_csv('utils/alanalldata.csv')
long, lat, pred = pred.T
longT, latT, truth = truth.T

norm = False
nbins = 50

#investigating
inv = pred[(pred > 10)&(pred < 12)]
print len(inv)

#plt.hist(pred, nbins, range=[min(truth),max(truth)], normed=norm, label='pred')
plt.hist(pred, nbins, range=[10,20], normed=norm, label='pred')
plt.hist(truth, nbins, range=[10,20], normed=norm, alpha=0.5, label='ground truth')
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
#plt.savefig('output_dir/images/ilen2500_bin50.png')
plt.show()
