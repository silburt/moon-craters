#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle

norm = False
nbins = 50

truthcsv1 = pd.read_csv('utils/alanalldata.csv')
truthcsv1 = truthcsv1[truthcsv1['Long']>60]        #region of test data
craters_names = ["Long", "Lat", "Radius (deg)","Diameter (km)", "D_range", "p", "Name"]
craters_types = [float, float, float, float, float, int, str]
truthcsv2 = pd.read_csv(open('utils/LU78287GT.csv', 'r'), sep=',',
                        usecols=list(range(1, 8)), header=0, engine="c", encoding = "ISO-8859-1",
                        names=craters_names, dtype=dict(zip(craters_names, craters_types)))
truthcsv2 = truthcsv2[(truthcsv2['Long']>60)&(truthcsv2['Diameter (km)']>20.)]

rad_alan = np.concatenate((truthcsv1['Diameter (km)'].values/2.,truthcsv2['Diameter (km)'].values/2.))
print len(rad_alan)

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
#pred = np.load('datasets/ilen_1500_to_2500/ilen_1500/_predcraterdist_n1000.npy')
#truth = np.load('datasets/ilen_1500_to_2500/ilen_1500/_GTcraterdist_n1000.npy')

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_unique_n30016.npy')
#long, lat, rad = pred.T
#truth = np.load('datasets/rings/Test_rings/unique_GTcraters_ut2_1.0e-06.npy')
#longT, latT, radT = truth.T
#plt.hist(rad, nbins, range=[min(rad_alan),max(rad_alan)], normed=norm, label='scale')
#plt.hist(radT, nbins, range=[min(rad_alan),max(rad_alan)], normed=norm, label='scale')
#plt.hist(rad_alan, nbins, normed=norm, alpha=0.5, label='ground truth')

#best - cannot reproduce for some reason
#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_debug_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
#rad, scale = pred.T
#plt.hist(rad*scale, nbins, range=[min(truth),max(truth)], normed=norm, label='scale')
#plt.hist(truth, nbins, normed=norm, alpha=0.5, label='ground truth')

#testing how the distribution changes between unique and non-unique crater distributions
#truth1 = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
#truth_unique = np.load('datasets/rings/Test_rings/unique_GTcraters_ut2_1.0e-06.npy')
#_, _, rad_TU = truth_unique.T
#plt.hist(truth1, nbins, normed=True, label='initial')
#plt.hist(rad_TU, nbins, normed=True, alpha=0.5, label='unique')

pred2 = np.load('datasets/rings/Test_rings/test_predcraterdist_debug_n30016_old.npy')
rad2, scale2 = pred2.T
pred = np.load('datasets/rings/Test_rings/test_predcraterdist_debug_n10000.npy')
truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_debug_n10000.npy')
long, lat, rad, scale, P0, P1, P2, P3 = pred.T
plt.hist(rad*scale, nbins, range=[min(truth),max(truth)], normed=norm, label='scale')
plt.hist(rad2*scale2, nbins, range=[min(truth),max(truth)], normed=norm, label='pred2', alpha=0.5)
plt.hist(rad_alan, nbins, range=[min(truth),max(truth)], normed=norm, alpha=0.5, label='alan ground truth')

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
