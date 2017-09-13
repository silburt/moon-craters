#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle

norm = False
nbins = 50

#Original ground truth dataset
truthalan = pd.read_csv('utils/alanalldata.csv')
truthalan = truthalan[truthalan['Long']<-60]        #region of train data
craters_names = ["Long", "Lat", "Radius (deg)","Diameter (km)", "D_range", "p", "Name"]
craters_types = [float, float, float, float, float, int, str]
truthLU = pd.read_csv(open('utils/LU78287GT.csv', 'r'), sep=',',
                        usecols=list(range(1, 8)), header=0, engine="c", encoding = "ISO-8859-1",
                        names=craters_names, dtype=dict(zip(craters_names, craters_types)))
truthLU = truthLU[(truthLU['Long']<-60)&(truthLU['Diameter (km)']>20.)]

rad_truth = np.concatenate((truthalan['Diameter (km)'].values/2.,truthLU['Diameter (km)'].values/2.))
#rad_truth = truthalan['Diameter (km)'].values/2.

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

#test set
#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_debug2_n30016.npy')
#pred = np.load('datasets/rings/Test_rings/test_uniquepred_llt1.0e+00_rt5.0e-01_n10016.npy')
#pred = np.load('datasets/rings/Test_rings/test_uniquepred_llt1.0e+00_rt1.0e+00_n10016_nonaugmented.npy')
#GT = np.load('datasets/rings/Test_rings/test_GTcraterdist_debug2_n30016.npy')

#train set
pred = np.load('datasets/rings/Train_rings/train_uniquepred_llt1.0e+00_rt1.0e+00_n10016.npy')
long, lat, rad = pred.T
GT = np.load('datasets/rings/Test_rings/test_uniqueGT_llt1.0e-06_rt1.0e-06_n10016.npy') #unique distribution for 10,000 images
longGT, latGT, radGT = GT.T
print len(pred)

plt.hist(rad, nbins, range=[min(rad_truth),50], normed=norm, label='pred extract')
#plt.hist(radGT, nbins, range=[min(rad_truth),50], alpha=0.5, normed=norm,label='GT extract')
plt.hist(rad_truth, nbins, range=[min(rad_truth),50], normed=norm, alpha=0.5, label='ground truth')

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_full.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_full.npy')
#long_pred, lat_pred, rad_pred = pred.T
#long_GT, lat_GT, rad_GT = truth.T
#plt.hist(rad_pred, nbins, range=[min(rad_GT),max(rad_GT)], normed=norm, label='scale')
#plt.hist(rad_GT, nbins, normed=norm, alpha=0.5, label='ground truth')

#plt.hist(pred, nbins, range=[5,20], normed=norm, label='pred')
#plt.hist(truth, nbins, range=[5,20], normed=norm, alpha=0.5, label='ground truth')

#plt.hist(rad, nbins, range=[5,20], normed=norm, label='pred')

plt.xlabel('crater radius (km)')
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
plt.savefig('output_dir/images/unique_llt2=1_rt2=1.png')
plt.show()
