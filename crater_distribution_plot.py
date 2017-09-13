#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle
from scipy import stats

norm = False
nbins = 60

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
pred = np.load('datasets/rings/Train_rings/train_uniquepred_llt5.0e-01_rt5.0e-01_n10016.npy')
long, lat, rad = pred.T
GT = np.load('datasets/rings/Test_rings/test_uniqueGT_llt1.0e-06_rt1.0e-06_n10016.npy') #unique distribution for 10,000 images
longGT, latGT, radGT = GT.T
print len(pred)

plt.hist(rad, nbins, range=[min(rad_truth),50], normed=norm, label='pred extract')
#plt.hist(radGT, nbins, range=[min(rad_truth),50], alpha=0.5, normed=norm,label='GT extract')
plt.hist(rad_truth, nbins, range=[min(rad_truth),50], normed=norm, alpha=0.5, label='ground truth')

#get list of most frequent values and confirm that they're caused by constant ilen values
master_img_height_pix = 23040.  #number of pixels for height
master_img_height_lat = 180.    #degrees used for latitude
r_moon = 1737.4                 #radius of the moon (km)
dim = 256.0                     #image dimension (pixels, assume dim=height=width), needs to be float
img_pix_height = np.asarray([1500,1750,2250])   #different scales in the image
pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/dim)*r_moon

arr = rad.copy()
print "r_km, \t    r_pix_1500 \t r_pix_1750 \t r_pix_2250 \t N_r_km"
for i in range(10):
    u, indices = np.unique(arr, return_inverse=True)
    r_km = u[np.argmax(np.bincount(indices))]
    print r_km, r_km/pix_to_km, len(arr[arr==r_km])
    arr = arr[arr != r_km]

plt.xlabel('crater radius (km)')
plt.legend()
plt.yscale('log')
plt.savefig('output_dir/images/unique_llt2=0.5_rt2=0.5.png')

#ks test
#print stats.ks_2samp(rad, rad_truth)

'''
#plot pickle
#P = cPickle.load(open('datasets/ilen_1500_to_2500/ilen_1750/outp_p0.p', 'r'))
P = cPickle.load(open('datasets/rings/Train_rings/lolaout_train.p', 'r'))
scales = []
for i in range(len(P)):
    P_ = P[i]
    scales.append(P_['box'][2] - P_['box'][0])

print np.mean(scales), np.min(scales), np.max(scales)
plt.hist(scales, nbins)
plt.xlabel('ilen image scales')
'''
plt.show()
