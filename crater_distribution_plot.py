#plot the results on local machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle
from scipy import stats
import os

norm = False
nbins = 50
maxrad = 45
cdf = 0
truth_datatype = 'test'

###############Original ground truth dataset################
truthalan = pd.read_csv('utils/alanalldata.csv')
craters_names = ["Long", "Lat", "Radius (deg)","Diameter (km)", "D_range", "p", "Name"]
craters_types = [float, float, float, float, float, int, str]
truthLU = pd.read_csv(open('utils/LU78287GT.csv', 'r'), sep=',',
                        usecols=list(range(1, 8)), header=0, engine="c", encoding = "ISO-8859-1",
                        names=craters_names, dtype=dict(zip(craters_names, craters_types)))
#limits - train
print "truth_datatype = %s"%truth_datatype
if truth_datatype == 'train':
    truthalan = truthalan[truthalan['Long']<-60]        #region of train data
    truthLU = truthLU[(truthLU['Long']<-60)&(truthLU['Diameter (km)']>20.)]
elif truth_datatype == 'test':
    truthalan = truthalan[truthalan['Long']>60]        #region of test data
    truthLU = truthLU[(truthLU['Long']>60)&(truthLU['Diameter (km)']>20.)]

rad_truth = np.concatenate((truthalan['Diameter (km)'].values/2.,truthLU['Diameter (km)'].values/2.))
############################################################

#########################################
#These dists were generated by excluding radii from the GT with radii larger than the maximum detected radius in the corresponding model-predicted image. This is done on an image-by-image basis. These datasets were generated via crater_distribution_extract_debug2.py
#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_debug2_n30016.npy')
#GT = np.load('datasets/rings/Test_rings/test_GTcraterdist_debug2_n30016.npy')
#########################################

#*****train set***** - reducing to a unique distribution by varying long/lat and rad thresholds
#From crater_distribution_unique_full.py
#########################################
#filename = 'datasets/rings/Train_rings/train_uniquepred_llt5.0e-01_rt5.0e-01_n10016.npy'
#filename = 'datasets/rings/Train_rings/train_highilenpred_llt6.0e-01_rt6.0e-01_n10016.npy'
filename = 'datasets/rings/Test_rings/test_highilenpred_llt6.0e-01_rt6.0e-01_n29976.npy'
#filename = 'datasets/rings/Test_rings/test_highlowilenpred_llt6.0e-01_rt6.0e-01_n10016.npy'
#filename = 'datasets/rings/Train_rings/train_highilenpred_llt6.0e-01_rt6.0e-01_n30016.npy'

#GT = np.load('datasets/rings/Test_rings/test_uniqueGT_llt1.0e-06_rt1.0e-06_n10016.npy') #unique distribution for 10,000 images
#########################################
#load data
pred = np.load(filename)
long, lat, rad = pred.T

outname = os.path.basename(filename).split('.npy')[0]
ext = ''
if cdf == 1:
    rad = rad[(rad>=np.min(rad_truth))&(rad<=maxrad)]
    rad_truth = rad_truth[rad_truth<=maxrad]
    plt.plot(np.sort(rad),np.arange(0,1,1./len(rad)),label='pred extract')
    plt.plot(np.sort(rad_truth),np.arange(0,1,1./len(rad_truth)),label='ground truth')
    plt.xlabel('crater radius (km)')
    plt.ylabel('cdf')
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.savefig('output_dir/images/%s_cdf.png'%outname)
else:
    n, dist_bins, _ = plt.hist(rad, nbins, range=[min(rad_truth),maxrad], normed=norm, label='pred extract')
    plt.hist(rad_truth, nbins, range=[min(rad_truth),maxrad], normed=norm, alpha=0.5, label='ground truth')
    
    #extended
    bin_diff = dist_bins[1]-dist_bins[0]
    binnies = [dist_bins[0]-bin_diff, dist_bins[0]]
    print binnies
    plt.hist(rad, bins=binnies, normed=norm, label='new craters')
    ext = '_ext'
    
    plt.xlabel('crater radius (km)')
    plt.yscale('log')
    plt.legend()
    plt.savefig('output_dir/images/%s%s.png'%(outname,ext))
    #plt.savefig('output_dir/images/debug2_removehighradGT_n30016.png')

#Theres a problem with this because I think the KStest chooses N=number of bins, which makes anything significant enough. I want a log-ks test with the high N included. 
ks_test = 0
if ks_test == 1:    #Do log-ks-test by binning the data, logging the counts in each bin, and then do cdf.
    print "Doing KS Test"
    dists = []
    for dist in [rad, rad_truth]:
        print len(dist)
        n, bins = np.histogram(dist, nbins, range=[min(rad_truth),maxrad])
        n = np.log10(n)
        count = []
        count.append(n[0])
        for i in range(1,len(n)):
            count.append(n[i]+count[i-1])
        dists.append(np.asarray(count)/count[-1])
    print stats.ks_2samp(dists[0], dists[1])

plt.show()

##########################################################################################
################## extra stuff ##################
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
#pred = np.load('datasets/rings/Test_rings/test_uniquepred_llt1.0e+00_rt5.0e-01_n10016.npy')
#pred = np.load('datasets/rings/Test_rings/test_uniquepred_llt1.0e+00_rt1.0e+00_n10016_nonaugmented.npy')

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

#get list of most frequent values and confirm that they're caused by constant ilen values
#master_img_height_pix = 23040.  #number of pixels for height
#master_img_height_lat = 180.    #degrees used for latitude
#r_moon = 1737.4                 #radius of the moon (km)
#dim = 256.0                     #image dimension (pixels, assume dim=height=width), needs to be float
#img_pix_height = np.asarray([1500,1750,2250])   #different scales in the image
#pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/dim)*r_moon
#arr = rad.copy()
#print "r_km, \t    r_pix_1500 \t r_pix_1750 \t r_pix_2250 \t N_r_km"
#for i in range(10):
#    u, indices = np.unique(arr, return_inverse=True)
#    r_km = u[np.argmax(np.bincount(indices))]
#    print r_km, r_km/pix_to_km, len(arr[arr==r_km])
#    arr = arr[arr != r_km]
#pixel_rad = np.arange(20)
#for i,img_pix_height in enumerate([1500,1750,2250]):
#    pix_to_km = (master_img_height_lat/master_img_height_pix)*(np.pi/180.0)*(img_pix_height/dim)*r_moon
#    rad_km = pixel_rad*pix_to_km
#    plt.plot([rad_km,rad_km],[0,1],'k--')
#rad_plot = rad[(rad>=min(rad_truth))&(rad<maxrad)]
#plt.plot(np.sort(rad_plot),np.arange(len(rad_plot))/float(len(rad_plot)),label='pred_extract',lw=2)
##plt.plot(np.sort(rad_truth),np.arange(len(rad_truth))/float(len(rad_truth)),label='ground_truth',lw=4)
#plt.xlabel('crater radius (km)')
#plt.xlim([min(rad_truth),maxrad])
#plt.legend(loc='lower right')
#plt.savefig('output_dir/images/unique_llt2=0.5_rt2=0.5_cdf.png')
################## extra stuff ##################
