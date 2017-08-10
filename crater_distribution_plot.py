#plot the results on local machine

import numpy as np
import matplotlib.pyplot as plt

#pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
#truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')
pred = np.load('datasets/ilen_1500_to_2500/ilen_1500/_predcraterdist_n1000.npy')
truth = np.load('datasets/ilen_1500_to_2500/ilen_1500/_GTcraterdist_n1000.npy')

norm = False
nbins = 100

#investigating
inv = pred[(pred > 10)&(pred < 12)]
print len(inv)

#plt.hist(pred, nbins, range=[min(truth),max(truth)], normed=norm, label='pred')
plt.hist(pred, nbins, range=[10,20], normed=norm, label='pred')
#plt.hist(truth, nbins, normed=norm, alpha=0.5, label='ground truth')
plt.legend()
plt.yscale('log')
#plt.savefig('output_dir/images/ilen2500_bin50.png')
plt.show()
