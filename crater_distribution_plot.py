#plot the results on local machine

import numpy as np
import matplotlib.pyplot as plt

pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad0.85.npy')

norm = False

plt.hist(pred, 30, range=[min(truth),max(truth)], normed=norm, label='pred')
#plt.hist(pred, 30, normed=norm, label='pred')
plt.hist(truth, 30, normed=norm, alpha=0.5, label='ground truth')
plt.legend()
#plt.savefig('output_dir/images/crater_dist_comp_samemin_norm.png')
plt.show()
