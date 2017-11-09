# This calculates the pos_weight value that is hardcoded into weighted_binary_cross_entropy.
# You want pos_weight (i.e. positive weights, given to pixels=1) equal to the ratio of zeros over ones

import numpy as np

d = np.load('datasets/rings/Train_rings/train_target.npy')

zeros = len(np.where(d==0)[0])
ones = len(np.where(d==1)[0])

print zeros/float(ones)
