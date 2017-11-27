#This looks through all the data and finds the number of craters in each image, as well as the max craters in a given image.
#When running this, port (>) to a txt file, as it prints once per file.

import numpy as np
import pandas as pd
import h5py

def get_id(i, zeropad=5):
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)

#dir = '/scratch/m/mhvk/czhu/newscripttest_for_ari'
dir = '/scratch/m/mhvk/czhu/newsala_for_ari/'
craters = h5py.File('%s/sala_train_craters.hdf5'%dir, 'r')

n_craters = []
for i in range(30000):
    n_craters.append(len( craters[get_id(i)]['block0_values'] ))

np.save('datasets/HEAD/sala_max_craters.npy',n_craters)

