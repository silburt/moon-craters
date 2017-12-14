import h5py
import numpy as np

def get_id(i, zeropad=5):
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)

dtype = 'dev'
data_dir = '/scratch/m/mhvk/czhu/moondata/final_data/%s_images.hdf5'%dtype
pred_dir = 'datasets/HEAD/HEAD_%spreds_n30000_final.hdf5'%dtype
raw_thresh = 3000

#large crater investigation
#xleft, xright, ylow, yhi = -39,-38,-20,-19
#xleft, xright, ylow, yhi = 18.4, 31.4, -27.8,-18.1
#xleft, xright, ylow, yhi = -10, -1, -29.86, -18.1

#small crater investigation (switch to less than in search!)
xleft, xright, ylow, yhi = 31.4, 38, -37.8,-18.1

data = h5py.File(data_dir, 'r')
for i in range(10000):
    llbd = data['longlat_bounds'][get_id(i)][...]
    rawlen = data['pix_bounds'][get_id(i)][2] - data['pix_bounds'][get_id(i)][0]
    #if xleft > llbd[0] and xright < llbd[1] and ylow > llbd[2] and yhi < llbd[3] and rawlen > raw_thresh:
    if llbd[0] > xleft and llbd[1] < xright and llbd[2] > ylow and llbd[3] < yhi and rawlen > raw_thresh:
        print(i,llbd)

