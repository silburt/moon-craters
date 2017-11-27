#This looks through all the data and finds the number of craters in each image, as well as the max craters in a given image.
#When running this, port (>) to a txt file, as it prints once per file.

import cv2
import os
import glob
import numpy as np
import pandas as pd

#####################
#load/read functions#
########################################################################
def load_data(path, data_type, img_width, img_height):
    N = []
    files = glob.glob('%s*.png'%path)
    print "number of %s files are: %d"%(data_type,len(files))
    max_N_craters = 0
    for fl in files:
        flbase = os.path.basename(fl)
        N_craters = get_csv_len(fl)
        print "%d craters in file %s"%(N_craters[0],fl)
        N.append(N_craters)
        max_N_craters = np.max((max_N_craters, N_craters[0]))
    return  N, max_N_craters

def get_csv_len(file_):                        #returns # craters in each image (target)
    file2_ = file_.split('.png')[0] + '.csv'
    df = pd.read_csv(file2_ , header=0)
    return [len(df.index)]

img_width = 256              #image width
img_height = 256             #image height
#dir = '/scratch/k/kristen/malidib/moon/'
dir = 'datasets/rings'
train_path, valid_path, test_path = '%s/Train_rings/'%dir, '%s/Dev_rings/'%dir, '%s/Test_rings/'%dir
N_craters, max_train = load_data(train_path, 'train', img_width, img_height)
N_craters, max_valid = load_data(valid_path, 'valid', img_width, img_height)
N_craters, max_test = load_data(test_path, 'test', img_width, img_height)
print "max train craters = %d, max valid craters = %d, max test craters = %d"%(max_train,max_valid,max_test)
