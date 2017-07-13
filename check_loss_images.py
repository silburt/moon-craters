# The purpose of this script is to look at the loss images visually and make sure that they are good images. If not, delete them.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.make_density_map_charles as mdm
import glob
import os
import cv2
from utils.rescale_invcolor import *

dir = 'datasets/rings'
path = '%s/Dev_rings_for_loss'%dir
load_from_source = 1

#load from npy files
if load_from_source == 0:
    imgs = rescale_and_invcolor(np.load("%s/custom_loss_images.npy"%path), 1, 1)
    csvs = np.load("%s/custom_loss_csvs.npy"%path)

    dim = imgs.shape[1]

    for i in range(len(imgs)):
        img = imgs[i]
        csv = pd.DataFrame(csvs[i],columns=['x','y','Diameter (pix)'])
        csv['Diameter (pix)'] = 2*csv['Diameter (pix)']
        target = mdm.make_mask(csv, img, binary=True, rings=True, ringwidth=1, truncate=True)
        plt.imshow(img.reshape(dim,dim), origin='upper', cmap="Greys_r")
        plt.imshow(target, origin='upper', cmap="Greys_r", alpha=0.3)
        plt.savefig('%s/check_loss_images/%d.png'%(path,i))
        plt.clf()

#load from source
else:
    files = glob.glob('%s/*.csv'%path)
    for f in files:
        
        #img
        img = cv2.imread('%s.png'%f.split('.csv')[0], cv2.IMREAD_GRAYSCALE)/255.
        #img[img > 0.] = 1. - img[img > 0.]
        minn, maxx = np.min(img[img>0]), np.max(img[img>0])
        low, hi = 0.1, 1                                                #low, hi rescaling values
        img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
        img[img > 0.] = 1. - img[img > 0.]
        
        #target
        csv = pd.read_csv(f,usecols=['x','y','Diameter (pix)'])
        csv['Diameter (pix)'] = 2*csv['Diameter (pix)']
        target = mdm.make_mask(csv, img, binary=True, rings=True, ringwidth=1, truncate=True)

        #plot
        dim = img.shape[0]
        plt.imshow(img.reshape(dim,dim), origin='upper', cmap="Greys_r")
        plt.imshow(target, origin='upper', cmap="Greys_r", alpha=0.3)
        name = os.path.basename(f).split('.csv')[0]
        plt.savefig('%s/check_loss_images/op_%s_norm.png'%(path,name))   #op = overplot
        plt.clf()
