# The purpose of this script is to look at the loss images visually and make sure that they are good images. If not, delete them.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.make_density_map_charles as mdm
from utils.rescale_invcolor import *

dir = 'datasets/rings'
path = '%s/Dev_rings_for_loss'%dir
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
