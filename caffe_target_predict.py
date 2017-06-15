#***********************************
#Instructions: This script is run on P8t03/04, and generates new targets based off the edge detection network by https://github.com/s9xie/hed
# You must first load on p8t03/04:
# module load cuda/8.0
# module load caffe/nv-0.14.5

# This script should be located in hed/examples/hed/, along with hed_pretrained_bsds.caffemodel.
# You might also (from the hed/ directory) have to 'make all', 'make pycaffe' according to these instructions: http://caffe.berkeleyvision.org/installation.html#compilation

#***********************************
import numpy as np
import scipy.misc
import Image
import cv2
import scipy.io
import glob
import os
import sys

#rescale and invert color
def rescale_and_invcolor(f, invc, rs):
    img = cv2.imread(f)
    if invc == 1:
        img[img > 0.] = 1. - img[img > 0.]
    if rs == 1:
        minn, maxx = np.min(img[img>0]), np.max(img[img>0])
        low, hi = 0.1*255, 1*255                                        #low, hi rescaling values
        img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    return img.astype(np.float32)

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
sys.path.insert(0, caffe_root + 'python')

import caffe
#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
model_root = './'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
print "**loaded net successfully**"

#Arguments
inv_color = 0
rescale = 1

#main function
data_root = '../../moon_images/'
files = glob.glob('%s*.png'%data_root)
print "analyzing %d files"%len(files)
for f in files:
    name = os.path.basename(f).split('.png')[0]
    img = rescale_and_invcolor(f, inv_color, rescale)
    img = img.transpose((2,0,1))
    net.blobs['data'].reshape(1, *img.shape)
    net.blobs['data'].data[...] = img
    # run net and take argmax for prediction
    net.forward()
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:] #this seems to be the best prediction
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    np.save('%s%s_invcol%s.npy'%(data_root,name,inv_color),np.asarray(out1))

print "Successfully finished analyzing files"
