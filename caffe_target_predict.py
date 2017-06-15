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

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

data_root = '../../moon_images/'
files = glob.glob('%s*.png'%data_root)

im_lst = []
for f in files:
    im = Image.open(f)
    in_ = cv2.imread(f)
    im_lst.append(in_)

idx = 0

in_ = im_lst[idx]
in_ = in_.transpose((2,0,1))

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
model_root = './'
net = caffe.Net(model_root+'deploy.prototxt', model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

out = np.asarray([out1,out2,out3,out4,out5,fuse])
np.save('%simages.npy'%data_root,out)
