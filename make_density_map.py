#!/usr/bin/env python
"""Density and Mask Making

Scripts for determining centroid density maps and segmentation masks as targetes for CNN.  See docstrings of functions for further help.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from scipy.spatial import cKDTree as kd


def gkern(l=5, sig=1.):
    """
    Creates Gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


# https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays
def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """

    # mask grid extent (+1 to ensure we capture radius)
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)

# http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#circle
# Though that autodoc is terrible, and should be supplemented with
# http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
# and (file that defines static void Circle(...))
# https://github.com/opencv/opencv/blob/05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/src/drawing.cpp
def ringmaker(r=10., dr=1):
    """
    Creates ring of radius r and thickness dr.

    Parameters
    ----------
    r : float
        Ring radius
    dr : int
        Ring thickness (cv2.circle requires int)
    """

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr/2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2*rhext + 1, 2*rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext,rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)


def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging 
    gaussian with base image, including edge case
    handling.  x and y are identical, so code
    is axis-neutral.

    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h; right = cen + ks_h + 1

    # Handle edge cases.
    # If left side of gaussian is beyond the left
    # end of the image.
    if left < 0:
        # Crop gaussian and shift image index
        # to lefthand side.
        img_l = 0; g_l = -left
    else:
        img_l = left; g_l = 0
    if right > imglen:
        img_r = imglen; g_r = ker_shp - (right - imglen)
    else:
        img_r = right; g_r = ker_shp

    return [img_l, img_r, g_l, g_r]


def make_density_map(craters, img, kernel=None, k_support = 8, k_sig=4., knn=10, 
                        beta=0.3, kdict={}, truncate=True):
    """Makes Gaussian kernel density maps.

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : numpy.ndarray
        original image; assumes colour channel is last axis (tf standard)
    kernel : function, "knn" or None
        If a function is inputted, function must return an array of 
        length craters.shape[0].  If "knn",  uses variable kernel with 
            sigma = beta*<d_knn>,
        where <d_knn> is the mean Euclidean distance of the k = knn nearest 
        neighbouring craters.  If anything else is inputted, will use
        constant kernel size with sigma = k_sigma.
    k_support : int
        Kernel support (i.e. size of kernel stencil) coefficient.  Support
        is determined by kernel_support = k_support*sigma.  Defaults to 8.
    k_sig : float
        Sigma for constant sigma kernel.  Defaults to 1.
    knn : int
        k nearest neighbours, used for "knn" kernel.  Defaults to 10.
    beta : float
        Beta value used to calculate sigma for "knn" kernel.  Default 
        is 0.3.
    kdict : dict
        If kernel is custom function, dictionary of arguments passed to kernel.
    truncate : bool
        If True, truncate mask where image truncates
    """

    # Load blank density map
    imgshape = img.shape[:2]
    dmap = np.zeros(imgshape)

    # Get number of craters
    N_ctrs = craters.shape[0]

    # Obtain gaussian kernel sigma values
    # callable checks if kernel is function
    if callable(kernel):
        sigma = kernel(**kdict)
    # If knn is used
    elif kernel == "knn":
        # If we have more than 1 crater, select either nearest 11 or N_ctrs
        # neighbours, whichever is closer
        if N_ctrs > 1:
            kdt = kd(craters[["x","y"]].as_matrix(), leafsize=10)
            dnn = kdt.query(craters[["x","y"]].as_matrix(), \
                                    k=min(N_ctrs, knn + 1))[0][:, 1:].mean(axis=1)
        # Otherwise, assume there are craters "offscreen" half an image away
        else:
            dnn = 0.5*imgshape[0]*np.ones(1)
        sigma = beta*dnn
    else:
        sigma = k_sig*np.ones(N_ctrs)

    # Gaussian adding loop
    for i in range(N_ctrs):
        cx = int(craters["x"][i]); cy = int(craters["y"][i])

        # A bit convoluted, but ensures that kernel_support
        # is always odd so that centre of gaussian falls on
        # a pixel.
        ks_half = int( k_support*sigma[i] / 2)
        kernel_support = ks_half * 2 + 1
        kernel = gkern(kernel_support, sigma[i])

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx, imgshape[1], 
                                                    ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy, imgshape[0], 
                                                    ks_half, kernel_support)

        # Add kernel to image
        dmap[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    # Removes
    if truncate:
        if img.ndim == 3:
            dmap[img[:,:,0] == 0] = 0
        else:
            dmap[img == 0] = 0

    return dmap


def make_mask(craters, img, binary=True, rings=False, 
                                ringwidth=1, truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : numpy.ndarray
        original image; assumes colour channel is last axis (tf standard)
    binary : bool
        If True, returns a binary image of crater masks
    rings : bool
        If True, mask uses hollow rings rather than filled circles
    ringwiddth : int
        If rings is True, ringwidth sets the width (dr) of the ring.
        
    truncate : bool
        If True, truncate mask where image truncates
    """

    # Load blank density map
    imgshape = img.shape[:2]
    dmap = np.zeros(imgshape)
    cx, cy = craters["x"].values.astype('int'), craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.

    for i in range(craters.shape[0]):
        if rings:
            kernel = ringmaker(r=radius[i], dr=ringwidth)
        else:
            kernel = circlemaker(r=radius[i])
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                    ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                    ks_half, kernel_support)

        # Add kernel to image
        dmap[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]
    
    if binary:
        dmap = (dmap > 0).astype(float)
    
    if truncate:
        if img.ndim == 3:
            dmap[img[:,:,0] == 0] = 0
        else:
            dmap[img == 0] = 0
    
    #add centroids to image
    #dmap[cy,cx] = 2

    return dmap
