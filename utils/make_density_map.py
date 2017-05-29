import numpy as np
import pandas as pd
from PIL import Image

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
def circlemaker(r=10., rings=False):
    """
    Creates circle mask of radius r.
    """

    # mask grid extent (+1 to ensure we capture radius)
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    if rings == True:
        circle = np.abs((xx**2 + yy**2) - r**2) <= r
    else:
        circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)


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


def make_density_map(craters, imgshape, kernel=None, k_support = 8, k_sig=4., knn=10, 
                        beta=0.3, kdict={}):
    """Makes Gaussian kernel density maps.

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    imgshape : listlike
        Image dimensions [y, x], i.e. output of img.shape
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
    """

    # Load blank density map
    dmap = np.zeros(imgshape)

    # Obtain gaussian kernel sigma values
    # callable checks if kernel is function
    if callable(kernel):
        sigma = kernel(**kdict)
    elif kernel == "knn":
        kdt = kd(craters[["x","y"]].as_matrix(), leafsize=10)
        dnn = kdt.query(craters[["x","y"]].as_matrix(), \
                                k=knn + 1)[0][:, 1:].mean(axis=1)
        sigma = beta*dnn
    else:
        sigma = k_sig*np.ones(craters.shape[0])


    #kdt = kd(craters[["x","y"]].as_matrix(), leafsize=10)
    #dnn = kdt.query(craters[["x","y"]].as_matrix(), k=11)[0][:, 1:].mean(axis=1)
    #ker_sigma = 0.3*dnn
    #ker_sigma = 4*np.ones(dnn.shape)

    for i in range(craters.shape[0]):
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

    return dmap

def make_mask(craters, img, binary=True, truncate=True, rings=False):
    """Makes crater mask binary image.
        
        Parameters
        ----------
        craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
        img : 3D numpy array
        original image.
        binary : bool
        If True, returns a binary image of crater masks
        truncate : bool
        If True, truncate mask where image truncates
        rings : bool
        If True, make rings instead of circles.
        """
    
    # Load blank density map
    imgshape = (img.shape[0], img.shape[1])     #Image dimensions [y, x], i.e. output of img.shape
    cx, cy = craters["x"].values.astype('int'), craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.
    
    dmap = np.zeros(imgshape)
    for i in range(craters.shape[0]):
        kernel = circlemaker(radius[i], rings)
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
        dmap[img[:,:,0] == 0] = 0
    
    return dmap

#A.S. defined function
def make_circlering_mask(craters, img, binary=True, truncate=True):
    """Makes circlering masks, i.e. crater edge -> ring = 1, crater itself -> circle = 0.5

    Parameters
    ----------
    craters : pandas.DataFrame
        craters dataframe that includes pixel x and y columns
    img : 3D numpy array
        original image. 
    binary : bool
        If True, returns a binary image of crater masks
    truncate : bool
        If True, truncate mask where image truncates
    rings : bool
        If True, make rings instead of circles.
    """

    # Load blank density map
    imgshape = (img.shape[0], img.shape[1])     #Image dimensions [y, x], i.e. output of img.shape
    cx, cy = craters["x"].values.astype('int'), craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.
    dmap_rings, dmap_circles = np.zeros(imgshape), np.zeros(imgshape)

    #generate circles
    for i in range(craters.shape[0]):
        kernel = circlemaker(radius[i], 0)
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2
        
        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                   ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                   ks_half, kernel_support)
                                                   
        # Add kernel to image
        dmap_circles[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]
    
    if binary:
        dmap_circles[dmap_circles > 0] = 0.5
    if truncate:
        dmap_circles[img[:,:,0] == 0] = 0

    #generate rings
    for i in range(craters.shape[0]):
        kernel = circlemaker(radius[i], 1)
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2
        
        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                   ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                   ks_half, kernel_support)
        # Add kernel to image
        dmap_rings[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

        if binary:
            dmap_rings[dmap_rings > 0] = 1
        if truncate:
            dmap_rings[img[:,:,0] == 0] = 0

    dmap = dmap_circles + dmap_rings
    dmap[dmap > 1] = 1

    return dmap
