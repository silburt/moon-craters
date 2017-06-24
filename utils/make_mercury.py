#!/usr/bin/env python
"""Moon Cratering Project Input Dataset Generator

Script that calls make_input_data and make_density_map to create a dataset of input moon images and target density maps, in either png image or numpy tensor format.  Because the number of tunable parameters that the user must consider, I've chosen not to use parser.  Instead, I recommend making a copy of this script.

The script uses mpi4py to speed up the processing.  Comment out the MPI code block below to remove this functionality on systems where it isn't installed.
"""

########################### Imports ###########################

# Past-proofing
from __future__ import absolute_import, division, print_function

# System modules
import os
import sys
import glob

# I/O and math stuff
import pandas as pd
import numpy as np
from PIL import Image, ImageChops, ImageOps

# Input making modules
import make_input_data as mkin
import make_density_map as densmap


########################### Global Variables ###########################


source_image_path = "/home/cczhu/cratering_big_data/Messenger/mercury_zerocentral.png"     # Source image path
#lu_csv_path = "./LU78287GT.csv"                     # Salamuniccar crater dataset csv path
#alan_csv_path = "./alanalldata.csv"                 # LROC crater dataset (from Alan) csv path
merc_csv_path = "./MercLargeCraters.csv"            # Fassett et al. >20km crater data
outhead = "/home/cczhu/cratering/mercury/merc"   # Output filepath and file header (if 
                                                    # outhead = "./out/lola", files will have extension
                                                    # "./out/lola_XXXX.png", "./out/lola_XXXX_mask.png", etc.)
zeropad = 5                                         # Number of zeros to pad numbers in output files (number of
                                                    # Xs in "...XXXX.png" above)

amt = 100                                         # Number of images each thread will make (multiply by number of
                                                    # threads for total number of images produced)

ilen_range = [1000., 3000.]                         # Range of image widths, in pixels, to crop from source image.  For
                                                    # Mercury (R = 2440 km) 11k pixel image, 20 km diameter crater = 30 pix
                                                    # so we can downsample by factor of 10 and be okay with it.

olen = 256                                          # Size of Mercury images
dmlen = 256                                         # Size of density maps (should be 2^i smaller than olen, for 
                                                    # some integer i >=0 dependent on CNN architecture)

source_cdim = [-180, 180, -90, 90]                  # [Min long, max long, min lat, max lat] dimensions of source 
                                                    # image (it'll almost certainly be the entire globe) DO NOT ALTER

sub_cdim = [-180, 180, -90, 90]                     # [Min long, max long, min lat, max lat] sub-range of long/lat to
                                                    # use when sampling random images, useful for separating train and 
                                                    # test sets

minpix = 1.                                         # Minimum pixel diameter of craters used in density map.  5 km on the moon

slivercut = 0.8                                     # Minimum width/height aspect ratio of acceptable image.  Keeping this
                                                    # < 0.8 or so prevents "wedge" images derived from polar regions from
                                                    # being created.  DO NOT SET VALUE TOO CLOSE TO UNITY!

outp = "outp"                                       # If str, script dumps pickle containing the long/lat boundary and crop 
                                                    # bounds of all images.  Filename will be of the form outhead + outp + ".p".
                                                    # If multithreading is enabled, rank will be appended to filename.


# Density map and mask arguments

maketype = "mask"                                   # Type of target to make - "dens" for density map, "mask" for mask

savetiff = True                                     # If true, save density maps as tiff files (8-bit pngs don't work for intensity maps)
                                                    # with arbitrary scaling.

savenpy = True                                      # If true, dumps input images to outhead + "input.npy" , and target density maps or masks to
                                                    # outhead + "targets.npy"

dmap_args = {}                                      # dmap kernel args

dmap_args["truncate"] = True                        # If True, truncate mask where image truncates (i.e. has padding rather than image content)


# Density map args
dmap_args["kernel"] = None                          # Specifies type of kernel to use.  Can be a function, "knn" (k nearest neighbours), 
                                                    # or None.  If a function is inputted, function must return an array of 
                                                    # length craters.shape[0].  If "knn",  uses variable kernel with 
                                                    #    sigma = beta*<d_knn>,
                                                    # where <d_knn> is the mean Euclidean distance of the k = knn nearest 
                                                    # neighbouring craters.  If anything else is inputted, will use
                                                    # constant kernel size with sigma = k_sigma.

dmap_args["k_support"] = 8                          # Kernel support (i.e. size of kernel stencil) coefficient.  Support
                                                    # is determined by kernel_support = k_support*sigma.

dmap_args["k_sig"] = 3.                             # Sigma for constant sigma kernel (kernel = None).

dmap_args["knn"] = 10                               # k nearest neighbours, used when kernel = "knn".

dmap_args["beta"] = 0.2                             # Beta value used to calculate sigma when kernel = "knn" (see above).

dmap_args["kdict"] = {}                             # If kernel is custom function, dictionary of arguments passed to kernel.


# Mask arguments

dmap_args["rings"] = True                          # If True, use rings as masks rather than circles

dmap_args["ringwidth"] = 2                          # If dmap_args["rings"] = True, thickness of ring

dmap_args["binary"] = True                          # If True, returns a binary image of crater masks 


# Determine outp, and set rank = 0 in case MPI is not used below
outp = outp + ".p"
rank = 0


########################### MPI ###########################


# Utilize mpi4py for multithreaded processing
# Comment this block out if mpi4py is not available
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Thread {0} of {1}".format(rank, size))
if outp: # Append rank to outp filename
    outp = outp.split(".p")[0] + "_p{0}.p".format(rank)


########################### Script ###########################


def load_img_make_target(filename, maketype, outshp, minpix, dmap_args):
    """Loads individual image.
    """
    # Load base image
    img = Image.open(filename).convert('L')
    # Dummy image of target size.  Bilinear interpolation is compromise
    # between Image.NEAREST, which creates artifacts, and Image.LANZCOS,
    # which is more expensive (though try that one if BILINEAR gives
    # crap)
    omg = np.asanyarray(img.resize(outshp, resample=Image.BILINEAR))
    img = np.asanyarray(img)

    # Load craters CSV
    craters = pd.read_csv(filename.split(".png")[0] + ".csv")
    # Resize crater positions and diameters to target size
    craters.loc[:, ["x", "y", "Diameter (pix)"]] *= outshp[0]/img.shape[0]
    # Cut craters that are now too small to be detected clearly
    craters = craters[craters["Diameter (pix)"] >= minpix]
    craters.reset_index(inplace=True, drop=True)

    if maketype == "mask":
        dmap = densmap.make_mask(craters, omg, binary=dmap_args["binary"],
                                        rings=dmap_args["rings"],
                                        ringwidth=dmap_args["ringwidth"],
                                        truncate=dmap_args["truncate"])
    else:
        dmap = densmap.make_density_map(craters, omg, kernel=dmap_args["kernel"], 
                        k_support=dmap_args["k_support"], 
                        k_sig=dmap_args["k_sig"], knn=dmap_args["knn"], 
                        beta=dmap_args["beta"], kdict=dmap_args["kdict"], 
                        truncate=dmap_args["truncate"])

    return img, dmap


def make_dmaps(files, maketype, outshp, minpix, dmap_args, savetiff=False):
    """Chain-loads input data pngs and make target density maps/masks

    Parameters
    ----------
    files : list
        List of files to process
    maketype : str
        "dens" or "mask", depending on if you want to make
        a density map or a mask
    outshp : listlike
        [height, width] of target image
    minpix : float
        Minimum crater diameter in pixels to be included in target
    dmap_args : dict
        Dictionary of arguments to pass to target generation 
        functions.
    savetiff : bool
        If True, saves target to output file with name = 
        filename.split(".png") + maketype + ".tiff".  Using
        tiff as file format because target is density map with
        arbitrary intensities, while most image formats go from 
        0 - 256 between 3 channels.        
    """
    X = []
    X_id = []
    Y = []
    Y_id = []

    #files = sorted([fn for fn in glob.glob('%s*.png'%path)
    #         if (not os.path.basename(fn).endswith('mask.png') and
    #        not os.path.basename(fn).endswith('dens.png'))])
    print("number of input image files: %d"%(len(files)))
    print("Generating target tmages ({0:s}).".format(maketype))

    for fl in files:
        cX, cY = load_img_make_target(fl, maketype, outshp, minpix, dmap_args)
        X.append(cX)
        X_id.append(fl)
        Y.append(cY)
        mname = fl.split(".png")[0] + maketype + ".tiff"
        Y_id.append(mname)
        if savetiff:
            imgo = Image.fromarray(cY)
            imgo.save(mname);

    return X, Y, X_id, Y_id


# Read source image
img = Image.open(source_image_path).convert("L")
    
if sub_cdim != source_cdim:
    img = mkin.InitialImageCut(img, source_cdim, sub_cdim)

craters = mkin.ReadMercuryCraterCSV(filename=merc_csv_path)
#craters = mkin.ReadCombinedCraterCSV(filealan=alan_csv_path, filelu=lu_csv_path,
#                                        dropfeatures=True)
# Co-opt ResampleCraters to remove all craters beyond subset cdim
# keep minpix = 0 (since we don't have pixel diameters yet)
craters = mkin.ResampleCraters(craters, sub_cdim, None)

# Generate input images
print("Generating input images")
mkin.GenDataset(img, craters, outhead, ilen_range=ilen_range,
                olen=olen, cdim=sub_cdim, amt=amt, zeropad=zeropad, minpix=minpix,
                slivercut=slivercut, outp=outp, istart = rank*amt)

files = [outhead + "_{i:0{zp}d}.png".format(i=i, zp=zeropad) 
                                            for i in range(rank*amt, (rank+1)*amt)]

# Generate target density maps/masks
outshp = (dmlen, dmlen)
X, Y, X_id, Y_id = make_dmaps(files, maketype, outshp, minpix, dmap_args, savetiff=savetiff)

# Optionally, save data as npy file
if savenpy:
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    np.save(outhead + "_{rank:01d}_input.npy".format(rank=rank), X)
    np.save(outhead + "_{rank:01d}_targets.npy".format(rank=rank), Y)
