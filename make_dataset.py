#!/usr/bin/env python
"""Moon Cratering Project Input Dataset Generator

Script that calls make_input_data and make_density_map to create a dataset of input moon images and target density maps, in either png image or numpy tensor format.  Because the number of tunable parameters that the user must consider, I've chosen not to use parser.  Instead, I recommend making a copy of this script.

The script uses mpi4py to speed up the processing.  Comment out the MPI code block below to remove this functionality on systems where it isn't installed.
"""

########################### Imports ###########################


from __future__ import absolute_import, division, print_function

#import numpy as np
#import pandas as pd
from PIL import Image, ImageChops, ImageOps
#import cartopy.crs as ccrs
#import cartopy.img_transform as cimg
#import matplotlib.pyplot as plt
#import matplotlib.axes as mplax
#import image_slicer as imsl
#import glob
#import collections
#import pickle
#import re

import make_input_data as mkid
import make_density_map as densmap


########################### Global Variables ###########################


source_image_path = "./LOLA_Global_20k.png"     # Source image path
lu_csv_path = "./LU78287GT.csv"                 # Salamuniccar crater dataset csv path
alan_csv_path = "./alanalldata.csv"             # LROC crater dataset (from Alan) csv path
outhead = "out/lola"                            # Output filepath and file header (if 
                                                # outhead = "./out/lola", files will have extension
                                                # "./out/lola_XXXX.png", "./out/lola_XXXX_mask.png", etc.)
zeropad = 5                                     # Number of zeros to pad numbers in output files (number of
                                                # Xs in "...XXXX.png" above)

amt = 7500                                      # Number of images each thread will make (multiply by number of
                                                # threads for total number of images produced)

ilen_range = [600., 2000.]                      # Range of image widths, in pixels, to crop from source image.  For
                                                # the LOLA 20k image, 23040 pixels = 180 degrees of latitude, so
                                                # 2000 pixels = 15.6 degrees of latitude, the approximate maximum
                                                # latitude size of the image to prevent severe

olen = 300                                      # Size of
dmlen = 300                                     # Size of density maps (should be 2^i smaller than olen, for 
                                                # some integer i >=0 dependent on CNN architecture)

source_cdim = [-180, 180, -90, 90]              # [Min long, max long, min lat, max lat] dimensions of source 
                                                # image (it'll almost certainly be the entire globe)
sub_cdim = [-180, 180, -90, 90]                 # [Min long, max long, min lat, max lat] sub-range of long/lat to
                                                # use when sampling random images, useful for separating train and 
                                                # test sets

minpix = 0                                      # Minimum pixel diameter of craters used in density map.  5 km on the moon

slivercut = 0.8                                 # Minimum width/height aspect ratio of acceptable image.  Keeping this
                                                # < 0.8 or so prevents "wedge" images derived from polar regions from
                                                # being created.  DO NOT SET VALUE TOO CLOSE TO UNITY!

outp = False                                    # If str, script dumps pickle containing the long/lat boundary and crop 
                                                # bounds of all images.  Filename will be of the form outhead + outp.
                                                # If multithreading is enabled, rank will be appended to filename.

# SCRIPT OPTIONS (bools)
make_nparray = True                             # Makes .npy files in addition to pngs
make_mask = False
make_



########################### MPI ###########################


# Utilize mpi4py for multithreaded processing

# Uncomment this block if mpi4py is not available
#rank = 0

# Comment this block out if mpi4py is not available
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Thread {0} of {1}".format(rank, size))
if outp: # Append rank to outp filename
    outp = outp.split(".p")[0] + "_p{0}.p".format(rank)


########################### Script ###########################


# Read source image
img = Image.open(source_image_path).convert("L")
    
if sub_cdim != source_cdim:
    img = mkin.InitialImageCut(img, source_cdim, sub_cdim)

craters = mkin.ReadCombinedCraterCSV(filealan=alan_csv_path, filelu=lu_csv_path,
                                        dropfeatures=True)
# Co-opt ResampleCraters to remove all craters beyond subset cdim
# keep minpix = 0 (since we don't have pixel diameters yet)
craters = ResampleCraters(craters, sub_cdim, None)

mkin.GenDataset(img, craters, outhead, ilen_range=ilen_range,
                olen=olen, cdim=sub_cdim, amt=amt, zeropad=zeropad, minpix=minpix,
                slivercut=slivercut, outp=outp, istart = rank*amt)


