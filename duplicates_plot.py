# The purpose of this script is to plot the images where duplicates were reconized when calculating the recall on the dev/test sets. Are they truly duplicates, or is there a mistake in the pipeline?

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

data = np.load('datasets/rings/Test_rings/test_data_n1000.npy')
target = np.load('datasets/rings/Test_rings/test_target_n1000.npy')

lines = open('duplicates/test/test_duplicates.txt','r').readlines()

#soln = [0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1,1,0,1,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0]
#soln_i = 0
#correct = 0

n_dupes = 0
dim=256
prev_i = 0
for i,l in enumerate(lines):
    if 'duplicate(s) (shown above)' in l:
        new_i = i
        image_i = int(l.split('image')[1])  #image number to check
        
        #plot image
        cimg = cv2.cvtColor(target[image_i].astype(np.uint8), cv2.COLOR_GRAY2BGR)
        x_prev,y_prev,r_prev = 0,0,0
        for j in range(prev_i,new_i):
            x,y,r = lines[j].split('[')[1].split(']')[0].split()
            x,y,r = int(float(x)), int(float(y)), int(float(r))
            cv2.circle(cimg,(x,y),r,(255,0,0),thickness=1)
            if j == new_i-1:
                longlat_diff = (x-x_prev)**2 + (y-y_prev)**2
                rad_diff = abs(r-r_prev)
                print "image %d: longlat_diff=%d, rad_diff=%f, rad_diff/r=%f,r=%d"%(image_i,longlat_diff,rad_diff,rad_diff/float(r),r)
                cond = (longlat_diff <= 15)&(rad_diff <= max(1.01,0.2*r))
                if cond:
                    title="duplicate found."
                else:
                    title="resolved duplicate."
#                if cond and soln[soln_i]:
#                    print "correctly predicted duplicate for image %d"%image_i
#                    correct += 1
#                elif cond and soln[soln_i] == 0:
#                    print "incorrectly predicted duplicate when distinct craters for image %d"%image_i
#                elif cond==False and soln[soln_i] == 1:
#                    print "incorrectly predicted distinct craters when duplicate for image %d"%image_i
#                else:
#                    print "correctly predicted genuine craters for image %d"%image_i
#                    correct += 1
#                soln_i +=1
                print ""
            else:
                x_prev,y_prev,r_prev = x,y,r

        n_dupes += 1
        #print "n_dupes = %d"%n_dupes
        prev_i = i+1
        
        plotting = 1
        if plotting == 1:
            f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=[12, 4])
            ax1.imshow(data[image_i].reshape(dim,dim),origin='upper', cmap="Greys_r")
            ax2.imshow(target[image_i],origin='upper', cmap="Greys_r")
            ax2.imshow(cimg, origin='upper',alpha=0.7)
            ax3.imshow(data[image_i].reshape(dim,dim),origin='upper', cmap="Greys_r")
            ax3.imshow(cimg,origin='upper', alpha=0.5)
            plt.title(title)
            plt.savefig('duplicates/duplicates_img%d.png'%image_i)
            plt.close()

print "%f accuracy"%(correct/float(len(soln)))
