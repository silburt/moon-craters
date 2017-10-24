# This script is to evaluate the number of times a single crater shows up across multiple images.

import numpy as np
import glob

dir = 'datasets/rings/Train_rings/'
N_draws = 20        #number of random draws (and see how often each shows up)
tol = 1e-3          #tolerance for duplicate, should be small

csvs = glob.glob('%s*.csv'%dir)
rN = np.random.randint(0,len(csvs),size=N_draws)

N_duplicates = []
random_numbers = []
for i in range(N_draws):
    csv = np.genfromtxt(csvs[rN[i]],delimiter=',',skip_header=1,usecols=[0,1,2])
    rS = np.random.randint(0,len(csv))
    sample = csv[rS]
    N_dupes = 0
    for c in csvs:
        file = np.genfromtxt(c,delimiter=',',skip_header=1,usecols=[0,1,2])
        if file.size >= 6:
            N_dupes += len(np.where(np.sum((file - sample)**2,axis=1) < tol)[0])
    random_numbers.append((rN[i],rS))
    N_duplicates.append(N_dupes - 1)
    print "completed draw %d"%i

N_duplicates, random_numbers = np.array(N_duplicates), np.array(random_numbers)
index = np.where(N_duplicates > 0)
N_duplicates, random_numbers = N_duplicates[index], random_numbers[index]
print random_numbers, N_duplicates
print "The mean number of duplicates is (total draws=%d, n_hits=%d): %f +/- %f"%(N_draws,len(N_duplicates),np.mean(N_duplicates),np.std(N_duplicates))
