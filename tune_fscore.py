#this function explores how different values of beta affect the optimum balance of precision and recall, numerically.

import itertools
import matplotlib.pyplot as plt
import numpy as np
import glob

beta = 1 #controls precision's weakness in f_beta score
real_data = 1

f_beta, precision, recall = [], [], []
if real_data == 1:
    dir = 'tune_jobs'
    files = glob.glob('%s/*.txt'%dir)
    for f in files:
        try:
            lines = open(f,'r').readlines()
            p = float(lines[-3].split('=')[1].split(',')[0])
            r = float(lines[-4].split('=')[1].split(',')[0])
            f = round((1+beta**2)*(r*p)/(p*beta**2 + r),3)
            precision.append(p)
            recall.append(r)
            f_beta.append(f)
        except:
            print "%s not done"%f
else:
    p = np.linspace(0.1,1,20)
    params = list(itertools.product(*[p,p]))
    for p,r in params:
        f=round((1+beta**2)*(r*p)/(p*beta**2 + r),3)
        f_beta.append(f)
        recall.append(r)
        precision.append(p)

pp = plt.scatter(precision,recall,c=f_beta,edgecolors='none')
plt.xlabel('precision')
plt.ylabel('recall')
plt.title('beta=%.2f'%beta)
cbar = plt.colorbar(pp)

#for i in range(len(f_beta)):
#    plt.text(precision[i], recall[i], f_beta[i], fontsize=8)

#plt.savefig('tune_jobs/tune_fscore_beta=%.2f.png'%beta)
plt.show()
