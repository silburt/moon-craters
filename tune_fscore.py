#this function explores how different values of beta affect the optimum balance of precision and recall, numerically.

import itertools
import matplotlib.pyplot as plt
import numpy as np
import glob

beta = 1 #controls precision's weakness in f_beta score
real_data = 1

f_beta, precision, recall, ext = [], [], [], ''
if real_data == 1:
    dir = 'tune_jobs'
    files = glob.glob('%s/*.txt'%dir)
    for file in files:
        try:
            lines = open(file,'r').readlines()
            elo = float(lines[-4].split(':')[1].split(',')[1])
            ela = float(lines[-3].split(':')[1].split(',')[1])
            er = float(lines[-2].split(':')[1].split(',')[1])
            p = float(lines[-6].split('=')[1].split(',')[0])
            r = float(lines[-7].split('=')[1].split(',')[0])
            f = round((1+beta**2)*(r*p)/(p*beta**2 + r),3)
            precision.append(p)
            recall.append(r)
            f_beta.append(f)
        
            temp = lines[-8].replace('\n','').replace('=', ', ').split(',')
            if f > 0.5 and float(temp[5]) > 0.4 and r > 0.8:
                #print file
                print "fscore=%f, p=%f, r=%f, elo=%f, ela=%f, er=%f, llt2=%d, temp_thresh=%f, rt2=%f"%(f,p,r,elo,ela,er,float(temp[1]),float(temp[3]),float(temp[5]))
        except:
            pass
            #print "%s not done"%f
else:
    p = np.linspace(0.1,1,20)
    params = list(itertools.product(*[p,p]))
    for p,r in params:
        f=round((1+beta**2)*(r*p)/(p*beta**2 + r),3)
        f_beta.append(f)
        recall.append(r)
        precision.append(p)
    ext = '_num'

pp = plt.scatter(precision,recall,c=f_beta,edgecolors='none')
plt.xlabel('precision')
plt.ylabel('recall')
plt.title('beta=%.2f'%beta)
cbar = plt.colorbar(pp)

#for i in range(len(f_beta)):
#    plt.text(precision[i], recall[i], f_beta[i], fontsize=8)

plt.savefig('tune_jobs/tune_fscore_beta=%.2f%s.png'%(beta,ext))
#plt.show()
