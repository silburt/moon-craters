#These jobs are submitted to ICS

import itertools
import numpy as np
import os

#iterate parameters
match_thresh2 = np.linspace(10,70,num=4)
template_thresh = np.linspace(0.1,0.7,num=6)
target_thresh = np.array([0.001,0.01,0.05,0.1,0.15])

#all combinations of above params
params = list(itertools.product(*[match_thresh2, template_thresh, target_thresh]))

#submit jobs as you make them. If ==0 just make them
submit_jobs = 0

#make jobs
jobs_dir = "tune_jobs"
counter = 0
for ma2,te,ta in params:
    pbs_script_name = "tune_ma%.2e_te%.2e_ta%.2e.pbs"%(ma2,te,ta)
    with open('%s/%s'%(jobs_dir,pbs_script_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#PBS -l nodes=1:ppn=1\n')
        f.write('#PBS -l walltime=24:00:00\n')
        f.write('#PBS -l pmem=2gb\n')
        f.write('#PBS -A ebf11_a_g_sc_default\n')
        f.write('#PBS -j oe\n')
        f.write('cd $PBS_O_WORKDIR\n')
        f.write('python tune_craterdetect_hypers.py %f %f %f > tune_ma%.2e_te%.2e_ta%.2e.txt\n'%(ma2,te,ta,ma2,te,ta))
    f.close()

    if submit_jobs == 1:
        os.system('mv %s/%s %s'%(jobs_dir,pbs_script_name, pbs_script_name))
        os.system('qsub %s'%pbs_script_name)
        os.system('mv %s %s/%s'%(pbs_script_name,jobs_dir,pbs_script_name))
        counter += 1

print "submitted %d jobs"%counter
