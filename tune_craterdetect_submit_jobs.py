#These jobs are submitted to ICS
#Note, by default the loaded environment for a submitted job script is the same as the one loaded in .bash_profile.

import itertools
import numpy as np
import os

#iterate parameters
longlat_thresh2 = np.linspace(10,70,num=4)
rad_thresh = np.linspace(0.1,1,num=4)
template_thresh = np.linspace(0.2,0.8,num=4)
target_thresh = np.array([0.05,0.1,0.15])

#all combinations of above params
params = list(itertools.product(*[longlat_thresh2, rad_thresh, template_thresh, target_thresh]))

#submit jobs as you make them. If ==0 just make them
submit_jobs = 1

#make jobs
jobs_dir = "tune_jobs"
counter = 0
for llt2,rt,te,ta in params:
    pbs_script_name = "tune_llt%.2e_rt%.2e_te%.2e_ta%.2e.pbs"%(llt2,rt,te,ta)
    with open('%s/%s'%(jobs_dir,pbs_script_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#PBS -l nodes=1:ppn=1\n')
        f.write('#PBS -l walltime=24:00:00\n')
        f.write('#PBS -l pmem=2gb\n')
        f.write('#PBS -A ebf11_a_g_sc_default\n')
        f.write('#PBS -j oe\n')
        f.write('module load gcc/5.3.1 python/2.7.8\n')
        f.write('source /storage/home/ajs725/venv/bin/activate\n')
        f.write('cd $PBS_O_WORKDIR\n')
        f.write('python tune_craterdetect_hypers_HEAD.py %f %f %f %f > tune_llt%.2e_rt%.2e_te%.2e_ta%.2e.txt\n'%(llt2,rt,te,ta,llt2,rt,te,ta))
    f.close()

    if submit_jobs == 1:
        os.system('mv %s/%s %s'%(jobs_dir,pbs_script_name, pbs_script_name))
        os.system('qsub %s'%pbs_script_name)
        os.system('mv %s %s/%s'%(pbs_script_name,jobs_dir,pbs_script_name))
        counter += 1
    break
print "submitted %d jobs"%counter
