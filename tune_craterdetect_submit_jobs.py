#These jobs are submitted to ICS
#Note, by default the loaded environment for a submitted job script is the same as the one loaded in .bash_profile.

import itertools
import numpy as np
import os

#iterate parameters
longlat_thresh2 = np.array([1,10,50,70,200])
template_thresh = np.array([0.3,0.4,0.5,0.6,0.7])
rad_thresh = np.array([0.1,0.2,0.3,0.4,0.5,0.6])
#target_thresh = np.array([0.05,0.1,0.15])

#all combinations of above params
params = list(itertools.product(*[longlat_thresh2, template_thresh, rad_thresh]))
#params = list(itertools.product(*[longlat_thresh2, rad_thresh, template_thresh, target_thresh]))

#submit jobs as you make them. If ==0 just make them
submit_jobs = 1

#make jobs
jobs_dir = "tune_jobs"
counter = 0
for llt2,tt,rt in params:
    pbs_script_name = "tune_mr%d_llt%.2e_tt%.2e.pbs"%(llt2,tt,rt)
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
        f.write('python tune_craterdetect_hypers_HEAD.py %f %f %f > %s.txt\n'%(llt2,tt,rt,pbs_script_name.split('.pbs')[0]))
    f.close()

    if submit_jobs == 1:
        os.system('mv %s/%s %s'%(jobs_dir,pbs_script_name, pbs_script_name))
        os.system('qsub %s'%pbs_script_name)
        os.system('mv %s %s/%s'%(pbs_script_name,jobs_dir,pbs_script_name))
        counter += 1

print("submitted %d jobs"%counter)
