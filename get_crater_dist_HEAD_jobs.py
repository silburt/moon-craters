# This submits a bunch of jobs to search for hypers
from subprocess import call

#thresholds = [(0.5,0.5),(0.6,0.6),(0.7,0.7),(0.8,0.8),(0.55,0.65),(0.65,0.55)]
thresholds = [(0.5,0.5)]

for llt2,rt2 in thresholds:
    call('nohup python get_crater_dist_HEAD.py %f %f &>> HEAD_craterr_dist_%.2f_%.2f.txt &'%(llt2,rt2,llt2,rt2))
