ssh silburt@login.scinet.utoronto.ca
ssh p8t03
cd $SCRATCH/moon-craters/

module load gcc/6.2.1 
module load cuda/8.0 
source /home/k/kristen/kristen/keras_venv_P8.v2/bin/activate 

CUDA_VISIBLE_DEVICES=0 python moon_vgg16.py --run_fold 1 > outputfold1.txt &
CUDA_VISIBLE_DEVICES=1 python moon_vgg16.py --run_fold 2 > outputfold2.txt &
CUDA_VISIBLE_DEVICES=2 python moon_vgg16.py --run_fold 3 > outputfold3.txt &
CUDA_VISIBLE_DEVICES=3 python moon_vgg16.py --run_fold 4 > outputfold4.txt &


