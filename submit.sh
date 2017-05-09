How to run on scinet:

ssh silburt@login.scinet.utoronto.ca
ssh p8t03
cd $SCRATCH/moon-craters/

module load gcc/6.2.1 
module load cuda/8.0 
source /home/k/kristen/kristen/keras_venv_P8.v2/bin/activate 

CUDA_VISIBLE_DEVICES=2 nohup python moon_vgg16_1.2.2_fcn.py > FCNoutput3.txt &
CUDA_VISIBLE_DEVICES=2 nohup python moon_vgg16_1.2.2_fcn2.py > FCNoutput_overnight.txt &
CUDA_VISIBLE_DEVICES=2 nohup python moon_vgg16_1.2.2.py > vgg16output.txt &

CUDA_VISIBLE_DEVICES=0 python moon_vgg16_1.2.2.py --run_fold 1 > outputfold1.txt &
CUDA_VISIBLE_DEVICES=1 python moon_vgg16_1.2.2.py --run_fold 2 > outputfold2.txt &
CUDA_VISIBLE_DEVICES=2 python moon_vgg16_1.2.2.py --run_fold 3 > outputfold3.txt &
CUDA_VISIBLE_DEVICES=3 python moon_vgg16_1.2.2.py --run_fold 4 > outputfold4.txt &

How to run on your computer:

create a conda env with the following:
Keras 1.2.2
Tensorflow 0.10.0
cv2 2.4.9.1
sklearn 0.18.1

nvidia-smi

