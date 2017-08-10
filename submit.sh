How to run on scinet:

ssh silburt@login.scinet.utoronto.ca
ssh p8t03
cd $SCRATCH/moon-craters/

#loading Keras
module load gcc/6.2.1 
module load cuda/8.0
source /home/k/kristen/kristen/keras_venv_P8.v2/bin/activate 

#If loading Caffe - do not load gcc - that will lead to an error during compilation.
module load cuda/8.0
module load caffe/nv-0.14.5

CUDA_VISIBLE_DEVICES=1 nohup python moon_unet_s256_rings_pred_weighted.py > unet_s256_rings_pred_jaccard.txt &
CUDA_VISIBLE_DEVICES=1 nohup python moon_unet_s256_rings_pred.py > unet_s256_rings_predfull.txt &
CUDA_VISIBLE_DEVICES=1 nohup python moon_unet_s256_rings_public.py > unet_s256_rings_public.txt &
CUDA_VISIBLE_DEVICES=2 nohup python moon_unet_s256_rings_customloss.py > unet_s256_rings_cl.txt &
CUDA_VISIBLE_DEVICES=0 nohup python moon_unet_s256_rings.py > unet_s256_rings_invcolor_datagen.txt &
CUDA_VISIBLE_DEVICES=1 nohup python moon_unet_s256_rings_forKristen.py > unet_s256_rings_forKristen.txt &

CUDA_VISIBLE_DEVICES=1 nohup python crater_distribution_extract_orig.py > debug.txt &

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

