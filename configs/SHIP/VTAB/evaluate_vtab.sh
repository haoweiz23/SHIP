#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=VPT-VTAB
CONFIG=$1
GPUS=1
GPU_ID=$2
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

LR=0.001
SHARE_THRE=0.95
SEED=1

for DATASET in cifar100 # oxford_pet oxford_flowers102 caltech101 dtd svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
do
    ZS_CKPT=pretrained/ViT-B_16.npz
    VPT_CKPT=path/to/checkpoint.pth
    SHIP_CKPT=path/to/checkpoint.pth
    CKPT=${SHIP_CKPT}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --eval --data-path=./data/vtab-1k/${DATASET} --share_threshold=${SHARE_THRE} --data-set=${DATASET} --cfg="${CONFIG}.yaml" --resume=${CKPT} --output_dir=./saves/${DATASET} --batch-size=64 --lr=${LR} --epochs=100 --is_visual_prompt_tuning --weight-decay=${WEIGHT_DECAY} --mixup=0 --cutmix=0 --mode=retrain --drop_rate_prompt=0.1 --no_aug --inception --direct_resize --seed=${SEED} --launcher="none"\
    wait
done


