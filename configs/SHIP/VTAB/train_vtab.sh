#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=VPT-VTAB
GPUS=1
CKPT=$1
GPU_ID=$2

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

K=50

CONFIG=experiments/SHIP/ViT-B_prompt_${K}
LR=0.001
WEIGHT_DECAY=0.001
SHARE_THRE=0.93
SEED=1

for DATASET in cifar100 #oxford_pet diabetic_retinopathy dtd oxford_flowers102 sun397 dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti svhn dtd caltech101 
do
    DIR=./saves/${DATASET}/${DATASET}_${LR}_${WEIGHT_DECAY}_sharethre_${SHARE_THRE}_K${K}_cfg_${CONFIG}/seed${SEED}
    LOG_FILE=${DIR}/log.txt
    if [ -f "$LOG_FILE" ]; then
        echo "Results are available in ${DIR}. Resuming..."
    else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} \
        --cfg="${CONFIG}.yaml" --resume=${CKPT} --output_dir=${DIR} --share_threshold=${SHARE_THRE} --batch-size=64 --lr=${LR} --epochs=100 --is_visual_prompt_tuning --weight-decay=${WEIGHT_DECAY} --mode=retrain --drop_rate_prompt=0.1  --seed=${SEED} --launcher="none" #\
        wait
    fi
    wait
done
