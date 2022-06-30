#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=default
#SBATCH --job-name=c100
#SBATCH --time=12:00:00
#SBATCH --mem=50G

port=$RANDOM

# Experiment parameters
ARCH="resnet18"
hidden_mlp=512
ARCH_name="resnet18"
DATASET_PATH="cifar100"

# Augmentation parameters 
min_scale_crops=0.14
max_scale_crops=1.0

# Training parameters 
nepochs=400
bs=128
LR_start=0.6 
LR_end=0.0006 
warmup_epochs=5
size_crops=32
nmb_crops=2
nmb_prototypes=100
queue_length=120
epoch_queue_starts=15
use_fp16=True
freeze_prototypes_niters=5005
run_swav=False # Set 'True' to run swav only
workers=2

# Experiment path
EXP_SUFFIX="_${bs}_${ARCH_name}_${nepochs}ep_${nmb_prototypes}prot_${DATASET_PATH}"
EXPERIMENT_PATH="./experiments/unmix_swav_cifar${EXP_SUFFIX}"
mkdir -p $EXPERIMENT_PATH

# Training 
# python -m torch.distributed.launch --nproc_per_node=1 main_swav_unmix.py \
srun --label python -u main_swav_unmix.py \
--dist_url 'tcp://localhost:10001' \
--data_path $DATASET_PATH \
--arch ${ARCH} \
--epochs $nepochs \
--base_lr $LR_start \
--final_lr $LR_end \
--warmup_epochs $warmup_epochs \
--batch_size $bs \
--size_crops $size_crops \
--nmb_crops $nmb_crops \
--min_scale_crops $min_scale_crops \
--max_scale_crops $max_scale_crops \
--use_fp16 $use_fp16 \
--run_swav $run_swav \
--freeze_prototypes_niters $freeze_prototypes_niters \
--nmb_prototypes $nmb_prototypes \
--queue_length $queue_length \
--hidden_mlp $hidden_mlp \
--epoch_queue_starts $epoch_queue_starts \
--dump_path $EXPERIMENT_PATH \
--temperature 0.1 \
--epsilon 0.05 \
--workers $workers

mkdir ${EXPERIMENT_PATH}_linear_eval

# Evaluation
# python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
srun --label python -u eval_linear.py \
--dist_url 'tcp://localhost:10001' \
--num_labels 100 \
--crop_size 32 \
--workers $workers \
--arch $ARCH \
--data_path $DATASET_PATH \
--pretrained ${EXPERIMENT_PATH}/checkpoint.pth.tar \
--dump_path ${EXPERIMENT_PATH}_linear_eval