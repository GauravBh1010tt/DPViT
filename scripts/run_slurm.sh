#!/bin/bash
#SBATCH --nodes=3
#SBATCH --gpus-per-node=a100:2
#SBATCH --tasks-per-node=2 
#SBATCH --cpus-per-task=8       # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=64G               # Request the full memory of the node
#SBATCH --time=11:50:00
#SBATCH --account=rrg-lsigal
#SBATCH --output=PKD_PM_FT-%j.out

source ../env_dpl/bin/activate
nvidia-smi

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "$SLURM_NODEID master: $MASTER_ADDR world_size: $SLURM_NTASKS"
echo "$SLURM_NODEID Launching python script"

exp_name="PKD_nw_PM_FT"

EXP_pre=exp/${exp_name}
EXP_fine=exp/${exp_name}_FT

#python -m wandb login 749dd8035d38a54262549bdfdc68bc3822b7522c

pretrain=0

if [[ $pretrain -gt 0 ]]
then
echo "Phase 1 pre-training ..." ${exp_name}
srun python -u ../main_dpvit.py --data_path ../data/mini_imagenet/train_comb \
    --output_dir ${EXP_pre} --evaluate_freq 50 --visualization_freq 50 --init_method=tcp://$MASTER_ADDR:3466 \
    --prod_mode=False --use_fp16 True --lr 0.0005 --epochs 1800 --image_path ../SMKD/img_viz \
    --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --num_workers=4 --n_gpus=$SLURM_NTASKS \
    --lr_mix 0 --lr_noise 1 --K 64 --num_fore 40 --use_parts 0 \
    --lambda1 1 --lambda2 0 --lambda3 1 --batch_size_per_gpu 100 --use_DDP=1

else
echo "Phase 2 tuning ..." ${exp_name}
srun python -u ../main_dpvit.py --data_path ../data/mini_imagenet/train_comb \
    --pretrained_path ${EXP_pre} --pretrained_file checkpoint.pth --init_method=tcp://$MASTER_ADDR:3456 \
    --output_dir ${EXP_fine} --evaluate_freq 5 --visualization_freq 5 --use_fp16 True --image_path ../SMKD/img_viz \
    --lr 0.0005 --epochs 150 --lambda1 1 --lambda2 0.45 --num_workers=4 --n_gpus=$SLURM_NTASKS \
    --lambda3 0 --supervised_contrastive --batch_size_per_gpu 90 --global_crops_scale 0.4 1 \
    --lr_mix 1 --lr_noise 1 --K 64 --num_fore 40 \
    --local_crops_scale 0.05 0.4 --partition test --saveckp_freq 5 --use_DDP=1

fi