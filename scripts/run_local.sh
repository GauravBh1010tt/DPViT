
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR world_size: $SLURM_NTASKS"
echo "r$SLURM_NODEID Launching python script"

dataset=Imagenet
dataset_path="../data/Imnet"
# dataset_path="/ubc/cs/research/shield/datasets/ILSVRC2012"

exp_name="PKD"_${dataset}

EXP_pre=exp/${exp_name}
EXP_fine=exp/${exp_name}_FT

pretrain=0

GPU=0,1,2,3,4,5,6,7

if [[ $pretrain -gt 0 ]]
then
echo "Phase 1 pre-training ..." ${exp_name}
CUDA_VISIBLE_DEVICES=${GPU} python ../main_dpvit.py --data_path ${dataset_path}/train --dataset_name ${dataset} --image_path ../SMKD/img_viz \
    --output_dir ${EXP_pre} --evaluate_freq 2000 --visualization_freq 200 --init_method=tcp://localhost:3456 \
    --prod_mode=False --use_fp16 True --lr 0.0005 --epochs 2000 --warmup_teacher_temp_epochs 10 \
    --global_crops_scale 0.4 1 --local_crops_scale 0.05 0.4 --saveckp_freq 30 --num_workers=4 --local=1 --seed 0 \
    --lr_mix 1 --lr_noise 1 --K 64 --num_fore 40 --use_parts 1 --lambda_nc 0 --lambda_sdc 0 --supervised_loss \
    --lambda1 1 --lambda2 0 --lambda3 1 --batch_size_per_gpu 22 --n_gpus=8 --use_DDP=1 --celoss_coefficient 1 \
    --masked_fore 0.0

else

echo "Phase 2 tuning ..." ${exp_name}
CUDA_VISIBLE_DEVICES=${GPU} python ../main_dpvit.py --data_path ${dataset_path}/val \
    --output_dir ${EXP_fine} --evaluate_freq 100 --visualization_freq 5 --use_fp16 True --image_path ../SMKD/img_viz \
    --lr 0.0005 --epochs 55 --lambda1 1 --lambda2 0.45 --lambda3 0 --num_workers=4 --init_method=tcp://localhost:3456 \
    --supervised_contrastive --batch_size_per_gpu 18 --global_crops_scale 0.4 1 --local=1 --seed 0 \
    --local_crops_scale 0.05 0.4 --partition test --saveckp_freq 5 --n_gpus=8 --use_DDP=1 \
    --lr_mix 0 --lr_noise 1 --K 64 --num_fore 40 --use_parts 1 --lambda_nc 0 --lambda_sdc 0 --masked_fore 0 \
    --supervised_loss --celoss_coefficient 1 --pretrained_path ${EXP_pre} --pretrained_file checkpoint.pth
fi