import os
import sys
import time
import math
import json
import copy
import torch
import socket
import glob
import argparse
import numpy as np
import datetime
from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch.backends.cudnn as cudnn
from distutils.util import strtobool
from tensorboardX import SummaryWriter

import models 
import pandas as pd
import utilities.utils as utils
import pdb
from torch import linalg as LA
import torch.distributed as dist
import torch.multiprocessing as mp
import utilities.loss_functions as loss_functions
from data_utils.loader import Custom_dataset, ImageFolderMask
from models.head import iBOTHead
from eval.imagenet_cls import eval_pred
import models.vision_transformer_attn as vits
from torchvision import datasets, transforms
from utilities.utils import local_setup, dist_setup

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('DPViT', add_help=False)

    # Wandb Visualization
    parser.add_argument('--prod_mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb_project_name', type=str, default="ViT",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default='Gaurav',
                        help="the entity (team) of wandb's project")

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--use_DDP', default=1, type=int)
    parser.add_argument('--local', default=0, type=int)

    parser.add_argument('--K', default=64, type=int)
    parser.add_argument('--num_fore', default=40, type=int)
    parser.add_argument('--max_iter_epoch', default=200, type=int)
    parser.add_argument('--norm_mix', default=0, type=int)
    parser.add_argument('--use_parts', default=1, type=int)
    parser.add_argument('--lr_noise', default=1, type=float)
    parser.add_argument('--lr_mix', default=1, type=float)
    parser.add_argument('--num_parts_viz', default=4, type=int)
    parser.add_argument('--decay_mix', default=1, type=float)
    parser.add_argument('--masked_fore', default=0, type=float)
    parser.add_argument('--lambda_s', default=1, type=float)
    parser.add_argument('--lambda_o', default=1, type=float)
    parser.add_argument('--lambda_inv_ft', default=1, type=float)
    parser.add_argument('--lambda_inv_p', default=1, type=float)
    parser.add_argument("--mix_loss", type=str, default='L2', choices=["dice","L1","KL","L2","cos"])
    parser.add_argument('--corrupt_masks', default=0, type=int)
    parser.add_argument('--lr_noise_mask', default=1, type=float)
    parser.add_argument("--noise_tensor", type=str, default='gauss', choices=["gauss","salt","speckle"])

    parser.add_argument('--use_trC', default=1, type=int)
    
    # Model Parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the projection head output for [cls] tokens. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        the projection head output for [patch] tokens. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the projection head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # Temperature Teacher Parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
                        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
                        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')    

    # Training/Optimization Parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs of training.')
    parser.add_argument('--start_epoch', default=0, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop Parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--local_crops_number', type=int, default=10, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--global_crops_size', type=int, default=224,
                        help=""" Size of global crop. No need to change this parameter. """)
    parser.add_argument('--local_crops_size', type=int, default=96,
                        help=""" Size of local crop. No need to change this parameter.""")
    
    # Misc
    parser.add_argument('--data_path', default='/path/to/mini_imagenet/train', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--data_path_test', default='/path/to/mini_imagenet/train', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="checkpoint_triplet", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help="""If this is not None, then we will train our model starting from the
                              pretrained checkpoint path """)
    parser.add_argument('--pretrained_file', default=None, type=str,
                        help="""If this is not None, then we will train our model starting from the
                              pretrained checkpoint file """)
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument("--init_method", type=str,default='env://')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    # Loss
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for class-level
                        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
                        loss over masked [patch] tokens (Default: 1.0)""")
    parser.add_argument('--lambda3', default=0.0, type=float, help="""loss weight for MIM term (default 0.0)""")
    parser.add_argument('--supervised_loss', action="store_true", default=False,
                        help="whether or not to add additional supervised CE loss") 
    parser.add_argument("--supervised_contrastive",action="store_true", default=False,
                        help=" whether or not to use supervised contrastive loss ")    
    parser.add_argument('--celoss_coefficient', type=float, default=0.0, 
                        help="coefficient for supervised cross-entropy loss")
    parser.add_argument('--weighted_pool', default=False, type=utils.bool_flag, 
                        help=""" If True, then the patch-level loss are calculated as 
                        the attention-coefficients weighted average of all patch-matching losses. (Defeault: False)""")
    
    # Few shot evaluation
    parser.add_argument('--evaluation_batch_size_per_gpu', default=225, type=int, help='Per-GPU batch-size')
    parser.add_argument('--evaluate_freq', type=int, default=50,
                        help='the frequency of doing few shot evaluation. None: not evaluate during training. (Default: 50)')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, 
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--cls_tokens', default=True, type=utils.bool_flag,
        help="""Whether ot not to use the [CLS] token during few-shot evaluation.
        We typically set this to True """)
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False """)
    parser.add_argument('--weighted_avgpool_patchtokens', default=True, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global weighted average pooled features to the [CLS] token.
        We typically set this to True """)
    parser.add_argument('--p_scale', type=float, default=1.0)
    parser.add_argument('--wp_scale', type=float, default=1.0)
    
    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--partition', default='test', type=str) 
    parser.add_argument('--save_features', default=1, type=int, 
                        help='whether or not to save the features after each evaluation')
    parser.add_argument('--evaluation_method', default='cosine', type=str, 
                        choices=['cosine', 'knn', 'classifier'])  
    parser.add_argument('--top_k', type=int, default=5)
    
    parser.add_argument('--img_size_imagenet', type=int, default=256, 
                        help='resize the image before evaluation')
    parser.add_argument('--img_resize_imagenet', type=int, default=224, 
                        help='center crop the resized image')
    parser.add_argument('--img_size_cifar', type=int, default=256, 
                        help='resize the image before evaluation')
    parser.add_argument('--img_resize_cifar', type=int, default=224, 
                        help='center crop the resized image')
    
    # Visualization
    parser.add_argument('--visualization_freq', type=int, default=None,
                        help='the frequency of doing visualization. None: not visualize')
    parser.add_argument('--image_path', type=str, default='./img.png',
                        help='Will use the default bird. If none, then the bird will also be used (but not stable)')
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224],
                        help='default image size to be used for visualization')
    parser.add_argument('--image_threshold', type=float, default=None,
                        help='threshold to mask out x% of the image')

    # Additional args from ibot 
    parser.add_argument('--pred_ratio', default=[0.0, 0.3], type=float, nargs='+', help="""Ratio of partial prediction.
                        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=[0.0, 0.2], type=float, nargs='+', help="""Variance of partial prediction
                        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
                        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--norm_in_head', default=None,
                        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
                        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
                        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--shared_head', default=True, type=utils.bool_flag, help="""Wether to share 
                        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
                        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
                        Only works for teacher model. (Defeault: True)""")

    return parser

def train_dpvit(rank=None, world_size=None, args=None):

    if args.use_DDP:
        if args.local:
            device = torch.device('cuda:{}'.format(rank))
            local_setup(rank, world_size, str(9997))
            torch.cuda.set_device(rank)
        else:
            rank = dist_setup(args)
            device = torch.device('cuda:{}'.format(rank))
            #dist.barrier()
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.fix_random_seeds(args.seed)
    
    if utils.is_main_process():
        print("git:\n  {}\n".format(utils.get_sha()))
    #print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ set up wandb ... ==============
    if args.prod_mode:
        if rank>0:
            print ('\n shutting wandb for multiple GPUs. Will only run for rank:0 process. \n')
            os.environ["WANDB_MODE"] = "offline"
        
        exp_name = args.output_dir.split("/")[-1]
        import wandb
        if utils.is_main_process():
            wandb.init(settings=wandb.Settings(start_method="fork"), config=vars(args),\
                        project=args.wandb_project_name, save_code=True, name=exp_name)
            #wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=False, config=vars(args),
            #           name=exp_name, monitor_gym=False, save_code=True)
    else:
        wandb = None
    
    # ============ building data loader ... ==============
    args.dataset_name = args.data_path.split('/')[-2] 
    
    if utils.is_main_process():
        print(args) # to print to the shell
    
    transform = utils.DataAugmentation_DPViT(args.global_crops_scale, args.local_crops_scale,
                                           args.global_crops_number, args.local_crops_number, 
                                           args.global_crops_size, args.local_crops_size,
                                           args.dataset_name, args=args)
    dataset = ImageFolderMask(args.data_path, 
                             transform=transform,
                             trans=transform,
                             patch_size=args.patch_size,
                             pred_ratio=args.pred_ratio,
                             pred_ratio_var=args.pred_ratio_var,
                             pred_aspect_ratio=(0.3, 1/0.3),
                             pred_shape=args.pred_shape,
                             pred_start_epoch=args.pred_start_epoch)
    
    if args.use_DDP:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset,
                sampler=sampler, batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    if utils.is_main_process():
        print(f"Data loaded: there are {len(dataset)} images.")


    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            args=args,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
            args=args,
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    student = utils.MultiCropWrapper_dpvit(student, 
                iBOTHead(embed_dim,args.out_dim,patch_out_dim=args.patch_out_dim,norm=args.norm_in_head,\
                        act=args.act_in_head,norm_last_layer=args.norm_last_layer,shared_head=args.shared_head))
    
    teacher = utils.MultiCropWrapper_dpvit(teacher, 
                iBOTHead(embed_dim, args.out_dim,patch_out_dim=args.patch_out_dim,norm=args.norm_in_head,\
                act=args.act_in_head,shared_head=args.shared_head_teacher))

    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student) and args.use_DDP:
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        
        student = student.to(device)
        teacher = teacher.to(device)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
        teacher_without_ddp.load_state_dict(student.state_dict(), strict=False)
    
    if args.use_DDP:
        student = nn.parallel.DistributedDataParallel(student, device_ids=[rank], output_device=rank, find_unused_parameters=False) # don't run this line when debugging
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    else:
        teacher_without_ddp.load_state_dict(student.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    if utils.is_main_process():
        print(f"Student and Teacher are built: they are both {args.arch} network.")
        pytorch_total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"Total trainable params : {pytorch_total_params}")
    
    # ============ preparing loss ... ===========
    num_train_classes = {'mini-imagenet-480': 64, 'cifar-fs': 64, 'in9':9, 'Imagenet':1000}
    
    # supervised loss
    if args.supervised_loss == True:
        ce_loss = loss_functions.CELoss(nclasses = num_train_classes[args.dataset_name],
                         in_dim = args.out_dim, batch_size = args.batch_size_per_gpu ).cuda()
    else:
        ce_loss = 0

    if not args.use_masked_im_modeling:
        assert args.lambda3 == 0, "lambda3 should be set as zero if we do not use MIM"
    same_dim = args.shared_head or args.shared_head_teacher
    
    # ibot loss
    ibot_loss = loss_functions.IBOT_Loss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        mim_start_epoch=args.pred_start_epoch, 
        batch_size_per_gpu = args.batch_size_per_gpu, 
        patch_num_global_crops = (args.global_crops_size // args.patch_size)**2, 
        weighted_pool = args.weighted_pool, 
        args=args,
    ).cuda()
        
    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
    
    # ============ preparing optimizer ... ============   
    params_groups = utils.get_params_groups(student)
    
    if args.supervised_loss == True:
        params_groups[0]['params'] = params_groups[0]['params'] + list(ce_loss.parameters())

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None 
        
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        #args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.lr,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
   
    # ============ optionally resume training ... ============    
    preempt_flag = False
    restart_checkpoint_path = args.output_dir
    restart_checkpoint_file = 'checkpoint.pth'

    if args.pretrained_path is not None:
        restart_checkpoint_path = args.pretrained_path
        if args.pretrained_file is not None:
            restart_checkpoint_file = args.pretrained_file
        else:
            print("Need to specify pretrained file", force=True)
            sys.exit(1)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        preempt_flag = True
        print("# restart checkpoint path {}, restart checkpoint file {} #".format(restart_checkpoint_path, restart_checkpoint_file))

    args = utils.restart_from_checkpoint(
        os.path.join(restart_checkpoint_path, restart_checkpoint_file),
        run_variables={"epoch": 0},
        student=student,
        teacher=teacher, 
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        ce_loss = ce_loss,
        ibot_loss=ibot_loss,
        args=args,
        preempt_flag=preempt_flag,
    )
 
    student, teacher = student.cuda(), teacher.cuda()

    # ============ start training ... ============   
    if utils.is_main_process():
        print("Starting DPViT training !\n")

    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.use_DDP:
            data_loader.sampler.set_epoch(epoch)
            data_loader.dataset.set_epoch(epoch)
        
        #print ('start')
        st_tr = time.time()
        # ============ training one epoch of SMKD ... ============
        train_stats, stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, ce_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, wandb)
        
        args.lr_mix = args.lr_mix * args.decay_mix

        end_tr = time.time()
        if epoch == 0:
            misc_time = 0

        if utils.is_main_process():
            print ('\n Stats \n ')
            print('Epoch time {}'.format(str(datetime.timedelta(seconds=int(end_tr - st_tr)))))
            print('Fwd time {}'.format(str(datetime.timedelta(seconds=int(sum(stats[0]))))))
            print('Bck time {}'.format(str(datetime.timedelta(seconds=int(sum(stats[1]))))))
            print('F+B time {}'.format(str(datetime.timedelta(seconds=int(sum(stats[2]))))))
            print('Loss calc time {}'.format(str(datetime.timedelta(seconds=int(sum(stats[3]))))))
            print('Data load time {}'.format(str(datetime.timedelta(seconds=int(end_tr - st_tr) - int(sum(stats[2]))))))
            print('No. of batches seen {}'.format(str(int(stats[-1]))))
            print('Samples seen {}'.format(str(args.batch_size_per_gpu*args.n_gpus*int(stats[-1]))))

        st_ms = time.time()
        #pdb.set_trace()
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if args.supervised_loss:
            save_dict['ce_loss'] = ce_loss.state_dict()
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
            utils.save_tb(log_stats)

            # few shot evaluation
            if args.evaluate_freq is not None and epoch>0:
                if (epoch % int(args.evaluate_freq) == 0) or (epoch == args.epochs - 1):
                    print("\n###### 5 way 1 shot ######")
                    args.num_shots = 1
                    utils.evaluation(args, vits, epoch, rollout=False, 
                                     evaluation_method=args.evaluation_method, top_k=args.top_k, 
                                     img_size_imagenet=args.img_size_imagenet, 
                                     img_resize_imagenet=args.img_resize_imagenet, 
                                     img_size_cifar=args.img_size_cifar, 
                                     img_resize_cifar=args.img_resize_cifar)

                    print("\n###### 5 way 5 shot ######")
                    args.num_shots = 5
                    utils.evaluation(args, vits, epoch, rollout=False, 
                                     evaluation_method=args.evaluation_method, top_k=args.top_k, 
                                     img_size_imagenet=args.img_size_imagenet, 
                                     img_resize_imagenet=args.img_resize_imagenet, 
                                     img_size_cifar=args.img_size_cifar, 
                                     img_resize_cifar=args.img_resize_cifar)
                    
            if args.visualization_freq is not None and epoch>0:
                if (epoch % int(args.visualization_freq) == 0) or (epoch == args.epochs - 1):
                    print("\n###### visualize parts ######")
                    utils.visualize(args, vits, epoch, image_path = args.image_path, image_size = args.image_size, threshold = args.image_threshold, rollout=False)

        misc_time = time.time() - st_ms
    
    if utils.is_main_process():
        print('Training time {}'.format(
            str(datetime.timedelta(seconds=int(time.time() - start_time)))))

def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, ce_loss, 
                    data_loader, optimizer, lr_schedule,  wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, wandb=None):
    
    #st_dat = time.time()
    fwd, bck, data_ld, syn = [],[],[], []
    count = 0
    return_attention = False if args.lambda2 == 0 else True 
    return_backbone_feat = True if args.lambda2 > 0 else False 
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []

    if args.use_DDP:
        student_params = student.module.named_parameters()
    else:
        student_params = student.named_parameters()

    for name_q, param_q in student_params:
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]    
    pred_labels, real_labels = [], []
    
    for it, next_data in enumerate(metric_logger.log_every(data_loader, 40, header)):
        # update weight decay and learning rate according to their schedule
        
        if it == args.max_epoch_iter:
            break

        images, mask_fb, labels, masks = next_data

        st_dat = time.time()
        count += 1
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        labels = labels.cuda()

        mask_fore = [msk.cuda(non_blocking=True) for msk in mask_fb[0]]
        mask_back = [msk.cuda(non_blocking=True) for msk in mask_fb[1]]

        st_fwd = time.time()
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            if np.random.random() < args.masked_fore:
                fore_images = [i*j for i in images[:args.global_crops_number] for j in mask_fore]
            else:
                fore_images = images
 
            teacher_output = teacher(images[:args.global_crops_number], 
                                     return_backbone_feat = return_backbone_feat, 
                                     return_attention=return_attention)
            
            student_output = student(fore_images[:args.global_crops_number], 
                                     mask=masks[:args.global_crops_number],
                                     return_backbone_feat = return_backbone_feat,
                                     return_attention=return_attention)
            
            if args.use_DDP:
                student.module.backbone.masked_im_modeling = False
                student_local_cls = student(images[args.global_crops_number:])[0][0] if len(images) > args.global_crops_number else None
                
                student.module.backbone.masked_im_modeling = args.use_masked_im_modeling
            else:
                student.backbone.masked_im_modeling = False
                student_local_cls = student(images[args.global_crops_number:])[0][0] if len(images) > args.global_crops_number else None
                
                student.backbone.masked_im_modeling = args.use_masked_im_modeling

            end_fwd = time.time()
            fwd.append(end_fwd-st_fwd)

            st_syn = time.time()
            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch, 
                                 labels=labels if args.supervised_contrastive else None ,)
            loss = all_loss.pop('loss')
            

            if args.use_parts and args.lr_mix:

                mix_loss = loss_functions.mix_loss(student_output[-1][0],torch.cat(mask_fore), loss_type=args.mix_loss, args=args) +\
                            loss_functions.mix_loss(student_output[-1][1], torch.cat(mask_back), loss_type=args.mix_loss, args=args)

                loss = loss + args.lr_mix * mix_loss

                if args.use_DDP:
                    norm_s = LA.norm(student.module.backbone.blocks[-1].P, ord=1) 
                    norm_o = student.module.backbone.ortho_loss()
                else:
                    norm_s = LA.norm(student.backbone.blocks[-1].P, ord=1)
                    norm_o = student.backbone.ortho_loss()

                if args.lambda_s and args.lambda_o:
                    loss = loss + (args.lambda_s * norm_s + args.lambda_o*norm_o)
                elif args.lambda_o:
                    loss = loss +  args.lambda_o*norm_o
                elif args.lambda_s:
                    loss = loss + args.lambda_s * norm_s
            else:
                norm_s = LA.norm(student.module.backbone.blocks[-1].P, ord=1) 
                norm_o = student.module.backbone.ortho_loss()
                mix_loss = torch.tensor([0.0]).cuda()

            if args.supervised_loss == True:
                try:
                    celoss = ce_loss(student_output[1][0][:2*args.batch_size_per_gpu], labels.cuda())
                except:
                    celoss = ce_loss(student_output[0][0][:2*args.batch_size_per_gpu], labels.cuda())
                    
                all_loss['ce_loss'] = celoss * args.celoss_coefficient
                loss = loss + all_loss['ce_loss']
            
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        if args.lambda2 == 0:
            probs1 = teacher_output[0][0].chunk(args.global_crops_number)
            probs2 = student_output[0][0].chunk(args.global_crops_number)
        else:
            probs1 = teacher_output[1][0].chunk(args.global_crops_number)
            probs2 = student_output[1][0].chunk(args.global_crops_number)
            
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1], args) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1], args)
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device), args))

        end_syn = time.time()
        syn.append(end_syn - st_syn)

        st_bck = time.time()
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)

            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging

        if args.use_DDP:
            torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(norm_nc=norm_s.item())
        metric_logger.update(norm_sdc=norm_o.item())
        metric_logger.update(mix_loss=mix_loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)
        
        # write stats to wandb
        if args.prod_mode:
            if utils.is_main_process():
                data_to_log = {
                   'pretrain/total_loss': loss.item(),
                   'pretrain/mix_loss': mix_loss.item(),
                   'pretrain/norm_nc': norm_s.item(),
                   'pretrain/norm_sdc': norm_o.item(),
                   'pretrain/cls_loss': all_loss['cls'].item(),
                   'pretrain/patch_loss': all_loss['patch'].item(),
                   'pretrain/mim_loss': all_loss['mim'].item(),
                   'pretrain/learning_rate': optimizer.param_groups[0]["lr"],
                   'pretrain/num_epochs': epoch,
                   'pretrain/acc': acc,
                }
                if args.supervised_loss == True:
                    data_to_log['pretrain/ce_loss'] = all_loss['ce_loss'].item() * args.celoss_coefficient
                wandb.log(data_to_log)

        end_bck = time.time()
        bck.append(end_bck-st_bck)
        end_dat = time.time()
        data_ld.append(end_dat - st_dat)

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes

    if args.use_DDP:
        metric_logger.synchronize_between_processes()

    if utils.is_main_process():
        print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
        print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})

    return return_dict, [fwd, bck, data_ld, syn, count]

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser('DPViT', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    #args.output_dir += f'_{socket.gethostname()}_{args.timestr}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.save_command(os.path.join(args.output_dir, 'command.txt'))

    if args.use_DDP and args.local:
        mp.spawn(
                train_dpvit,
                args=(args.n_gpus, args, ),
                nprocs=args.n_gpus,
                daemon=False,
                join=True,
            )
        exit()
    train_dpvit(args=args)