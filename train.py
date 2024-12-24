import argparse
import copy
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from lib.datasets import build_dataset, build_transform
from engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from model.vision_transformer import VisionTransformer
import sklearn.cluster as cluster

import model as models
from timm.models import load_checkpoint
import os


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # ship related parameters
    parser.add_argument('--share_threshold', type=float, default=1.0, metavar='PCT',
                        help='Similarity threshold across layers (default: 1.0)')
    parser.add_argument('--end_layer', type=int, default=12, help='insert prompt up to N layer.')

    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--pretrained_feature_exp', type=str, default='zeroshot', help='extracted feature path')
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'vp','retrain','search'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    # parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

    # parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--test', action='store_true', help='using test-split or validation split')
    parser.add_argument('--save_trainset_feature', action='store_true', help='save trainset feature')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help='train data percentage for FGVC tasks.')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    # parser.set_defaults(amp=True)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--is_visual_prompt_tuning', action='store_true')
    parser.add_argument('--is_adapter', action='store_true')
    parser.add_argument('--is_LoRA', action='store_true')
    parser.add_argument('--is_prefix', action='store_true')

    parser.add_argument('--no_aug', action='store_true')
    
    parser.add_argument('--val_interval', default=10, type=int, help='validataion interval')

    parser.add_argument('--drop_rate_LoRA', type=float, default=0.1)
    parser.add_argument('--drop_rate_prompt', type=float, default=0.1)
    parser.add_argument('--drop_rate_adapter', type=float, default=0.1)

    parser.add_argument('--few-shot-seed', type=int, default=0)
    parser.add_argument('--few-shot-shot', type=int, default=2)

    parser.add_argument('--inception',action='store_true')
    parser.add_argument('--direct_resize',action='store_true')

    parser.add_argument('--IS_not_position_VPT',action='store_true')

    parser.set_defaults(no_aug=True)
    parser.set_defaults(inception=True)
    parser.set_defaults(direct_resize=True)
    parser.set_defaults(IS_not_position_VPT=True)

    return parser


def dfs(adj_matrix, visited, node, group, alpha):
    visited[node] = True
    group.append(node)
    for neighbor in range(node + 1, len(adj_matrix)):
        similarity = adj_matrix[node][neighbor]
        if not visited[neighbor]:
            visited[neighbor] = True
            if similarity < alpha:
                dfs(adj_matrix, visited, neighbor, group, alpha)


def group_elements(adj_matrix, alpha):
    n = len(adj_matrix)
    visited = [False] * n
    groups = []

    group = []
    dfs(adj_matrix, visited, 0, group, alpha)
    groups.append(group)

    return groups[0]

def cosine_similarity_matrix(features):
    norm_features = features / np.linalg.norm(features, axis=-1, keepdims=True)
    similarity_matrix = np.matmul(norm_features, norm_features.transpose(1, 0))
    return similarity_matrix

def batch_similarity_matrix(features):
    n, batch_size, dim = features.shape
    similarity_matrices = []

    for i in range(batch_size):
        similarity_matrix = cosine_similarity_matrix(features[:, i])
        similarity_matrices.append(similarity_matrix)

    avg_similarity_matrix = np.mean(similarity_matrices, axis=0)
    return avg_similarity_matrix


def _coarse_clustering(args, n_cluster=200):
    prototype_save_path = './save_attribute/{}/prototype_{}_clusters.npy'.format(args.data_set, n_cluster)
    if os.path.exists(prototype_save_path):
        prototype_gather = torch.tensor(np.load(prototype_save_path))
        print("Nums of prototypes of coarse clusters for test dataset: {}".format(prototype_gather.size(0)))
        return prototype_gather

    features = np.load('./saves/{}_{}/feats.npy'.format(args.data_set, args.pretrained_feature_exp))
    
    if features.shape[1] > 1000:
        indices = np.random.choice(features.shape[1], size=1000, replace=False)
        features = features[:, indices, :]

    features = torch.tensor(features)

    kmeans = cluster.KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(features[-1].cpu().numpy())
    y_pred = kmeans.labels_ 

    y_pred = torch.from_numpy(y_pred)
    coarse_class_idx = torch.unique(y_pred)
    num_coarse_classes = len(coarse_class_idx)
    print("Nums of coarsely divided categories for test dataset: {}".format(num_coarse_classes))

    prototype_gather = []
    for i in range(len(coarse_class_idx)):
        pos = torch.where(y_pred == i)[0]
        prototype = features[:, pos].mean(1).unsqueeze(1)
        prototype_gather.append(prototype)

    prototype_gather = torch.cat(prototype_gather, dim=1)[-1]
    print("Nums of prototypes of coarse clusters for test dataset: {}".format(prototype_gather.size(0)))

    os.makedirs("save_attribute/{}".format(args.data_set), exist_ok=True)
    np.save(prototype_save_path, prototype_gather.cpu().numpy())
    return prototype_gather



def main(args):
    update_config_from_file(args.cfg)
    if args.launcher == 'none':
        args.distributed = False

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args,is_individual_prompt=(args.is_visual_prompt_tuning or args.is_adapter or args.is_LoRA or args.is_prefix))
    dataset_val, _ = build_dataset(is_train=False, args=args,is_individual_prompt=(args.is_visual_prompt_tuning or args.is_adapter or args.is_LoRA or args.is_prefix))

    print(f"{args.data_set} dataset, train: {len(dataset_train)}, evaluation: {len(dataset_val)}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.save_trainset_feature:
        dataset_train_temp = copy.deepcopy(dataset_train)
        dataset_train_temp.transform = build_transform(False, args)
        trainset_sampler = torch.utils.data.SequentialSampler(dataset_train)
        trainset_dataloader = torch.utils.data.DataLoader(
            dataset_train_temp, batch_size=int(2 * args.batch_size),
            sampler=trainset_sampler, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )
        original_model = models.__dict__[cfg.MODEL_NAME](   
                img_size=args.input_size,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path, visual_prompt_indices=[], # duplicate_prompt=cfg.SUPERNET.DUPLICATE,
                super_prompt_tuning_dim=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                super_LoRA_dim=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                super_adapter_dim=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                super_prefix_dim=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                drop_rate_LoRA=args.drop_rate_LoRA,
                drop_rate_prompt=args.drop_rate_prompt,
                drop_rate_adapter=args.drop_rate_adapter,
                )
        original_model.eval()
        original_model = original_model.to(device)
        return

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    mixup_fn = None
    
    print(f"Creating SuperVisionTransformer")
    print(cfg)

    share_threshold = args.share_threshold
    if share_threshold < 1.0:
        if os.path.exists("save_affinity/{}/affinity.npy".format(args.data_set)):
            adj_matrix = np.load("save_affinity/{}/affinity.npy".format(args.data_set))
        else:
            features = np.load('./saves/{}_{}/feats.npy'.format(args.data_set, args.pretrained_feature_exp))
            adj_matrix = batch_similarity_matrix(features)
            os.makedirs("save_affinity/{}".format(args.data_set), exist_ok=True)
            np.save("save_affinity/{}/affinity.npy".format(args.data_set), adj_matrix)

        visual_prompt_indices = group_elements(adj_matrix, share_threshold)
    else:
        visual_prompt_indices = list(range(12))

    visual_prompt_indices = [x for x in visual_prompt_indices if x < args.end_layer]

    print("Visual Prompting in {} layers.".format(visual_prompt_indices))

    model = models.__dict__[cfg.MODEL_NAME](   
                img_size=args.input_size,
                drop_rate=args.drop, 
                drop_path_rate=args.drop_path, visual_prompt_indices=visual_prompt_indices, # duplicate_prompt=cfg.SUPERNET.DUPLICATE,
                super_prompt_tuning_dim=cfg.RETRAIN.VISUAL_PROMPT_DIM, 
                super_LoRA_dim=cfg.SUPERNET.LORA_DIM,
                super_adapter_dim=cfg.SUPERNET.ADAPTER_DIM,
                super_prefix_dim=cfg.SUPERNET.PREFIX_DIM,
                drop_rate_LoRA=args.drop_rate_LoRA,
                drop_rate_prompt=args.drop_rate_prompt,
                drop_rate_adapter=args.drop_rate_adapter,
                )

    model.feature_bank = _coarse_clustering(args, 200).to(device)

    choices = {'depth': cfg.SUPERNET.DEPTH,
               'super_prompt_tuning_dim':cfg.RETRAIN.VISUAL_PROMPT_DIM,
               'super_LoRA_dim':cfg.SUPERNET.LORA_DIM,
               'super_adapter_dim':cfg.SUPERNET.ADAPTER_DIM,
               'super_prefix_dim':cfg.SUPERNET.PREFIX_DIM,
               }

    if args.resume:
        if "linear-vit-b-300ep.pth.tar" in args.resume:
            model.reset_classifier(1000)
            incompatible_keys = load_checkpoint(model, args.resume, strict=False)
            print(incompatible_keys)
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)
        elif 'pth' in args.resume:
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)
            incompatible_keys = load_checkpoint(model, args.resume,strict=False)
            print(incompatible_keys)
        else:
            load_checkpoint(model, args.resume)
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)

    model.to(device)
    
    teacher_model = None
    teacher_loss = None

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    optimizer = create_optimizer(args, model_without_ddp)


    total_params = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            total_params += param.numel()

    print(f"Optimizer total parameters: {total_params}")


    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # save config for later experiments
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)

    retrain_config = None
    if args.mode == 'retrain' and "RETRAIN" in cfg:
        retrain_config = {'visual_prompt_dim':cfg.RETRAIN.VISUAL_PROMPT_DIM,
                          'lora_dim':cfg.RETRAIN.LORA_DIM,'adapter_dim':cfg.RETRAIN.ADAPTER_DIM,'prefix_dim':cfg.RETRAIN.PREFIX_DIM}

    if args.save_trainset_feature:
        evaluate(trainset_dataloader, model, device, output_dir=args.output_dir, mode=args.mode, save_feature=False,
                              retrain_config=retrain_config, is_visual_prompt_tuning=args.is_visual_prompt_tuning,
                              is_adapter=args.is_adapter, is_LoRA=args.is_LoRA, is_prefix=args.is_prefix)
        return

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, output_dir=args.output_dir, mode = args.mode, save_feature=False,
                              retrain_config=retrain_config,is_visual_prompt_tuning=args.is_visual_prompt_tuning,
                              is_adapter=args.is_adapter,is_LoRA=args.is_LoRA,is_prefix=args.is_prefix)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        use_matching_loss = cfg.SUPERNET.MATCHING_LOSS and epoch >= 10  # warm up epoch = 10

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            amp=args.amp, teacher_model=teacher_model,
            teach_loss=teacher_loss, matching_loss=use_matching_loss,
            choices=choices, mode = args.mode, retrain_config=retrain_config,
            is_visual_prompt_tuning=args.is_visual_prompt_tuning,
            is_adapter=args.is_adapter,is_LoRA=args.is_LoRA,is_prefix=args.is_prefix,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        if epoch % args.val_interval == 0 or epoch == args.epochs-1:
            test_stats = evaluate(data_loader_val, model, device, output_dir=args.output_dir, amp=args.amp, choices=choices, mode = args.mode, retrain_config=retrain_config,is_visual_prompt_tuning=args.is_visual_prompt_tuning,is_adapter=args.is_adapter,is_LoRA=args.is_LoRA,is_prefix=args.is_prefix)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'max_accuracy': max_accuracy,
                        'epoch': epoch,
                        'prompt_indices': model.visual_prompt_indices,
                        'n_parameters': total_params}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SHIP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(' '.join([f'--{arg} {getattr(args, arg)}' for arg in vars(args)]) + '\n')
    main(args)
