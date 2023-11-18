import os
import argparse
import datetime
import json
import time
import copy
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetuning import train_one_epoch, val_one_epoch
from transformers import BertTokenizer, GPT2Tokenizer
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from models_clip import clip_model
from pointclip.clip import clip

from dataloader import ModelNet40
from dataloader import modelnet40_collate
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./llama', type=str, help='path of llama model')
    parser.add_argument('--model', default='llama_adapterModel', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--size', default='7B', type=str, help="Llama model size")
    parser.add_argument('--adapter_layer', type=int, default=30, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')
    parser.add_argument('--generate', action='store_true', help='vaq loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str, help="Name of dataset to use")
    parser.add_argument('--n_candidates', default=5, type=int)
    parser.add_argument('--sub',action='store_true', help="to use subtitles/diaglogue/script")
    parser.add_argument('--data_path', default='/instruction_dataset/', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--mgpu', action='store_true')

    
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vaq', action='store_true', help='vaq loss')
    parser.add_argument('--qav', action='store_true', help='qav loss')
    parser.add_argument('--v_loss', action='store_true', help='video loss')
    parser.add_argument('--bias', type=float, default=3., help='attention bias')

    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb name')
    parser.add_argument('--flag', type=str, default=None, help='flag')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--mc', action='store_true', help="Multiple Choice dataset")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--addrotation', action='store_true')
    

    return parser

def load_data(args, tokenizer, split='train', reason=False):

    if args.dataset == "modelnet40":
        dataset = ModelNet40(args=args, tokenizer=tokenizer, split=split)
        collate_dataset = modelnet40_collate
    else:
        raise NotImplementedError("Dataset not available")

    return dataset, collate_dataset

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # tokenizer = Tokenizer( args.model, args.llama_model_path )
    tokenizer = clip.tokenize
    print("#" * 15, f"Loading dataset {args.dataset}", "#" * 15)

    dataset_train, collate_dataset = load_data(args, tokenizer, split='train')
    dataset_val, collate_dataset = load_data(args, tokenizer, split='test')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if not args.mgpu:  # args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        if not args.evaluate:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        if args.wandb_project is not None:
            wandb.init(project=args.wandb_project, config=args, name=args.wandb_name, entity='mejilee')
    else:
        log_writer = None

    if not args.evaluate:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_dataset, pin_memory=args.pin_mem, drop_last=False)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_dataset,
                                                  pin_memory=args.pin_mem, drop_last=False)

    # define the model
    model = clip_model(args)
    model_without_ddp = model
    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    lr_orig = args.lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    model_without_ddp = model.module
    # following timm: set wd as 0 for bias and norm layers

    loss_scaler = NativeScaler(args)
    best_loss = 1000000

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.save_model:
        model_name = 'checkpoint_best'
        misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=args.start_epoch-1, name=model_name)
        breakpoint()
    
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({"lr": args.blr, "wd": args.weight_decay, "accum_inter": args.accum_iter, "batch": args.batch_size, "warm_up epochs": args.warmup_epochs}) + "\n")
    
    # Train
    if not args.evaluate:  
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            if not args.mgpu:
                data_loader_train.sampler.set_epoch(epoch)
                data_loader_val.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)
            val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, device, epoch, loss_scaler, tokenizer, log_writer=log_writer, args=args)

            if args.output_dir and best_loss >= val_stats['loss']:
                print("Saving checkpoint...")
                best_loss = val_stats['loss']
                model_name = 'checkpoint_best'
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)

            if global_rank == 0 and args.wandb_project is not None:
                wandb.log({'acc': val_stats['loss'], 'best_loss': best_loss, 'learning_rate': train_stats['lr'], 'loss': train_stats['loss'], 'rotation':args.addrotation})

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, **{f'val_{k}': v for k, v in val_stats.items()}}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    # Evaluate
    else:
        epoch = 0
        print("#####################")
        print(f"Start Evaluation")
        print("#####################")

        start_time = time.time()
        data_loader_val.sampler.set_epoch(epoch)
        
        val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, device, epoch, loss_scaler, tokenizer, log_writer=log_writer, args=args)
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "eval.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
