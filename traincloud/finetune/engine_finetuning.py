import math
import sys
from typing import Iterable
import os
import json

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.metric import updateDatasetMetric
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import pdb


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None):
  
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        pc = data['data'].cuda()
        text_id = data['text_id'].cuda()
        label = data['label']
        classname = data['classname']

        with torch.autocast(device_type='cuda'): 
            logits, loss = model(pc, text_id )
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler,
                  tokenizer, log_writer=None, args=None, reason=False, metric_logger=None, validation=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
        
    header = 'Eval Epoch: [{}]'.format(epoch)
    print_freq = 50

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        pc = data['data'].cuda()
        text_id = data['text_id'].cuda()
        label = data['label']
        classname = data['classname']

        with torch.no_grad():
            logits, loss = model(pc, text_id )
            loss_value = loss.item()
            # breakpoint()

        metric_logger.update(loss=loss_value)
            
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger, validation