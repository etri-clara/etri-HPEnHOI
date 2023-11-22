# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import logging
import numpy as np
import math
import copy
import contextlib
import shutil
import wandb
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators

from .distributed_trainer import DistributedTrainer
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable

logger = logging.getLogger(__name__)


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path =  os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        self.pipeline = pipeline_class(self.opt)
        
        self.wdb = None
        if opt["WANDB"]:
            self.wdb:wandb = opt["WANDB"]

    def eval(self, ):
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.mode = "eval"

        # self.model_names, self.raw_models, self.criteria = self.pipeline.set_up_model()
        self.raw_models = self.pipeline.initialize_model()
        self.model_names = self.raw_models.keys()

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        # This means the model will be evaluated without loading the pretrained weights, perhaps for profiling reasons.
        eval_without_loading = self.opt.get('DONT_LOAD_MODEL', False)

        if not eval_without_loading:
            if os.path.isfile(self.opt['PYLEARN_MODEL']):
                model_path = self.opt['PYLEARN_MODEL']  # this is a directory, not a file
                self.load_model(model_path)
            else:
                raise ValueError(f"Model directory not found: {model_path}")

        results = self._eval_on_set(self.save_folder)
        return results

    def _eval_on_set(self, save_folder):
        logger.info(f"Evaluation start ...")
        results = self.pipeline.evaluate_model(self, save_folder)
        if self.opt['rank'] == 0:
            logger.info(results)
        return results

    def compute_loss(self, forward_func, batch):

        def forward(func, trainer, batch):
            if self.opt['FP16']:
                from torch.cuda.amp import autocast
                with autocast():
                    loss = func(trainer, batch)
            else:
                loss = func(trainer, batch)
            return loss
        loss = forward(forward_func, self, batch)
        return loss

    def backward_loss(self, loss, model_names=['default']):  # noqa: E252

        def backward(loss_tensor):
            if self.opt['FP16']:
                self.grad_scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()
            
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps

        backward(loss)
        return loss

    def update_model(self, model_name='default'):
        if self.opt['FP16']:
            self.grad_scaler.unscale_(self.optimizers[model_name])
            self.grad_scaler.step(self.optimizers[model_name])
        else:
            self.optimizers[model_name].step()

        self.optimizers[model_name].zero_grad()
        self.train_params['optim_steps'][model_name] += 1

    def train_step(self, batch):
        self.grad_acc_batches.append(batch) # support batch accumulation

        if self.is_gradient_accumulation_boundary():
            # set all modules and criteria into training mode
            for model_name in self.model_names:
                self.models[model_name].train()

            # assert len(self.grad_acc_batches) == self.grad_acc_steps

            total_batch_sample = 0
            for batch_index, batch in enumerate(self.grad_acc_batches):

                loss_info, sample_size_info, extra_info = \
                    self.pipeline.forward_step(self,
                                            batch,
                                            self.grad_acc_batches,
                                            batch_index,
                                            is_distributed=(self.opt['world_size'] > 1))
                # print(loss_info)
                self.train_loss.update_iter(loss_info)
                total_batch_sample += sample_size_info['num_samples']

            if self.opt['FP16']:
                # Update GradScaler after an effective batch
                self.grad_scaler.update()

            # update losses and item counts of an effective batch to the AverageMeters
            if self.opt['world_size'] > 1:
                total_batch_sample = torch.tensor(total_batch_sample).to(self.opt['device'])
                torch.distributed.all_reduce(total_batch_sample, torch.distributed.ReduceOp.SUM)
                total_batch_sample = total_batch_sample.item()

            self.train_params['total_batch_size'] += total_batch_sample
            self.grad_acc_batches = []
        self.train_params['num_updates'] += 1
        
    def init_train(self):
        self.mode = "train"
        logger.info('-------------------------------------------------------')
        logger.info("Training on rank: {}".format(self.opt['rank']))

        self.raw_models = self.pipeline.initialize_model()
        self.model_names = list(self.raw_models.keys())

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        self.train_dataloaders = self.pipeline.get_dataloaders(self, 'train', is_evaluation=False)
        self.dataset_length = len(self.train_dataloaders)
        self.train_params = {
                             "updates_per_epoch": self.dataset_length,
                             "total_batch_size": 0,
                             "num_updates": 0,
                             "optim_steps": {module_name: 0 for module_name in self.model_names},
                             "start_epoch_idx": 0,
                             "start_batch_idx": 0,
                             "current_epoch_idx": 0,
                             "current_batch_idx": 0,
                             }

        self.train_loss = LossMeter()
        self.grad_acc_batches = []

        if self.opt['CUDA']:
            torch.cuda.empty_cache()

        self.create_optimizer_and_scheduler()
        self.models = {model_name: self.raw_models[model_name] for model_name in self.model_names}
        self._initialize_ddp()

        if self.opt.get('WEIGHT', False):
            self.load_weight(self.opt['RESUME_FROM'], must_exist=True)
        if self.opt.get('RESUME', False):
            self.load_checkpoint(self.opt['RESUME_FROM'], must_exist=True)

        ######################
        # Start the main loop
        ######################
        if self.opt['rank'] == 0:
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num of GPUs = {self.opt['world_size']}")
            logger.info(f"  Num Epochs = {self.opt['SOLVER']['MAX_NUM_EPOCHS']}")
            logger.info(f"  Num of Mini Batches per Epoch = {self.train_params['updates_per_epoch']}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch']}")
            logger.info(f"  Gradient Accumulation steps = {self.grad_acc_steps}")
            logger.info(f"  Total optimization steps = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch'] // self.grad_acc_steps}")

    def train(self):
        """
        Training
        """
        self.init_train()

        current_optim_steps = self._get_and_validate_current_optim_steps()
        num_epochs = self.opt['SOLVER']['MAX_NUM_EPOCHS']

        train_prev_logged_time = datetime.now()
        
        best_perf = -1.0
        best_model = False
        best_prev_tags = "_best"
        last_prev_tags = "_last"

        for epoch in range(self.train_params['start_epoch_idx'], num_epochs):
            self.train_params['current_epoch_idx'] = epoch
            logger.info(f"Start epoch: {epoch} training.")

            epoch_start_time = datetime.now()
            for batch_idx, batch in enumerate(self.train_dataloaders):
                if self.train_params['current_epoch_idx'] == self.train_params['start_epoch_idx']:
                    if batch_idx < self.train_params['start_batch_idx']: # skip the first few batches for resuming
                        continue

                self.train_params['current_batch_idx'] = batch_idx
                # prev_optim_steps = current_optim_steps
                prev_total_batch_size = self.train_params['total_batch_size']

                # update
                self.train_step(batch)

                # for i, g in enumerate(self.lr_schedulers['default'].optimizer.param_groups):
                #     is_close = math.isclose(g['initial_lr'], 1e-05, rel_tol=1e-9, abs_tol=1e-12)
                #     if is_close:
                #         g["lr"] = 1e-05

                # current_optim_steps = self._get_and_validate_current_optim_steps()

                # logging
                # if prev_optim_steps != current_optim_steps:  # an optimizer update was made
                log_first = self.opt.get("LOG_FIRST", 10)
                log_every = self.opt.get("LOG_EVERY", 100)
                if ((batch_idx + 1) % log_every == 0) or (epoch == 0 and (batch_idx + 1) <= log_first): # print logging
                    last_lr = {}
                    for module_name in self.model_names:
                        last_lr[module_name] = self.lr_schedulers[module_name].optimizer.param_groups[0]['lr']

                    train_time_delta = (datetime.now() - train_prev_logged_time).total_seconds()
                    train_prev_logged_time = datetime.now()
                    MB = 1024.0 * 1024.0
                    memory = torch.cuda.max_memory_allocated() / MB

                    if self.opt['rank'] == 0:
                        if self.wdb:
                            self.wdb.log({"learning rate": list(last_lr.values())[0]})
                            for key, obj in self.train_loss.losses.items():
                                average_name = key + "_avg"
                                self.wdb.log({average_name: obj.avg})
                                
                        logger.info(f"epochs[{epoch:6}] optim steps[{batch_idx+1:.0f}/{self.dataset_length}] "
                            f"learning rate[{', '.join([f'{key}: {val:.5e}' for key, val in last_lr.items()])}] "
                            f"train loss[{', '.join([f'{key}: {obj.val:.5f}/{obj.avg:.5f}' for key, obj in self.train_loss.losses.items()])}] "
                            # f"total_loss[{total_loss:.5f}/{total_loss_avg:.5f} "
                            f"items per batch[{self.train_params['total_batch_size'] - prev_total_batch_size}] "
                            f"items per second[{(self.train_params['total_batch_size'] - prev_total_batch_size) / train_time_delta:.2f}] "
                            f"total items[{self.train_params['total_batch_size']}] "
                            f"mini batches[{self.train_params['num_updates']:6}] "
                            f"memory[{memory:.0f}] "
                            f"epoch remaining[{str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (self.train_params['updates_per_epoch'] - batch_idx - 1)).split('.')[0]}]")

                # evaluate and save ckpt every epoch
                if batch_idx + 1 == self.train_params['updates_per_epoch']:
                    results = self._eval_on_set(self.save_folder)

                    if (epoch % 10) == 0:
                        str_last_10_epoch = "_" + str(epoch) + "_epoch"
                        self.save_checkpoint(self.train_params['num_updates'], extra=str_last_10_epoch)
                    
        
                    if self.opt['rank'] == 0:
                        save_dir = os.path.join(self.save_folder, last_prev_tags)
                        if os.path.isdir(save_dir):
                            shutil.rmtree(save_dir)
                            logger.info(f"removed previous save directory..")
                    self.save_checkpoint(self.train_params['num_updates'], extra="_last")
                    last_prev_tags = str(self.train_params['num_updates']).zfill(8) + "_last"
                    
                    best_mAP_all = results['vcoco_val/vcoco']['mAP_all']
                    if self.wdb:
                        self.wdb.log({"mAP_all": best_mAP_all})
                        self.wdb.log({"mAP_thesis": results['vcoco_val/vcoco']['mAP_thesis']})
                    
                    if best_mAP_all > best_perf:
                        best_perf = best_mAP_all
                        best_model = True
                    else:
                        best_model = False

                    if self.opt['rank'] == 0:
                        if best_model:
                            save_dir = os.path.join(self.save_folder, best_prev_tags)
                            save_dir = os.path.abspath(save_dir)
                            if os.path.isdir(save_dir):
                                shutil.rmtree(save_dir)
                                logger.info(f"removed previous save directory..")
                            self.save_checkpoint(self.train_params['num_updates'], extra="_best")
                            best_prev_tags = str(self.train_params['num_updates']).zfill(8) + "_best"
                    break

            self.lr_schedulers['default'].step()
            logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
            logger.info(f"PROGRESS: {100.0 * (epoch + 1) / num_epochs:.2f}%")
            logger.info(f"Config files are at {self.opt['conf_files']}")