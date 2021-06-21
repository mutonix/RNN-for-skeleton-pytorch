#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Yang Dongjie, Zhe Li, Jiajie Wu

from torch.optim import lr_scheduler, optimizer
from utils import AverageMeter
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn

from data import get_dataloader
from utils import AverageMeter
from model import Temporal_RNN, Two_Stream_RNN
# TODO:
# from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class RNN_Skeleton():
    def __init__(self, args) -> None:
        self.args = args
        
        self.task_name = args.task_name
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device

        self.lr_decay_step = args.lr_decay_step
        self.lr_decay_gamma = args.lr_decay_gamma
        self.eval_batch_size = args.eval_batch_size
        self.eval_period = args.eval_period
        self.log_save_dir = args.log_save_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.num_workers = args.num_workers
        self.resume = args.resume

        self.model = Two_Stream_RNN(model_type=args.temp_rnn_type, 
                                    seq_type=args.spatial_seq_type, 
                                    modified=args.modified)
        self.set_up()

    def get_lr_scheduler(self, optimizer, step_size=40, gamma=0.7, last_epoch=-1):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
        return scheduler

    def set_up(self):
        if not os.path.exists(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_dir = self.checkpoint_dir + '/' + self.task_name
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_file_path = self.log_save_dir+'/'+self.task_name+".txt"
        with open(self.log_file_path, 'w'):
            pass

    def train(self):
        
        train_dl, test_dl = get_dataloader(self.batch_size, self.eval_batch_size, self.device, self.num_workers)
        print('\n', self.args, '\n')

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr)
        if self.resume:
            self.model.load_state_dict(torch.load(self.resume))
            logger.info("Success loading the pre-trained weights!")

        # TODO: error with the last_epoch: wonder how to fix it to resume lr
        self.lr_scheduler = self.get_lr_scheduler(optimizer, self.lr_decay_step, self.lr_decay_gamma)

        loss_func = nn.CrossEntropyLoss()
        train_loss_meter = AverageMeter()
        
        self.model.to(self.device)
        self.model.train()
        logger.info("***** Start training *****")

        self.current_lr = self.lr
        self.best_acc = 0
        for epoch in range(self.epochs):
            train_batch_iter = tqdm(train_dl,
                        desc="Training (X / X epochs) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
            self.current_epoch = epoch
            for x, y in train_batch_iter:
                pred = self.model(x)
                loss = loss_func(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss_meter.update(loss.item())
                train_batch_iter.set_description(
                    "Training (%d / %d epochs) (loss=%2.5f)" % (epoch, self.epochs, train_loss_meter.val)
                )   

            self.lr_scheduler.step()           

            if (epoch + 1) % self.eval_period == 0 and epoch >= 0:
                self.model_eval(test_dl)
                if self.best_acc < self.accuracy:
                    self.best_acc = self.accuracy

            if (epoch + 1) >= self.eval_period:
                with open(self.log_file_path, 'a') as log_file:
                    log_file.write(f"epoch {self.current_epoch} acc {self.accuracy} best_acc {self.best_acc} loss {train_loss_meter.avg} lr {self.current_lr}\n")
        
        self.model_eval(test_dl)
        if self.best_acc < self.accuracy:
            self.best_acc = self.accuracy
        logger.info("Best Accuracy: \t%f" % self.best_acc)
        logger.info("End Training!")

    def model_eval(self, test_loader):

        self.eval_loss_meter = AverageMeter()

        logger.info("***** Running Validation *****")
        logger.info("  Batches of testset: %d", len(test_loader))
        logger.info("  Batch size: %d", self.eval_batch_size)

        self.model.eval()
        all_preds, all_labels = [], []
        test_iter = tqdm(test_loader,
                    desc="Validating... (loss=X.X)",
                    dynamic_ncols=True)

        eval_loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in test_iter:
                # (N, 100, 75)
                pred = self.model(x)
                eval_loss = eval_loss_func(pred, y)
                self.eval_loss_meter.update(eval_loss.item())

                preds = torch.argmax(pred, dim=-1) # -> (N,)
                all_preds.append(preds)
                all_labels.append(y)

                test_iter.set_description("Validating... (loss=%2.5f)" % self.eval_loss_meter.val)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)


            self.accuracy = (all_preds == all_labels).detach().cpu().numpy().mean()

        self.current_lr = self.lr_scheduler.get_last_lr()[0]

        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir)
        checkpoint_save_path = self.checkpoint_dir + '/' + \
                                self.task_name + "_ckpt_epoch%s.pth" % self.current_epoch
        torch.save(self.model.state_dict(), checkpoint_save_path)


        self.model.train()
        logger.info("\n")
        logger.info(f"Saving checkpoint to {checkpoint_save_path}")
        logger.info("Validation Results")
        logger.info("Current Learning Rate: %2.5f" % self.current_lr)
        logger.info("Valid Loss: %2.5f" % self.eval_loss_meter.avg)
        logger.info("Valid Accuracy: %2.5f" % self.accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default="train_task1", type=str)
    parser.add_argument("--temp_rnn_type", default="hierarchical", choices=["stacked", "hierarchical"],
                        help="RNN type for Temporal RNN")
    parser.add_argument("--spatial_seq_type", default="traversal", choices=["chain", "traversal"],
                        help="Sequence type for Spatial RNN")
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=64, type=int)   # for tesla v100 512
    parser.add_argument("--eval_batch_size", default=64, type=int) # for tesla v100 512
    parser.add_argument("--eval_period", default=10, type=int)
    parser.add_argument("--lr_decay_step", default=40, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.7, type=float)
    parser.add_argument("--log_save_dir", default="./log", type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoint", type=str)
    parser.add_argument("--num_workers", default=0, type=int) # num_workers for linux to load dataset, 0 for windows
    parser.add_argument("--resume", default="", type=str) # path of the last trained weight to resume
    parser.add_argument("--modified", action="store_true", 
                        help="whether to load the modified model")
    # parser.add_argument("--last_epoch", default=100, type=int) # resume last epoch not fixed

    args = parser.parse_args()
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    rnn_skeleton = RNN_Skeleton(args)
    rnn_skeleton.train()