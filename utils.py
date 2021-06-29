#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Dongjie Yang, Zhe Li, Jiajie Wu

import torch
from einops import rearrange

traversal_seq_order = [20, 8, 9, 10, 11, 23, 24, 23, 11, 10, 9, 8,  # left arm
                       4, 5, 6, 7, 21, 22, 21, 7, 6, 5, 4,          # right arm
                       20, 2, 3, 2, 1, 0, 16, 17, 18, 19, 18, 17, 16, 0, 12, # legs
                       13, 14, 15, 14, 13, 12, 0, 1, 20             # trunk
                            ] # 47 joints
chain_seq_order = [23,24,11,10,9,8, 20, 4,5,6,7,21,22, # arms
                    3,2,20,1,0, # trunk
                    19,18,17,16,12,13,14,15 ] # legs
                    
temporal_rnn_order=[23, 24, 11, 10, 9, 8, 4, 5, 6, 7, 22, 21, 19, 18, 17, 16, 12, 13, 14, 15, 3, 2, 20, 1, 0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def data_reshape(X, type, num_joints=25, window_size=25, num_seq=100):
        """
        B -> Batchsize(B)
        A -> Axis(3)
        T -> num of frames(100)
        J -> num of joints(25)    
        """
        # X=[N,T,25,3]
        batch_size = X.shape[0]
        if type == 'temporal':
            X_new = rearrange(X[:, :, temporal_rnn_order, :], 'B T J A-> B T (J A)')
        elif type == 'chain':
            X_new = rearrange(X[:, :, chain_seq_order, :], 'B T J A -> B (T A) J')
            X_new = X_new.view((batch_size * int(num_seq / window_size), 3*window_size, len(chain_seq_order)))#N*4 T/4*3 25
            X_new = rearrange(X_new[:, :, :], 'B T J ->B J T')            
        elif type == 'traversal':
            X_new = torch.ones((X.shape[0], X.shape[1], len(traversal_seq_order), X.shape[3]))
            X_new = X_new.to(X.device)
            for idx, ord in enumerate(traversal_seq_order):
                X_new[:, :, idx] = X[:, :, ord]
            X_new = rearrange(X_new, 'B T J A  -> B (T A) J')
            X_new = X_new.view((batch_size * int(num_seq / window_size), 3*window_size, len(traversal_seq_order)))#N*4 T/4*3 47
            X_new = rearrange(X_new[:, :, :], 'B T J ->B J T')   
        return X_new
