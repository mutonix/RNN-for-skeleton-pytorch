
import numpy as np
import random
import math
import h5py
from torch.utils import data
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

class Construct_raw_dataset(object):
    def __init__(self, param, dim_point=3, num_joints=25, num_class=60):
        self._param = param
        self._dim_point = dim_point
        self._num_joints = num_joints
        self._num_class = num_class

    # 仿射变换
    def rand_view_transform(self, X, angle1=-10, angle2=10, s1=0.9, s2=1.1):
        agx = random.randint(angle1, angle2)
        agy = random.randint(angle1, angle2)
        s = random.uniform(s1, s2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X = np.dot(X, np.dot(Ry,np.dot(Rx,Ss)))
        X = X.astype(np.float32)
        return X

    #若subtract_mean中scale为True，则skeleton归一化，需计算中心点
    def calculate_height(self, skeleton):
        # 胸部中心 3 9 5 21
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        w1 = skeleton[:,23,:] - center1 # 右手-胸部中心
        w2 = skeleton[:,22,:] - center1 # 左手-胸部中心
        # 髋部中心 2 1 17 13
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        h0 = skeleton[:,3,:] - center2  # 头-髋部中心
        h1 = skeleton[:,19,:] - center2 # 右脚 - 髋部中心
        h2 = skeleton[:,15,:] - center2 # 左脚 - 髋部中心
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])  # 人的宽度的一半
        heigh = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1])), np.max(h0[:,1])]) # 人的高度的一半
        return width, heigh

    # skeleton归一化
    def subtract_mean(self, skeleton, smooth=False, scale=True):
        if smooth:
            skeleton = self.smooth_skeleton(skeleton)
        # substract mean values
        # notice: use two different mean values to normalize skeleton data
        if 0:
            center = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
            # center = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
            for idx in range(skeleton.shape[1]):
                skeleton[:, idx] = skeleton[:, idx] - center

        # 胸部中心 3 9 5 21
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4

        # 髋部中心 2 1 17 13
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        
        # 对两个手臂使用胸部归一化
        for idx in [24, 25, 12, 11, 10, 9, 5, 6, 7, 8, 23, 22]:
            skeleton[:, idx-1] = skeleton[:, idx-1] - center1

        # 对身体其他部分使用髋部归一化
        for idx in (set(range(1, 1+skeleton.shape[1]))-set([24, 25, 12, 11, 10, 9, 5, 6, 7, 8, 23, 22])):
            skeleton[:, idx-1] = skeleton[:, idx-1] - center2

        # 尺度归一化
        if scale:
            width, heigh = self.calculate_height(skeleton)
            scale_factor1, scale_factor2 = 0.36026082, 0.61363413
            skeleton[:,:,0] = scale_factor1*skeleton[:,:,0]/width
            skeleton[:,:,1] = scale_factor2*skeleton[:,:,1]/heigh
        return skeleton

    # 加载训练和测试数据
    # step=1, train:start_zero=True , val:start_zero=False
    def load_sample_step_list(self, h5_file, list_file, num_seq, data_type, step=1, start_zero=True, sub_mean=False, scale=False):
        # 读取file_list_train.txt
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        # label获取使用文件名Aaaa中aaa
        label_list = [(int(name[17:20])-1) for name in name_list]
        X = []
        label = []
        with h5py.File(h5_file,'r') as hf:
            name_list_iter = tqdm(name_list, desc=f"Constructing {data_type} dataset...")
            for idx, name in enumerate(name_list_iter):
                # 获取name的skeleton
                skeleton = np.asarray(hf.get(name))
                if sub_mean:
                    skeleton = self.subtract_mean(skeleton, scale=scale)
                for start in range(0, 1 if start_zero else step): # range(0, 1) step = 1
                    # 获取该人的所有skeleton
                    skt = skeleton[start:skeleton.shape[0]:step]
                    if skt.shape[0] > num_seq:
                        # process sequences longer than num_seq, sample two sequences, if start_zero=True, only sample once from 0
                        # start_zero True -> 只取前面num_seq个骨架
                        for sidx in ([np.arange(num_seq)] if start_zero else [np.arange(num_seq), np.arange(skt.shape[0]-num_seq, skt.shape[0])]):
                            X.append(skt[sidx])
                            label.append(label_list[idx])
                    else:
                        if skt.shape[0] < 0.05*num_seq: # skip very small sequences
                            continue
                        # 前向填充
                        skt = np.concatenate((np.zeros((num_seq-skt.shape[0], skt.shape[1], skt.shape[2])), skt), axis=0)
                        X.append(skt)
                        label.append(label_list[idx])

        X = self.rand_view_transform(X) # view transformation

        X = torch.Tensor(X)   # -> (file_persons, num_seq, joints, xyz)
        label = torch.LongTensor(label)    # -> (file_persons,)

        return X, label

    def make(self):
        
        trainX, trainY = self.load_sample_step_list(
            self._param['trn_arr_file'],
            self._param['trn_lst_file'],
            self._param['num_seq'],
            data_type="train", 
            step=self._param['step'],
            start_zero=False,
            scale=self._param['scale'],
            sub_mean=self._param['sub_mean'])  

        valX, valY = self.load_sample_step_list(self._param['tst_arr_file'], self._param['tst_lst_file'], self._param['num_seq'],
                data_type="test", step=self._param['step'], start_zero=True, scale=self._param['scale'],
                sub_mean=self._param['sub_mean'])
        return trainX, trainY, valX, valY

def construct_raw_dataset() -> tuple: 
    param = {}
    param['trn_arr_file'] = './data/subj_seq2/array_list_train.h5'
    param['trn_lst_file'] = './data/subj_seq2/file_list_train.txt'
    param['tst_arr_file'] = './data/subj_seq2/array_list_test.h5'
    param['tst_lst_file'] = './data/subj_seq2/file_list_test.txt'
    param['num_seq'] = 100
    param['step'] = 1
    param['sub_mean']=True
    param['scale']=True

    trainX, trainY, valX, valY = Construct_raw_dataset(param).make()

    return trainX, trainY, valX, valY

def get_dataloader(batch_size, eval_batch_size, device):
    raw_dataset = construct_raw_dataset()
    raw_dataset = [i.to(device) for i in raw_dataset]
    train_set, train_labels, test_set, test_labels = raw_dataset
    print("Dataset almost done...")

    # train_set = torch.randn((100, 100, 25, 3)).to(device)
    # train_labels = torch.randint(0, 60, (100,)).to(device)
    # test_set = torch.randn((32, 100, 25, 3)).to(device)
    # test_labels = torch.randint(0, 60, (32,)).to(device)

    train_ds = TensorDataset(train_set, train_labels)
    test_ds = TensorDataset(test_set, test_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=True, num_workers=0)
    print("************ Dataset loaded ************")

    return train_dl, test_dl
    
if __name__ == '__main__':
    pass