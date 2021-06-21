import os
import numpy as np
import h5py
from tqdm import tqdm

class Load_ntu_rgbd(object):
    def __init__(self, data_path):
        self._data_path = data_path

    def skeleton_miss_list(self):
        lines = open('./NTU_RGBD_samples_with_missing_skeletons2.txt', 'r').readlines()
        return [line.strip()+'.skeleton' for line in lines]

    def filter_list(self, file_list):
        miss_list = self.skeleton_miss_list()
        return list(set(file_list)-set(miss_list))

    def cross_subject_split(self):
        print ('cross subject evaluation ...')
        trn_sub = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[9:12]) in trn_sub]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list

    def cross_view_split(self):
        print ('cross view evaluation ...')
        trn_view = [2, 3]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[5:8]) in trn_view]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list
    
    def person_position_std(self, filename, num_joints=25):
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        step = int(lines[0].strip())
        pid_set = []
        # idx_set length of sequence
        idx_set = []
        skeleton_set = []
        start = 1
        sidx = [0,1,2,7,8,9,10]
        while start < len(lines): # and idx < step
            if lines[start].strip()=='25':
                pid = lines[start-1].split()[0]
                if pid not in pid_set:
                    idx_set.append(0)
                    pid_set.append(pid)
                    skeleton_set.append(np.zeros((step, num_joints, 7)))
                idx2 = pid_set.index(pid)
                skeleton_set[idx2][idx_set[idx2]] = np.asarray([np.array(line_per.strip().split())[sidx].astype(np.float) \
                                            for line_per in lines[start+1:start+26]])
                idx_set[idx2] = idx_set[idx2] + 1
                start = start + 26
            else:
                start = start + 1
        std_set=[]
        for idx2 in range(len(idx_set)):
            skeleton_set[idx2] = skeleton_set[idx2][0:idx_set[idx2]]
            xm = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,0] - skeleton_set[idx2][0:idx_set[idx2]-1,:,0])
            xm = xm.sum(axis=-1)
            ym = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,1] - skeleton_set[idx2][0:idx_set[idx2]-1,:,1])
            ym = ym.sum(axis=-1)
            std_set.append((np.std(xm), np.std(ym)))
        return skeleton_set, pid_set, std_set
    
    def save_h5_file_skeleton_list(self, save_home, trn_list, split='train', angle=False):
        
        # save file list to txt
        save_name = os.path.join(save_home, 'file_list_' +  split + '.txt')
        with open(save_name, 'w') as fid_txt:  # fid.write(file+'\n')
            # save array list to hdf5
            save_name = os.path.join(save_home, 'array_list_' + split + '.h5')
            with h5py.File(save_name, 'w') as fid_h5:
                for fn in tqdm(trn_list):
                    skeleton_set, pid_set, std_set = self.person_position_std(fn)
                    # filter skeleton by standard value
                    count = 0
                    for idx2 in range(len(pid_set)):
                        if std_set[idx2][0] > 0.1 or std_set[idx2][1] > 0.1:
                            count = count + 1
                            name=fn+pid_set[idx2]
                            if angle:
                                fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                            else:
                                fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                            fid_txt.write(name + '\n')
                    if count == 0:
                        std_sum = [np.sum(it) for it in std_set]
                        idx2 = np.argmax(std_sum)
                        name=fn+pid_set[idx2]
                        if angle:
                            fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                        else:
                            fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                        fid_txt.write(name + '\n')

if __name__ == '__main__':
    data_path = 'D:/browserdownload/Firefox/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons2/'
    db = Load_ntu_rgbd(data_path)

    trn_list, tst_list = db.cross_subject_split()
    # trn_list, tst_list = db.cross_view_split()
    db.save_h5_file_skeleton_list('./subj_seq3', trn_list, split='train')
    db.save_h5_file_skeleton_list('./subj_seq3', tst_list, split='test')