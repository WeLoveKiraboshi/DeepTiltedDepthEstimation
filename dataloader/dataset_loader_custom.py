import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os
import fnmatch


class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.to_tensor = transforms.ToTensor()
        self.root = dataset_path
        self.idx = fnmatch.filter(os.listdir(self.root), '*.jpg')
        self.data_len = len(self.idx)

    def __getitem__(self, index):
        image_name = self.idx[index]
        rgb_info = os.path.join(self.root, image_name)
        rgb_img = sio.imread(rgb_info)
        rgb_img = cv2.resize(rgb_img, (320, 240), interpolation=cv2.INTER_CUBIC)
        rgb_tensor = self.to_tensor(rgb_img)

        dummy_mask = torch.Tensor(np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype='float32'))
        dummy_depth = self.to_tensor(np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype='float32'))

        # gravity_array = np.loadtxt(rgb_info.replace('jpg', 'txt'), dtype=np.float32)
        # gravity_array[2] = -gravity_array[2]
        # gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        # gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)
        dummy_gravity = torch.tensor([0., 1., 0.], dtype=torch.float)
        aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)

        rgb_mask_tensor = np.where(rgb_img == 0, 0, 1.0).astype(
            np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)
        
        data_split = 'e2'

        scene_idx = 0 #int( rgb_info[31:-4])

        return {'image': rgb_tensor, 'mask': dummy_mask, 'depth': dummy_depth, 'rgb_mask': rgb_mask_tensors,
                 'gravity': dummy_gravity, 'aligned_directions': aligned_directions,'scene': scene_idx, 'ga_split': data_split} #'ga_split': None
        #return {'image': rgb_tensor, 'aligned_directions': aligned_directions}

    def __len__(self):
        return self.data_len



class TUMDataset(Dataset):
    def __init__(self, dataset_path, associate_path, dataset='TUMrgbd_frei1rpy', depth_mask_RGB=False, cfg=None):
        self.to_tensor = transforms.ToTensor()
        self.root = dataset_path
        f = open(associate_path)
        line = f.readline()
        self.data_info = [[], [], []] #0;rgb 1;depth 2;pose
        while line:
            rgbpath = line.strip().split()[1]
            depthpath = line.strip().split()[3]
            self.data_info[0].append(rgbpath)
            self.data_info[1].append(depthpath)
            line = f.readline()
        f.close()
        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.dataset = dataset
        self.depth_mask_RGB = depth_mask_RGB
        self.cfg = cfg

        if self.dataset == 'TUMrgbd_frei1rpy' or self.dataset == 'TUMrgbd_frei2rpy' or self.dataset == 'TUMrgbd_frei3rpy':
            #uint max = 65535
            # scalefactor=5000 = 1 meter, 0.8 ~ 4.0m = 4000 ~ 20000
            self.maxdepth = 5000
        elif 'OurDataset' in self.dataset:
            # scalefactor=1000 = 1 meter,
            self.maxdepth = 1000



    def __getitem__(self, index):
        color_info_ = self.data_info[0][self.idx[index]]
        depth_info_ = self.data_info[1][self.idx[index]]
        color_info = os.path.join(self.root, color_info_)
        depth_info = os.path.join(self.root, depth_info_)

        #if self.dataset == 'OurDataset':
        gravity_info_ = color_info_.replace('rgb', 'gravity-dir').replace('png', 'txt')
        gravity_info = os.path.join(self.root, gravity_info_)


        if self.cfg.vps:
            vps_info_ = color_info_.replace('rgb', 'vps').replace('png', 'npy')
            vps_info = os.path.join(self.root, vps_info_)
            vps_array = np.load(vps_info)
            vps_tensor = self.to_tensor(vps_array).squeeze(0)

        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
        depth_img = cv2.resize(
            cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32) / self.maxdepth,
            (320, 240), interpolation=cv2.INTER_NEAREST)
        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0)  # this means valid pixel for depth

        aligned_directions = F.normalize(torch.tensor([0., 1., 0.], dtype=torch.float),dim=-1, p=2) #-0.73275027  0.67734738 -0.06540313, 5.34249721e-17, 8.72496007e-01, 4.88621241e-01
        #if self.dataset != 'TUMrgbd_frei2rpy' and self.dataset != 'TUMrgbd_frei3rpy':
        gravity_array = np.loadtxt(gravity_info, dtype=np.float32)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        depth_mask_tensor = torch.Tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)

        rgb_mask_tensor = np.where(color_img == 0, 0, 1.0).astype(
            np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)


        if self.dataset == 'TUMrgbd_frei1rpy' or self.dataset == 'TUMrgbd_frei2rpy':
            scene_idx = color_info[58:] #color_info[58:75]
            timestamp = color_info.split('/')[6][:-4]
            print(timestamp)
        elif self.dataset == 'TUMrgbd_frei3rpy':
            scene_idx = color_info[66:] # color_info[66:76]
            timestamp = color_info.split('/')[5][:-4]
        elif 'OurDataset' in self.dataset:
            scene_idx = int(self.root[-2:-1])
            timestamp = color_info.split('/')[8][:-4]
        data_split = 'e2'

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        if self.depth_mask_RGB:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0:1, :, :] = depth_mask_tensor
        else:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0, :, :] = depth_mask_tensor

        if self.cfg.vps:
            return {'image': input_tensor, 'rgb_mask': rgb_mask_tensors,'mask': depth_mask_tensors, 'depth': depth_tensor,
                'gravity': aligned_directions, 'aligned_directions': aligned_directions, 'scene': scene_idx,'ga_split': data_split, 'vps': vps_tensor, 'timestamp':timestamp}
        else:
            return {'image': input_tensor, 'rgb_mask': rgb_mask_tensors,'mask': depth_mask_tensors, 'depth': depth_tensor,
                    'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,
                    'ga_split': data_split, 'timestamp':timestamp}

    def __len__(self):
        return self.data_len



class OurFullDataset(Dataset):
    def __init__(self, train_test_split = None, depth_mask_RGB=False, cfg=None, mode='test'):
        self.to_tensor = transforms.ToTensor()
        self.data_info = pickle.load(open(train_test_split, 'rb'))[mode]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.cfg = cfg
        # scalefactor=1000 = 1 meter,
        self.maxdepth = 1000
        self.depth_mask_RGB = depth_mask_RGB


    def __getitem__(self, index):
        color_info = self.data_info[0][self.idx[index]]
        depth_info = self.data_info[1][self.idx[index]]
        gravity_info = self.data_info[2][self.idx[index]]

        if self.cfg.vps:
            vps_info_ = color_info_.replace('rgb', 'vps').replace('png', 'npy')
            vps_info = os.path.join(self.root, vps_info_)
            vps_array = np.load(vps_info)
            vps_tensor = self.to_tensor(vps_array).squeeze(0)

        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
        depth_img = cv2.resize(
            cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32) / self.maxdepth,
            (320, 240), interpolation=cv2.INTER_NEAREST)
        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0)  # this means valid pixel for depth

        aligned_directions = F.normalize(torch.tensor([0., 1., 0.], dtype=torch.float),dim=-1, p=2) #-0.73275027  0.67734738 -0.06540313, 5.34249721e-17, 8.72496007e-01, 4.88621241e-01
        gravity_array = np.loadtxt(gravity_info, dtype=np.float32)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        depth_mask_tensor = torch.Tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)

        scene_idx = 0
        data_split = 'e2'

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        if self.depth_mask_RGB:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0:1, :, :] = depth_mask_tensor
        else:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0, :, :] = depth_mask_tensor

        rgb_mask_tensor = np.where(color_img == 0, 0, 1.0).astype(np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)

        if self.cfg.vps:
            return {'image': input_tensor, 'rgb_mask': rgb_mask_tensors, 'mask': depth_mask_tensors, 'depth': depth_tensor,
                'gravity': aligned_directions, 'aligned_directions': aligned_directions, 'scene': scene_idx,'ga_split': data_split, 'vps': vps_tensor}
        else:
            return {'image': input_tensor, 'rgb_mask': rgb_mask_tensors, 'mask': depth_mask_tensors, 'depth': depth_tensor,
                    'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,
                    'ga_split': data_split}

    def __len__(self):
        return self.data_len



##for demo use only
class OurFullDataset_train_val(Dataset):
    def __init__(self, train_test_split = None, depth_mask_RGB=False, cfg=None, mode='test'):
        self.to_tensor = transforms.ToTensor()
        self.data_info = pickle.load(open(train_test_split, 'rb'))['test']

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.cfg = cfg
        # scalefactor=1000 = 1 meter,
        self.maxdepth = 1000
        self.depth_mask_RGB = depth_mask_RGB
        self.depth_scale_mode = 'tanh'
        self.mode = mode
        self.MIN_DEPTH_CLIP = 1.0  # meter scale
        self.MAX_DEPTH_CLIP = 10.0  # meter scale


    def __getitem__(self, index):
        color_info = self.data_info[0][self.idx[index]]
        depth_info = self.data_info[1][self.idx[index]]
        gravity_info = self.data_info[2][self.idx[index]]

        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
        depth_img = cv2.resize(
            cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32) / self.maxdepth,
            (320, 240), interpolation=cv2.INTER_NEAREST)
        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0)  # this means valid pixel for depth

        aligned_directions = F.normalize(torch.tensor([0., 1., 0.], dtype=torch.float),dim=-1, p=2) #-0.73275027  0.67734738 -0.06540313, 5.34249721e-17, 8.72496007e-01, 4.88621241e-01
        gravity_array = np.loadtxt(gravity_info, dtype=np.float32)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        depth_mask_tensor = torch.Tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)
        depth_tensor = self.preprocess_depth(depth_tensor, mode=self.depth_scale_mode)

        scene_idx = 0
        data_split = 'e2'

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        if self.depth_mask_RGB:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0:1, :, :] = depth_mask_tensor
        else:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0, :, :] = depth_mask_tensor

        rgb_mask_tensor = np.where(color_img == 0, 0, 1.0).astype(np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)

        return {'image': input_tensor, 'rgb_mask': rgb_mask_tensors, 'mask': depth_mask_tensors, 'depth': depth_tensor,
                'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,
                'ga_split': data_split}

    def preprocess_depth(self, depthT, mode='tanh'):
        '''
            preprocess depth tensor before feed into the network
            mode: choose from depth [0, max_depth], disparity [0, 1], tanh [-1.0, 1.0]
        '''
        if self.mode != 'test':
            if mode == 'tanh':
                return (((depthT - self.MIN_DEPTH_CLIP) / (
                        self.MAX_DEPTH_CLIP - self.MIN_DEPTH_CLIP)) - 0.5) * 2.0  # mask out depth over
            elif mode == 'depth':
                return depthT  # just meter scale
        else:
            return depthT


    def __len__(self):
        return self.data_len
