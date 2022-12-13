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




class TitanAirDataset(Dataset):
    def __init__(self, dataset_path, associate_path, dataset='titan', depth_mask_RGB=False, cfg=None):
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

        self.maxdepth = 1000



    def __getitem__(self, index):
        color_info_ = self.data_info[0][self.idx[index]]
        depth_info_ = self.data_info[1][self.idx[index]]
        color_info = os.path.join(self.root, color_info_)
        depth_info = os.path.join(self.root, depth_info_)

        #if self.dataset == 'OurDataset':
        gravity_info_ = color_info_.replace('image', 'gravity-dir').replace('png', 'txt')
        gravity_info = os.path.join(self.root, gravity_info_)


        if self.cfg.vps:
            vps_info_ = color_info_.replace('image', 'vps').replace('png', 'npy')
            vps_info = os.path.join(self.root, vps_info_)
            vps_array = np.load(vps_info)
            vps_tensor = self.to_tensor(vps_array).squeeze(0)

        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
        depth_img = cv2.resize(
            cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32) / self.maxdepth,
            (320, 240), interpolation=cv2.INTER_NEAREST)

        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0)  # this means valid pixel for depth
        edge_width = 3
        depth_mask_tensor[:, 0:edge_width] = 0
        depth_mask_tensor[:, 320-edge_width:320] = 0
        depth_mask_tensor[0:edge_width, :] = 0
        depth_mask_tensor[240-edge_width:240, :] = 0
        # cv2.imshow('test', np.uint8(depth_mask_tensor[:, :, np.newaxis]*color_img))
        # cv2.waitKey(0)



        aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)

        gravity_array = np.loadtxt(gravity_info, dtype=np.float32)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)
        #gravity_tensor = torch.tensor([0., 1., 0.], dtype=torch.float) #F.normalize(gravity_tensor, dim=-1, p=2)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        depth_mask_tensor = torch.Tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)

        scene_idx = self.idx[index]
        data_split = 'e2'

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        #if self.depth_mask_RGB:
        #    depth_mask_tensors = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        #    depth_mask_tensors[0:3, :, :] = depth_mask_tensor
        #else:
        depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
        depth_mask_tensors[0, :, :] = depth_mask_tensor
            
        rgb_mask_tensor = np.where(color_img == 0, 0, 1.0).astype(np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)

        if self.cfg.vps:
            return {'image': input_tensor, 'mask': depth_mask_tensors, 'depth': depth_tensor, 'rgb_mask': rgb_mask_tensors,
                'gravity': aligned_directions, 'aligned_directions': aligned_directions, 'scene': scene_idx,'ga_split': data_split, 'vps': vps_tensor}
        else:
            return {'image': input_tensor, 'mask': depth_mask_tensors, 'depth': depth_tensor, 'rgb_mask': rgb_mask_tensors,
                    'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,
                    'ga_split': data_split}



    def __len__(self):
        return self.data_len

