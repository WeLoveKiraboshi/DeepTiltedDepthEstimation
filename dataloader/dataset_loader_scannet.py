import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os


class ScannetDataset(Dataset):
    def __init__(self, root='/media/yukisaito/ssd2/ScanNetv2/scans',
                       usage='test',
                       train_test_split='./data/scannet_standard_train_test_val_split.pkl',depth_mask_RGB=False, depth_scale_mode='tanh', pad_mode='zerps'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()

        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]

        #to test quickly, use this split
        # if usage == 'test' or usage=='val':
        #     self.idx = [i for i in range(0, len(self.data_info[0]), 200)]

        self.data_len = len(self.idx)
        self.root = root
        self.maxdepth = 1000.0 # get meter scale
        self.init_grav = np.array([0.0, 0.0, -1.0]).reshape(3,1)
        self.counter = 0
        self.depth_mask_RGB = depth_mask_RGB
        self.MIN_DEPTH_CLIP = 1.0 #meter scale
        self.MAX_DEPTH_CLIP = 10.0 #meter scale
        self.depth_scale_mode = depth_scale_mode
        self.mode = usage
        self.pad_mode = pad_mode

    def __getitem__(self, index):
        if self.train_test_plit == './data/framenet_train_test_split.pkl': # get proper path from framenet pkl
            pass
        else:
            color_info_ = self.data_info[0][self.idx[index]]
            depth_info_ = self.data_info[1][self.idx[index]]
            pose_info_ = self.data_info[2][self.idx[index]]
            color_info = os.path.join(self.root, color_info_)
            depth_info = os.path.join(self.root, depth_info_)
            gravity_info_ = pose_info_.replace('pose', 'gravity-dir')
            gravity_info = os.path.join(self.root, gravity_info_)

        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_LINEAR)
        if self.pad_mode == 'border':
            border_size = 0
            color_img = cv2.resize(color_img[border_size:240-border_size, border_size:320-border_size], (320, 240), interpolation=cv2.INTER_LINEAR)


        depth_img = cv2.resize(cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32)/ self.maxdepth, (320, 240), interpolation=cv2.INTER_NEAREST)#
        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0).astype(np.float32) #.reshape(240, 320, 1)  # this means valid pixel for depth

        gravity_array = np.loadtxt(gravity_info, dtype=np.float32)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor[2] = -gravity_tensor[2]
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        data_split = 'with_ga'

        aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)

        # To tensor
        color_tensor = self.to_tensor(color_img)

        depth_mask_tensors = self.to_tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)
        if self.mode != 'test':
            depth_tensor = self.preprocess_depth(depth_tensor, mode=self.depth_scale_mode)

        scene_idx = int(color_info[43:47])

        #if self.depth_mask_RGB:
        rgb_mask_tensor = np.where(color_img == 0, 0, 1.0).astype(np.float32)  # .reshape(240, 320, 1)  # this means valid pixel for depth
        rgb_mask_tensors = self.to_tensor(rgb_mask_tensor)
        #else:

        return {'image': color_tensor, 'rgb_mask': rgb_mask_tensors, 'mask': depth_mask_tensors, 'depth': depth_tensor,
                'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,'ga_split': data_split}

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
                return depthT #just meter scale
        else:
            return depthT

    def __len__(self):
        return self.data_len


class Rectified2DOF(Dataset):
    def __init__(self, root='/media/yukisaito/ssd2/ScanNetv2/',
                       usage='test',
                       train_test_split='./data/rectified_2dofa_scannet.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split

        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 1)]
        self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 1)]

        if usage == 'test':
            self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 50)]
            self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 50)]

        self.data_len = max((len(self.idx_e2), len(self.idx_me2)))
        print('idx_e2: %d, idx_me2: %d' % (len(self.idx_e2), len(self.idx_me2)))

        self.root = root

    def __getitem__(self, index):
        if np.random.ranf() < 2./3:
            data_idx = self.idx_e2[index % len(self.idx_e2)]
            data_split = 'e2'
        else:
            data_idx = self.idx_me2[index % len(self.idx_me2)]
            data_split = '-e2'



        color_info = os.path.join(self.root, self.data_info[data_split][data_idx])
        mask_info = color_info.replace('color', 'orient-mask')
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        mask_valid_size = np.sum((orient_mask_tensor > 0))

        while mask_valid_size < 3e4:
            data_split = 'e2'
            index = np.random.randint(0, len(self.idx_e2))
            data_idx = self.idx_e2[index % len(self.idx_e2)]
            color_info = os.path.join(self.root, self.data_info[data_split][data_idx])
            mask_info = color_info.replace('color', 'orient-mask')
            orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
            mask_valid_size = np.sum((orient_mask_tensor > 0))

        orient_info = color_info.replace('color', 'normal')
        gravity_info = color_info.replace('color.png', 'gravity.txt')
        gravity_info = gravity_info.replace('scannet-frames', 'scannet-small-frames')

        if data_split == 'e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
            gravity_tensor[2] = -gravity_tensor[2]
        elif data_split == '-e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            gravity_tensor = -torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
            gravity_tensor[2] = -gravity_tensor[2]
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_CUBIC)

        # To tensor
        if self.depth_mask_RGB:
            depth_mask_tensors = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0:3, :, :] = depth_mask_tensor
        else:
            depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0, :, :] = depth_mask_tensor

        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
                'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'ga_split': data_split}

    def __len__(self):
        return self.data_len


class Full2DOF(Dataset):
    def __init__(self, root='/media/yukisaito/ssd2/ScanNetv2/scans',
                       usage='test',
                       train_test_split='./data/my_full_2dofa_scannet.pkl', depth_mask_RGB=False):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split

        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]
        self.data_with_ga = self.data_info['with_ga']
        self.data_with_ga_e2 = self.data_with_ga['e2']
        self.data_with_ga_me2 = self.data_with_ga['-e2']
        self.data_no_ga = self.data_info['no_ga']

        # if usage == 'test':
        #     self.data_with_ga_e2 = [self.data_with_ga['e2'][i] for i in range(0, len(self.data_with_ga['e2']), 50)]
        #     self.data_with_ga_me2 = [self.data_with_ga['-e2'][i] for i in range(0, len(self.data_with_ga['-e2']), 50)]
        #     self.data_no_ga = [self.data_info['no_ga'][i] for i in range(0, len(self.data_info['no_ga']), 50)]

        self.ga_part_len = len(self.data_with_ga_e2[0]) + len(self.data_with_ga_me2[0])
        self.data_len = len(self.data_with_ga_e2[0]) + len(self.data_with_ga_me2[0]) + len(self.data_no_ga[0])

        # index
        self.idx_with_ga_e2 = [i for i in range(0, len(self.data_with_ga_e2[0]), 1)]
        self.idx_with_ga_me2 = [i for i in range(0, len(self.data_with_ga_me2[0]), 1)]
        self.idx_no_ga = [i for i in range(0, len(self.data_no_ga[0]), 1)]

        self.root = root
        self.maxdepth = 1000.0
        self.init_grav = np.array([0.0, 0.0, -1.0]).reshape(3, 1)
        self.counter = 0
        self.depth_mask_RGB = depth_mask_RGB



    def __getitem__(self, index):
        if np.random.ranf() < self.ga_part_len/float(self.data_len):
            # draw from ga_part (e2 or -e2)
            if np.random.ranf() < 2. / 3:
                data_idx = self.idx_with_ga_e2[index % len(self.idx_with_ga_e2)]
                data_split = 'e2'
            else:
                data_idx = self.idx_with_ga_me2[index % len(self.idx_with_ga_me2)]
                data_split = '-e2'

            color_info = os.path.join(self.root, self.data_with_ga[data_split][0][data_idx])
            depth_info = os.path.join(self.root, self.data_with_ga[data_split][1][data_idx])
            pose_info = os.path.join(self.root, self.data_with_ga[data_split][2][data_idx])
            gravity_info = pose_info.replace('pose', 'gravity-dir')
        else:
            # draw from no_ga_part
            data_idx = self.idx_no_ga[index % len(self.idx_no_ga)]
            data_split = 'no_ga'
            # print(self.data_no_ga[0])
            # print(len(self.data_no_ga))
            # print(len(self.data_no_ga[0]))
            #print(len(self.data_no_ga[1]))
            color_info = os.path.join(self.root, self.data_no_ga[0][data_idx])
            depth_info = os.path.join(self.root, self.data_no_ga[1][data_idx])
            pose_info = os.path.join(self.root, self.data_no_ga[2][data_idx])
            gravity_info = pose_info.replace('pose', 'gravity-dir')

        # Image resize and load
        color_img = cv2.resize(cv2.imread(color_info, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
        depth_img = cv2.resize(cv2.imread(depth_info, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float32) / self.maxdepth,
            (320, 240), interpolation=cv2.INTER_NEAREST)
        depth_mask_tensor = np.where(depth_img == 0, 0, 1.0)  # this means valid pixel for depth
        aligned_directions, gravity_tensor = None, None
        gravity_array = np.loadtxt(gravity_info, dtype=np.float32) #[:3, :3]
        #gravity_array = np.linalg.inv(pose_array) @ self.init_grav
        #gravity_array = gravity_array / np.linalg.norm(gravity_array)
        gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
        gravity_tensor[2] = -gravity_tensor[2]
        gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)

        if data_split == 'e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            #gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        elif data_split == '-e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            #gravity_tensor = -torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        else:
            # dummy values
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            #gravity_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)

        # To tensor
        depth_mask_tensor = torch.Tensor(depth_mask_tensor)
        depth_tensor = self.to_tensor(depth_img)

        if self.depth_mask_RGB:
            depth_mask_tensors = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
            depth_mask_tensors[0:3, :, :] = depth_mask_tensor

        depth_mask_tensors = np.zeros((1, color_img.shape[0], color_img.shape[1]), dtype='float32')
        depth_mask_tensors[0, :, :] = depth_mask_tensor

        scene_idx = int(color_info[43:47])

        color_tensor = self.to_tensor(color_img)
        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': depth_mask_tensors, 'depth': depth_tensor,
                'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'scene': scene_idx,
                'ga_split': data_split }


    def __len__(self):
        return self.data_len
