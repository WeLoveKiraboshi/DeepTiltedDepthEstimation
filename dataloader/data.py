from torch.utils.data import DataLoader
from dataloader.dataset_loader_scannet import ScannetDataset
from dataloader.dataset_loader_custom import CustomDataset, TUMDataset, OurFullDataset,OurFullDataset_train_val
from dataloader.dataset_loader_nyu import NYUDataset
# from dataloader.dataset_loader_nyud import NYUD_Dataset
# from dataloader.dataset_loader_kinectazure import KinectAzureDataset
# from dataloader.dataset_loader_scannet import Rectified2DOF
from dataloader.dataset_loader_scannet import Full2DOF
import numpy as np
import torch


def create_dataset_loader(cfg):
    # Testing on NYUD
    ## network config
    if 'PartialConv' in cfg.network:
        depth_mask_RGB_cfg = True
    else:
        depth_mask_RGB_cfg = False

    if cfg.dataset == 'TUMrgbd_frei1rpy':
        dataset_dir = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg1_rpy/'
        associate_path = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg1_rpy/associate.txt'
        test_dataset = TUMDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                  depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
        return None, test_dataloader, None
    elif cfg.dataset == 'TUMrgbd_frei2rpy':
        dataset_dir = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg2_rpy/'
        associate_path = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg2_rpy/associate.txt'
        test_dataset = TUMDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                  depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
        return None, test_dataloader, None
    elif cfg.dataset == 'TUMrgbd_frei3rpy':
        dataset_dir = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg3_sitting_rpy/'
        associate_path = '/home/yukisaito/TUMdataset/rgbd_dataset_freiburg3_sitting_rpy/associate.txt'
        test_dataset = TUMDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                  depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
        return None, test_dataloader, None
    elif cfg.dataset == 'OurDataset_all':
        pkl_path = '/home/yukisaito/TiltedDepthEstimation/data/my_dataset_all_split'
        train_dataset = OurFullDataset_train_val(pkl_path, depth_mask_RGB_cfg, cfg, 'train')
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.bs,
                                      shuffle=True, num_workers=cfg.num_worker, pin_memory=True)
        return train_dataloader, None, None
    elif 'OurDataset_roll' in cfg.dataset: #OurDataset_roll_seq1
        if cfg.dataset == 'OurDataset_roll_full':
            pkl_path = '/home/yukisaito/TiltedDepthEstimation/data/my_dataset_roll_split'
            test_dataset = OurFullDataset(pkl_path, depth_mask_RGB_cfg, cfg, 'test')
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
            return test_dataloader, test_dataloader, None
        else:
            seq_idx = cfg.dataset[-4:]
            dataset_dir = '/home/yukisaito/mydataset/NFOV/roll/' + seq_idx + '/'
            associate_path = '/home/yukisaito/mydataset/NFOV/roll/' + seq_idx + '/associate.txt'
            test_dataset = TUMDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                      depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
            return None, test_dataloader, None
    elif 'OurDataset_pitch' in cfg.dataset:  # OurDataset_roll_seq1
        if cfg.dataset == 'OurDataset_pitch_full':
            pkl_path = '/home/yukisaito/TiltedDepthEstimation/data/my_dataset_pitch_split'
            test_dataset = OurFullDataset(pkl_path, depth_mask_RGB_cfg, cfg, 'test')
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
            return None, test_dataloader, None
        else:
            seq_idx = cfg.dataset[-4:]
            dataset_dir = '/home/yukisaito/mydataset/NFOV/pitch/'+ seq_idx + '/'
            associate_path = '/home/yukisaito/mydataset/NFOV/pitch/' + seq_idx + '/associate.txt'
            test_dataset = TUMDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                      depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
            return None, test_dataloader, None
    # ScanNet standard split
    elif cfg.dataset == 'scannet':
        if cfg.train_dataset_split == './data/my_scannet_standard_train_test_val_split.pkl':
            train_dataset = ScannetDataset(usage='train', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg, depth_scale_mode=cfg.scale_mode, pad_mode=cfg.image_padding_mode)
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.bs,
                                        shuffle=True, num_workers=cfg.num_worker, pin_memory=True)

            test_dataset = ScannetDataset(usage='test', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg, depth_scale_mode=cfg.scale_mode)
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs,
                                     shuffle=False, num_workers=cfg.num_worker)

            val_dataset = ScannetDataset(usage='val', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg, depth_scale_mode=cfg.scale_mode)
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.bs,
                                    shuffle=False, num_workers=cfg.num_worker)

            return train_dataloader, test_dataloader, val_dataloader

        elif cfg.train_dataset_split == './data/my_full_2dofa_scannet.pkl':
            train_dataset = Full2DOF(usage='train', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg)
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.bs,
                                      shuffle=True, num_workers=cfg.num_worker, pin_memory=True)

            test_dataset = Full2DOF(usage='test', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg)
            test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs,
                                     shuffle=False, num_workers=cfg.num_worker)

            val_dataset = Full2DOF(usage='test', train_test_split=cfg.train_dataset_split, depth_mask_RGB=depth_mask_RGB_cfg)
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.bs,
                                    shuffle=False, num_workers=cfg.num_worker)
            return train_dataloader, test_dataloader, val_dataloader
        else:
            print('train_test_data_split not found error...    {}'.format(cfg.train_dataset_split))
            exit(0)
    elif cfg.dataset == 'NYUv2':
        dataset_dir = '/home/yukisaito/NYUv2/nyuv2-python-toolkit-master/NYUv2/'
        associate_path = '/home/yukisaito/NYUv2/nyuv2-python-toolkit-master/NYUv2/associate_test.txt'
        test_dataset = NYUDataset(dataset_path=dataset_dir, associate_path=associate_path, dataset=cfg.dataset,
                                  depth_mask_RGB=depth_mask_RGB_cfg, cfg=cfg)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs, shuffle=False, num_workers=cfg.num_worker)
        return None, test_dataloader, None
    elif cfg.dataset == 'demo_dataset':
        dataset_path = './demo_dataset/demo_imgs/'
        test_dataset = CustomDataset(dataset_path)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.bs,
                                     shuffle=False, num_workers=cfg.num_worker)
        return None, test_dataloader, None
    else:
        print('Indicated Dataset doesn not exist... {}'.format(cfg.dataset))
        exit(0)





def data_augmentation(sample_batched, cfg, warper, epoch, iter, mode):
    if 'ga_split' in sample_batched:
        ga_split = sample_batched['ga_split']

    if cfg.input_augmentation == 'warp_input':
        gravity_dir = sample_batched['gravity']
        gravity_dir = gravity_dir.cuda()
        alignment_dir = sample_batched['aligned_directions']
        alignment_dir = alignment_dir.cuda()
        sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched,
                                                                     I_g=gravity_dir,
                                                                     I_a=alignment_dir,
                                                                     image_border=cfg.image_padding_mode)
        sample_batched['ga_split'] = ga_split


    elif cfg.input_augmentation == 'random_warp_input':
        num_img_in_batch = sample_batched['mask'].shape[0]
        if mode == 'train' or mode == 'val':
            #v1 params
            theta = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 2.0 # yaw augmentation,-22.5, np.zeros(num_img_in_batch)
            phi = (np.random.ranf(num_img_in_batch) -0.5) * np.pi / 1.2 # roll augmentation 75
            #v3 params
            # for i in range(0, num_img_in_batch):
            #     #if np.random.ranf() <= 1.0: #apply roll only augment
            #     # theta[i] = 0
            #     # phi[i] = (np.random.ranf() - 0.5) * np.pi
            #     # if np.random.ranf() < 0.5: #apply pitch only augment
            #     theta[i] = (np.random.ranf() - 0.5) * np.pi / 2 # -60 ~ 60 degree.
            #     phi[i] = 0
        else:
            #use fixed rot
            #theta = np.pi / 8.0 * np.ones(num_img_in_batch)
            #phi = np.pi / 7.0 * np.ones(num_img_in_batch)
            #v1 params
            theta = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 4.0  # yaw augmentation,
            phi = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 1.2  # roll augmentation -75 ~ 75

            #theta = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 1.5  # yaw augmentation,np.zeros(num_img_in_batch)
            #phi = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 1.0  # roll augmentation
            # for i in range(0, num_img_in_batch):
            #     theta[i] = 0
            #     phi[i] = (1- 0.5) * np.pi / 1.0
            #     # if np.random.ranf() < 0.3: #apply roll only augment
            #     #     theta[i] = 0
            #     #     phi[i] = (np.random.ranf() - 0.5) * np.pi / 1.0
            #     # elif np.random.ranf() < 0.3: #apply yaw only augment
            #     #     theta[i] = (np.random.ranf() - 0.5) * np.pi / 1.5
            #     #     phi[i] = 0

        gravity_dir = np.vstack((-np.cos(theta)*np.sin(phi),
                                 np.cos(theta)*np.cos(phi),
                                 np.sin(theta)))
        gravity_dir = torch.tensor(gravity_dir.transpose(), dtype=torch.float).view(num_img_in_batch, 3)
        gravity_dir = gravity_dir.cuda()
        Y_dir = torch.tensor([0.0, 1.0, 0.0]).cuda()

        for i in range(0, num_img_in_batch):
            if np.random.ranf() < 0.3: # original image 1/4 images
                gravity_dir[i, :] = Y_dir


        alignment_dir = Y_dir.repeat(num_img_in_batch, 1)
        sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched, I_g=gravity_dir, I_a=alignment_dir, image_border=cfg.image_padding_mode)

        sample_batched['ga_split'] = ga_split

    return sample_batched







class OpticalConverter(object):
    def __init__(self, train_dataset='scannet', test_dataset='OurDataset'):

        self.train_dataset = train_dataset
        if self.train_dataset == 'scannet':
            self.train_focal = 288.79533 #=1169.621094*320/1296
            self.train_maxdepth = 500
        else:
            print('Erros : No train dataset exist...', train_dataset)
            exit(0)
        self.MIN_DEPTH_CLIP = 1.0  # meter scale in ScanNet dataset
        self.MAX_DEPTH_CLIP = 10.0  # meter scale in ScanNet dataset
        self.test_dataset = test_dataset
        if 'OurDataset' in self.test_dataset:
            self.test_focal = 299.59104 #focal = 599.18208  *320/640
            self.test_maxdepth = 5460
        elif self.test_dataset == 'TUMrgbd_frei1rpy':
            self.test_focal = 258.65 #517.3*320/640
            self.test_maxdepth = 5000
        elif self.test_dataset == 'TUMrgbd_frei2rpy':
            self.test_focal = 260.45 #520.9*320/640
            self.test_maxdepth = 5000
        elif self.test_dataset == 'TUMrgbd_frei3rpy':
            self.test_focal = 267.7 #535.4*320/640
            self.test_maxdepth = 5000
        elif self.test_dataset == 'scannet' or self.test_dataset == 'demo_dataset':
            self.test_focal = 288.79533
            self.test_maxdepth = 1000
        elif self.test_dataset == 'NYUv2':
            self.test_focal = 290.5 #581/2
            self.test_maxdepth = 1000
        else:
            print('Erros : No test dataset exist...', self.test_dataset)
            exit(0)
        print('Scale Refine Law Defined : TrainDataset={} train_focal={} train_maxdepth={}'.format(self.train_dataset,
                                                                                                   self.train_focal,
                                                                                                   self.train_maxdepth))
        print('                         : TestDataset={} test_focal={} test_maxdepth={}'.format(self.test_dataset,
                                                                                                   self.test_focal,
                                                                                                   self.test_maxdepth))
    def scale_refine(self, x):
        # x *= self.train_maxdepth
        # x = x * (self.test_focal / self.train_focal)
        # return x / self.test_maxdepth

        x = ((x + 1.0) * 0.5 * (self.MAX_DEPTH_CLIP - self.MIN_DEPTH_CLIP)) + self.MIN_DEPTH_CLIP
        x = x * (self.test_focal / self.train_focal)
        return x




if __name__ == '__main__':
    theta = math.radians(-30)
    phi = math.radians(0)
    gravity_dir = np.vstack((-np.cos(theta) * np.sin(phi),
                             np.cos(theta) * np.cos(phi),
                             np.sin(theta)))
    print(gravity_dir)
