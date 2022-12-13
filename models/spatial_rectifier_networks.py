import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.models
import collections
import math
from models.warping_2dof_alignment import Warping2DOFAlignment

from models.DenseDepthNet import DenseDepthModel
from models.ResnetUnet import ResnetUnetHybrid
import pickle
from utils.utils import check_nan_ckpt
import torchvision.transforms as T

from models.ResnetUnetPartialConv_v1 import ResnetUnetHybridPartialConv_v1
from models.ResnetUnetPartialConv_v2 import ResnetUnetHybridPartialConv_v2
from models.ResnetUnetPartialConv_v3 import ResnetUnetHybridPartialConv_v3
from models.ResnetUnetGatedConv import ResnetUnetHybridGatedConv


class SpatialRectifier(nn.Module):
    def __init__(self, in_channels=3, out_channel=3, is_dropout=False, drop_out=0.3):
        super(SpatialRectifier, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(18)](pretrained=True)

        self.channel = in_channels

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avg_pool = pretrained_model._modules['avgpool']
        self.warping_params_output = nn.Sequential(nn.Linear(512, 128),
                                                   nn.ReLU(True),
                                                   nn.Dropout(),
                                                   nn.Linear(128, out_channel))
        self.is_dropout = is_dropout
        self.dropout1 = torch.nn.Dropout2d(p=drop_out)
        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.is_dropout:
            x = self.dropout1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.avg_pool(x4)
        z = self.warping_params_output(torch.flatten(y, 1))#.register_hook(get_hook("block1_output"))
        return z



#please set appripiate focal lengths of input dataset, ScanNet, FrameNet, NYUv2
class SpatialRectifierDenseDepth(nn.Module):
    def __init__(self, fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]), depth_estimation_cnn_ckpt=None, sr_cnn_ckpt=None, pretrained_net=False):
        super(SpatialRectifierDenseDepth, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.depth_estimation_cnn = DenseDepthModel(pretrained=pretrained_net)

        fc = fc_img
        cc = np.array([160, 120])
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        if sr_cnn_ckpt != None:
            model_sr_cp = pickle.load(open(sr_cnn_ckpt, "rb"))
            self.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            if check_nan_ckpt(self.warp_params_cnn):
                print('SR_cnn model detect nan values in its weight')
                exit(0)
        if depth_estimation_cnn_ckpt != None:
            model_depthnet_cp = pickle.load(open(depth_estimation_cnn_ckpt, "rb"))
            self.depth_estimation_cnn.load_state_dict(model_depthnet_cp['depth_net'], strict=False)
            if check_nan_ckpt(self.depth_estimation_cnn):
                print('DepthEstimationNet model detect nan values in its weight')
                exit(0)


    def forward(self, x):
        # x = sample_batched['image']
        # #gravity_dir = sample_batched['gravity']
        # aligned_dir = sample_batched['aligned_directions']
        # Step 1: Construct warping parameters
        with torch.no_grad():
            v = self.warp_params_cnn(x)
            I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #gravoity_dir
            aligned_dir = torch.tensor([[0, 1, 0]]).to(torch.float32).cuda()  # v[:, 3:6]
            aligned_dirs = aligned_dir.repeat(I_g.shape[0], 1)
            I_a = torch.nn.functional.normalize(aligned_dirs, dim=1, eps=1e-6)

            # Step 2: Construct image sampler forward and inverse
            R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g, I_a)

            # Step 3: Warp input to be canonical, ==> pose rectified image
            w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')

        # Step 4: Depth prediction
        w_y = self.depth_estimation_cnn(w_x)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
        # #since output is depth(ch=1), axis:1 must be dim=1
        y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])
        # print(y.shape)
        # print(R_inv.shape)
        # print(R_inv.bmm(y).shape)
        # n_pred_c = (R_inv.bmm(y)).view(x.shape[0], 1, x.shape[2], x.shape[3])

        #return {'I_g': v[:, 0:3], 'I_a': v[:, 3:6], 'n': n_pred_c, 'W_I': w_x, 'W_O': w_y}
        return {'I_g': v[:, 0:3], 'depth': y, 'W_I': w_x, 'W_O': w_y}




class SpatialRectifierResnetUnet(nn.Module):
    def __init__(self, K=np.eye(3), depth_estimation_cnn_ckpt=None, sr_cnn_ckpt=None, pretrained_net=False, mode='train', vps=False, dataset='scannet', pad_mode='zeros'):
        super(SpatialRectifierResnetUnet, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.depth_estimation_cnn = ResnetUnetHybrid(pretrained=False)
        self.K = K
        self.H = 240
        self.W = 320
        self.mode = mode
        print('fx={},  fy={},  cx={},  cy={}'.format(K[0,0],K[1,1],K[0,2],K[1,2]))
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], mode=mode, dataset=dataset)
        self.vps = vps #only use in test phase
        self.dataset = dataset
        self.pad_mode = pad_mode
        self.IMG_MEAN = torch.tensor([97.9909, 113.3113, 126.4953], dtype=torch.float)

        if sr_cnn_ckpt != None:
            model_sr_cp = pickle.load(open(sr_cnn_ckpt, "rb"))
            self.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            print('loading sr checkpoint')
            if check_nan_ckpt(self.warp_params_cnn):
                print('SR_cnn model detect nan values in its weight')
                exit(0)
        if depth_estimation_cnn_ckpt != None:
            model_depthnet_cp = pickle.load(open(depth_estimation_cnn_ckpt, "rb"))
            self.depth_estimation_cnn.load_state_dict(model_depthnet_cp['depth_net'], strict=False)
            print('loading DepthCNN checkpoint')
            if check_nan_ckpt(self.depth_estimation_cnn):
                print('DepthEstimationNet model detect nan values in its weight')
                exit(0)

    def forward(self, sample_batched):
        x = sample_batched['image']
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = sample_batched['gravity'] #torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #v[:, 0:3] sample_batched['gravity']
        if self.mode == 'test' and self.vps == True:
            g = I_g.view(I_g.shape[0], 1, 3)
            g_dot_es = torch.bmm(g, sample_batched['vps'].permute(0, 2, 1))
            closest_principle_dirs = torch.max(g_dot_es.view(I_g.shape[0], -1), dim=1)
            aligned_dir = torch.empty(size=(I_g.shape[0], 3)).cuda()
            for idx in range(len(closest_principle_dirs.indices)):
                aligned_dir_idx = closest_principle_dirs.indices[idx]
                aligned_dir_sample = sample_batched['vps'][idx, aligned_dir_idx]
                aligned_dir[idx] = aligned_dir_sample
            I_g = torch.nn.functional.normalize(aligned_dir, dim=1, eps=1e-6)
            sample_batched['I_g'] = I_g

        I_a = torch.nn.functional.normalize(sample_batched['aligned_directions'], dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        if 'corners_points' in sample_batched.keys():
            original_corner_pts = sample_batched['corners_points']#[bs, 2, 4]
        else:
            original_corner_pts = torch.Tensor(np.array([[0, 0], [self.W - 1, 0], [0, self.H - 1], [self.W - 1, self.H - 1]]).transpose()).unsqueeze(0).repeat(I_g.shape[0], 1, 1).cuda()
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g.detach(), I_a.detach(), original_corner_pts)

        if self.pad_mode == 'mean':
            w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')
            w_mask = torch.nn.functional.grid_sample(sample_batched['rgb_mask'], img_sampler, padding_mode='zeros', mode='nearest')
            w_mask = w_mask.view(w_x.shape[0], w_x.shape[1], w_x.shape[2], w_x.shape[3])
            for i_ch in range(3):
                w_x[:, i_ch, :, :] = w_x[:, i_ch, :, :].masked_fill(w_mask[:, i_ch, :, :] == 0, self.IMG_MEAN[i_ch]/255)
        else:
            w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode=self.pad_mode, mode='bilinear',align_corners=True)



        # Step 4: Depth prediction
        w_y = self.depth_estimation_cnn(w_x) #if we change w_x => x, loss converged

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='nearest')
        # #since output is depth(ch=1), axis:1 must be dim=1
        y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])
        return {'I_g': I_g, 'I_a': I_a, 'depth': y, 'W_I': w_x, 'W_O': w_y}

    # def forward(self, x):
    #     # Step 1: Construct warping parameters
    #     v = self.warp_params_cnn(x)
    #     I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #v[:, 0:3] sample_batched['gravity']
    #
    #     I_a = torch.Tensor([0, 1, 0]).unsqueeze(0)
    #     print(I_a.shape)
    #     print(I_g.shape)
    #
    #     #original_corner_pts = torch.Tensor(np.array([[0, 0], [self.W - 1, 0], [0, self.H - 1], [self.W - 1, self.H - 1]]).transpose()).unsqueeze(0).repeat(I_g.shape[0], 1, 1).cuda()
    #     #R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g.detach(), I_a.detach(), original_corner_pts)
    #
    #     #w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')
    #
    #     w_y = self.depth_estimation_cnn(x) #if we change w_x => x, loss converged
    #
    #     # Step 5: Inverse warp the output to be pixel wise with input
    #     #y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
    #     # #since output is depth(ch=1), axis:1 must be dim=1
    #     #y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])
    #     return w_y




class SpatialRectifierResnetUnetPartialConv_v2(nn.Module):
    def __init__(self, K=np.eye(3), depth_estimation_cnn_ckpt=None, sr_cnn_ckpt=None, pretrained_net=False, mode='train', dataset='scannet', pad_mode='zeros'):
        super(SpatialRectifierResnetUnetPartialConv_v2, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.depth_estimation_cnn = ResnetUnetHybridPartialConv_v2(pretrained=True)
        self.K = K
        self.H = 240
        self.W = 320
        print('fx={},  fy={},  cx={},  cy={}'.format(K[0,0],K[1,1],K[0,2],K[1,2]))
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2],mode=mode, dataset=dataset)
        self.pad_mode = pad_mode
        self.IMG_MEAN = torch.tensor([97.9909, 113.3113, 126.4953], dtype=torch.float)

        if sr_cnn_ckpt != None:
            model_sr_cp = pickle.load(open(sr_cnn_ckpt, "rb"))
            self.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            print('loading sr checkpoint')
            if check_nan_ckpt(self.warp_params_cnn):
                print('SR_cnn model detect nan values in its weight')
                exit(0)
        if depth_estimation_cnn_ckpt != None:
            model_depthnet_cp = pickle.load(open(depth_estimation_cnn_ckpt, "rb"))
            self.depth_estimation_cnn.load_state_dict(model_depthnet_cp['depth_net'], strict=False)
            print('loading DepthCNN checkpoint')
            if check_nan_ckpt(self.depth_estimation_cnn):
                print('DepthEstimationNet model detect nan values in its weight')
                exit(0)

    def forward(self, sample_batched):
        x = sample_batched['image']
        mask = sample_batched['rgb_mask']
        # #gravity_dir = sample_batched['gravity']
        aligned_dirs = sample_batched['aligned_directions']
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #gravoity_dir
        #aligned_dir = torch.tensor([[0, 1, 0]]).to(torch.float32).cuda() #v[:, 3:6]
        #aligned_dirs = aligned_dir.repeat(I_g.shape[0],1)
        I_a = torch.nn.functional.normalize(aligned_dirs, dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        if 'corners_points' in sample_batched.keys():
            original_corner_pts = sample_batched['corners_points']#[bs, 2, 4]
        else:
            #original_corner_pts = np.array([[0, 0, 1], [self.W - 1, 0, 1], [0, self.H - 1, 1], [self.W - 1, self.H - 1, 1]]).transpose() #[bs, 2, 4]
            original_corner_pts = torch.Tensor(np.array([[0, 0], [self.W - 1, 0], [0, self.H - 1], [self.W - 1, self.H - 1]]).transpose()).unsqueeze(0).repeat(I_g.shape[0], 1, 1).cuda()
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g.detach(), I_a.detach(), original_corner_pts)

        # Step 3: Warp input to be canonical, ==> pose rectified image
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')
        w_mask = torch.nn.functional.grid_sample(mask, img_sampler, padding_mode='zeros', mode='nearest')
        w_mask = w_mask.view(w_x.shape[0], w_x.shape[1], w_x.shape[2], w_x.shape[3])
        if self.pad_mode == 'mean':
            for i_ch in range(3):
                w_x[:, i_ch, :, :] = w_x[:, i_ch, :, :].masked_fill(w_mask[:, i_ch, :, :] == 0, self.IMG_MEAN[i_ch]/255)


        # Step 4: Depth prediction
        w_y, w_y_mask = self.depth_estimation_cnn(w_x, w_mask)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='nearest')
        # #since output is depth(ch=1), axis:1 must be dim=1
        y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])

        return {'I_g': I_g, 'depth': y, 'W_I': w_x, 'W_mask': w_mask, 'W_O': w_y, 'W_O_mask': w_y_mask}




class SpatialRectifierResnetUnetPartialConv_v3(nn.Module):
    def __init__(self, K=np.eye(3), depth_estimation_cnn_ckpt=None, sr_cnn_ckpt=None, pretrained_net=False, mode='train',dataset='scannet'):
        super(SpatialRectifierResnetUnetPartialConv_v3, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.depth_estimation_cnn = ResnetUnetHybridPartialConv_v3(pretrained=True)
        self.K = K
        self.H = 240
        self.W = 320
        print('fx={},  fy={},  cx={},  cy={}'.format(K[0,0],K[1,1],K[0,2],K[1,2]))
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], mode=mode, dataset=dataset)

        if sr_cnn_ckpt != None:
            model_sr_cp = pickle.load(open(sr_cnn_ckpt, "rb"))
            self.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            print('loading sr checkpoint')
            if check_nan_ckpt(self.warp_params_cnn):
                print('SR_cnn model detect nan values in its weight')
                exit(0)
        if depth_estimation_cnn_ckpt != None:
            model_depthnet_cp = pickle.load(open(depth_estimation_cnn_ckpt, "rb"))
            self.depth_estimation_cnn.load_state_dict(model_depthnet_cp['depth_net'], strict=False)
            print('loading DepthCNN checkpoint')
            if check_nan_ckpt(self.depth_estimation_cnn):
                print('DepthEstimationNet model detect nan values in its weight')
                exit(0)

    def forward(self, sample_batched):
        x = sample_batched['image']
        mask = sample_batched['rgb_mask']
        # #gravity_dir = sample_batched['gravity']
        aligned_dirs = sample_batched['aligned_directions']
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #gravoity_dir
        #aligned_dir = torch.tensor([[0, 1, 0]]).to(torch.float32).cuda() #v[:, 3:6]
        #aligned_dirs = aligned_dir.repeat(I_g.shape[0],1)
        I_a = torch.nn.functional.normalize(aligned_dirs, dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        if 'corners_points' in sample_batched.keys():
            original_corner_pts = sample_batched['corners_points']#[bs, 2, 4]
        else:
            #original_corner_pts = np.array([[0, 0, 1], [self.W - 1, 0, 1], [0, self.H - 1, 1], [self.W - 1, self.H - 1, 1]]).transpose() #[bs, 2, 4]
            original_corner_pts = torch.Tensor(np.array([[0, 0], [self.W - 1, 0], [0, self.H - 1], [self.W - 1, self.H - 1]]).transpose()).unsqueeze(0).repeat(I_g.shape[0], 1, 1).cuda()
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g.detach(), I_a.detach(), original_corner_pts)

        # Step 3: Warp input to be canonical, ==> pose rectified image
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')
        w_mask = torch.nn.functional.grid_sample(mask, img_sampler, padding_mode='zeros', mode='nearest')
        w_mask = w_mask.view(mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])


        # Step 4: Depth prediction
        w_y, w_y_mask = self.depth_estimation_cnn(w_x, w_mask)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='nearest')
        # #since output is depth(ch=1), axis:1 must be dim=1
        y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])
        return {'I_g': I_g, 'depth': y, 'W_I': w_x, 'W_mask': w_mask, 'W_O': w_y}



class SpatialRectifierResnetUnetGatedConv(nn.Module):
    def __init__(self, K=np.eye(3), depth_estimation_cnn_ckpt=None, sr_cnn_ckpt=None, pretrained_net=False):
        super(SpatialRectifierResnetUnetGatedConv, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.depth_estimation_cnn = ResnetUnetHybridGatedConv(pretrained=True)
        self.K = K
        self.H = 240
        self.W = 320
        print('fx={},  fy={},  cx={},  cy={}'.format(K[0,0],K[1,1],K[0,2],K[1,2]))
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])

        if sr_cnn_ckpt != None:
            model_sr_cp = pickle.load(open(sr_cnn_ckpt, "rb"))
            self.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            print('loading sr checkpoint')
            if check_nan_ckpt(self.warp_params_cnn):
                print('SR_cnn model detect nan values in its weight')
                exit(0)
        if depth_estimation_cnn_ckpt != None:
            model_depthnet_cp = pickle.load(open(depth_estimation_cnn_ckpt, "rb"))
            self.depth_estimation_cnn.load_state_dict(model_depthnet_cp['depth_net'], strict=False)
            print('loading DepthCNN checkpoint')
            if check_nan_ckpt(self.depth_estimation_cnn):
                print('DepthEstimationNet model detect nan values in its weight')
                exit(0)

    def forward(self, sample_batched):
        x = sample_batched['image']
        mask = sample_batched['rgb_mask']
        aligned_dirs = sample_batched['aligned_directions']
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6) #gravoity_dir
        I_a = torch.nn.functional.normalize(aligned_dirs, dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        if 'corners_points' in sample_batched.keys():
            original_corner_pts = sample_batched['corners_points']#[bs, 2, 4]
        else:
            #original_corner_pts = np.array([[0, 0, 1], [self.W - 1, 0, 1], [0, self.H - 1, 1], [self.W - 1, self.H - 1, 1]]).transpose() #[bs, 2, 4]
            original_corner_pts = torch.Tensor(np.array([[0, 0], [self.W - 1, 0], [0, self.H - 1], [self.W - 1, self.H - 1]]).transpose()).unsqueeze(0).repeat(I_g.shape[0], 1, 1).cuda()
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g.detach(), I_a.detach(), original_corner_pts)

        # Step 3: Warp input to be canonical, ==> pose rectified image
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')
        w_mask = torch.nn.functional.grid_sample(mask, img_sampler, padding_mode='zeros', mode='nearest')
        w_mask = w_mask.view(w_x.shape[0], w_x.shape[1], w_x.shape[2], w_x.shape[3])


        # Step 4: Depth prediction
        w_y = self.depth_estimation_cnn(w_x)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='nearest')
        # #since output is depth(ch=1), axis:1 must be dim=1
        y = y.view(x.shape[0], 1, x.shape[2], x.shape[3])
        return {'I_g': I_g, 'depth': y, 'W_I': w_x, 'W_mask': w_mask, 'W_O': w_y}


if __name__ == '__main__':
    warp_param_net = SpatialRectifier()
    warp_param_net.cuda()
