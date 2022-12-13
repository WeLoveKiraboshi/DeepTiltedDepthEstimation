from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.DenseDepth_loss import DenseDepthLoss, DenseDepthLoss_, ssim
from loss.sr_loss import SRonly_Loss, SRonly_Loss_


class SR_DenseDepth_Full_Loss(nn.Module):
    def __init__(self, w_net=1.0, w_pose=0.01, w_l1=0.1, w_ssim=1.0, w_grad=0.0, pose_mode='optimize'):
        nn.Module.__init__(self)
        # Loss weight param
        self.w_net = w_net
        self.w_pose = w_pose
        #self.sr_loss = SRonly_Loss_(pose_mode)
        #self.net_loss = DenseDepthLoss_(w_l1=w_l1, w_ssim=w_ssim, w_grad=w_grad)

        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.l1_criterion = nn.L1Loss()


    def forward(self, output_prediction, sample_batched):
        prediction_error_g = torch.cosine_similarity(output_prediction['I_g'], sample_batched['gravity'], dim=1,eps=1e-6)
        acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
        cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
        acos_mask_g = acos_mask_g > 0.0
        cos_mask_g = cos_mask_g > 0.0
        optimize_loss = torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])

        depth_pred = torch.mul(output_prediction['depth'], sample_batched['mask'])  # pred
        depth_gt = torch.mul(sample_batched['depth'], sample_batched['mask'])
        l_depth = self.l1_criterion(depth_gt, depth_pred)
        l_ssim = torch.clamp((1 - ssim(depth_gt, depth_pred, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
        net_loss = (self.w_ssim * l_ssim) + (self.w_l1 * l_depth)
        #print('sr = {},   net = {}'.format(0.01 * optimize_loss, net_loss))
        if sample_batched['ga_split'] != 'no_ga':
            loss = self.w_pose * optimize_loss + self.w_net * net_loss
        else:
            loss = self.w_net * net_loss

        return loss, optimize_loss, net_loss

    # def forward(self, output_prediction, sample_batched):
    #
    #     sample_batched['depth'] = torch.nn.functional.grid_sample(sample_batched['depth'],
    #                                                               pred_sample_batched['sampler'], padding_mode='zeros',
    #                                                               mode='bilinear')
    #     sample_batched['depth'] = sample_batched['depth'].view(pred_sample_batched['W_O'].shape[0],
    #                                                            pred_sample_batched['W_O'].shape[1],
    #                                                            pred_sample_batched['W_O'].shape[2],
    #                                                            pred_sample_batched['W_O'].shape[3])
    #
    #     prediction_error_g = torch.cosine_similarity(output_prediction['I_g'], sample_batched['gravity'], dim=1,eps=1e-6)
    #     acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
    #     cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
    #     acos_mask_g = acos_mask_g > 0.0
    #     cos_mask_g = cos_mask_g > 0.0
    #     optimize_loss = torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])
    #
    #     depth_pred = torch.mul(output_prediction['W_O'], output_prediction['W_mask'][:, 0:1, :, :])  # pred
    #     depth_gt = torch.mul(sample_batched['depth'], output_prediction['W_mask'][:, 0:1, :, :])
    #     l_depth = self.l1_criterion(depth_gt, depth_pred)
    #     l_ssim = torch.clamp((1 - ssim(depth_gt, depth_pred, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
    #     net_loss = (self.w_ssim * l_ssim) + (self.w_l1 * l_depth)
    #     #print('sr = {},   net = {}'.format(0.01 * optimize_loss, net_loss))
    #     if sample_batched['ga_split'] != 'no_ga':
    #         loss = self.w_pose * optimize_loss + self.w_net * net_loss
    #     else:
    #         loss = self.w_net * net_loss
    #
    #     return loss, optimize_loss, net_loss






