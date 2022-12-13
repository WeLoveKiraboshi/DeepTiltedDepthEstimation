"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description:
"""

#import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import scipy.misc
import os
import torch
from os import path

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.utils import colorize
from dataloader.data import OpticalConverter

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

def DepthNorm(depth, maxDepth=500.0):
    return maxDepth / depth


class Logger(object):

    def __init__(self, log_dir='./', network='ResnetUnet', name=None):
        """Create a summary writer logging to log_dir."""
        self.name = name
        self.network = network
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            print('Tensorboard Log is logged : ', os.path.join(log_dir, name))
            self.writer = SummaryWriter(logdir="{}".format(os.path.join(log_dir, name)))
        else:
            print('Tensorboard Log is logged : ', log_dir)
            self.writer = SummaryWriter(logdir="{}".format(log_dir))

        self.opt = OpticalConverter(train_dataset='scannet', test_dataset='scannet')

    def scalar_summary(self, tags, values, step):
        """Log a scalar variable.
        """
        self.writer.add_scalar(tag=tags, scalar_value=values, global_step=step)
        self.writer.flush()

    def LogProgressImage(self, model, sample_batched, epoch):
        with torch.no_grad():
            image = torch.autograd.Variable(sample_batched['image'].cuda())

            depth = self.opt.scale_refine(torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True)))
            mask = torch.autograd.Variable(sample_batched['rgb_mask'].cuda())
            depth = torch.mul(depth, mask)

            if epoch % 50 == 0:
                self.writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
            if epoch % 50 == 0:
                for idx in range(sample_batched['image'].size(0)):
                    depth[idx] = depth[idx] / torch.amax(depth[idx])
                self.writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)),
                                      epoch)

            if self.network == 'ResnetUnetPartialConv_v1' or self.network == 'ResnetUnetPartialConv_v2' or self.network == 'UnetPartialConv' or self.network == 'ResnetUnetPartialConv_v3':
                pred_depth, pred_mask = model(image, mask)
                pred_depth = self.opt.scale_refine(pred_depth)

                ##masking process
                # mask = mask[:, 0, :, :].unsqueeze(1)
                pred_depth = torch.mul(pred_depth, mask)  # pred

                for idx in range(sample_batched['image'].size(0)):
                    pred_depth[idx] = pred_depth[idx] / torch.amax(pred_depth[idx])
                    # pred_mask[idx] = pred_mask[idx] / torch.amax(pred_mask[idx])
                # pred_mask = torch.clamp(pred_mask, min=0.0, max=1.0)
                self.writer.add_image('Train.3.Output Depth',
                                      colorize(vutils.make_grid(pred_depth.data, nrow=6, normalize=False)), epoch)
                # self.writer.add_image('Train.4.Output Mask',
                #                       colorize(vutils.make_grid(pred_mask.data, nrow=6, normalize=False)), epoch)
                del pred_mask
            elif self.network == 'ResnetUnet' or self.network == 'DenseDepthNet' or self.network == 'ResnetUnetGatedConv':
                pred_depth = model(image)
                pred_depth = self.opt.scale_refine(pred_depth)
                for idx in range(sample_batched['image'].size(0)):
                    pred_depth[idx] = pred_depth[idx] / torch.amax(pred_depth[idx])
                self.writer.add_image('Train.3.Output Depth',
                                      colorize(vutils.make_grid(pred_depth.data, nrow=6, normalize=False)), epoch)
            elif 'SpatialRectifierResnetUnet' in self.network:
                pred_depth = model(sample_batched)
                pred_depth['depth'] = self.opt.scale_refine(pred_depth['depth'])
                pred_depth['depth'] = torch.mul(pred_depth['depth'], mask)  # pred

                for idx in range(sample_batched['image'].size(0)):
                    pred_depth['depth'][idx] = pred_depth['depth'][idx] / torch.amax(pred_depth['depth'][idx])
                self.writer.add_image('Train.3.Output Depth',
                                      colorize(vutils.make_grid(pred_depth['depth'].data, nrow=6, normalize=False)),
                                      epoch)
            del image
            del depth
            del mask
            del pred_depth