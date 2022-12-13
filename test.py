import numpy as np
import argparse
import time
import datetime
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import pickle
import psutil

# for val
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from models.warping_2dof_alignment import Warping2DOFAlignment
from dataloader.data import create_dataset_loader, data_augmentation, OpticalConverter

from models.spatial_rectifier_networks import *
from models.DenseDepthNet import DenseDepthModel
from models.ResnetUnet import ResnetUnetHybrid, IMUResnetUnetHybrid
from models.ResnetUnetGatedConv import ResnetUnetHybridGatedConv
from models.ResnetUnetPartialConv_v1 import ResnetUnetHybridPartialConv_v1
from models.ResnetUnetPartialConv_v2 import ResnetUnetHybridPartialConv_v2
from models.ResnetUnetPartialConv_v3 import ResnetUnetHybridPartialConv_v3
from models.UnetVGGPartialConvolution import PConvUNet

from config_loader import Config
from utils.tb_logger import Logger
from utils.utils import AverageMeter,compute_errors, BinAverageMeter, abs_rel_loss
from utils.visualize_gravitydir import visualize_extrinsic
from utils.save_tensor_im import saving_gravity_tensor_to_file,saving_rgb_tensor_to_file, draw_gravity_dir

from utils.torch import to_cpu
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.VideoWriter import VideoWriter
from utils.save_tensor_im import rotation_matrix_from_vectors


def system_setup(cfg, args):
    """setup"""
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    if args.gpus == 2:
        gpus = (0, 1)
        device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
    elif args.gpus == 1:
        device = torch.device('cuda', index=int(args.gpuids)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuids)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.sr_checkpoint_path != None or cfg.checkpoint_path != None or cfg.full_checkpoint_path != None:
        if args.imu == True:
            print('Loaded... model imu_fixed version....')
            model = IMUResnetUnetHybrid(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path, mode='test')
        elif cfg.network == 'SpatialRectifier':
            model = SpatialRectifier(in_channels=3, out_channel=3, is_dropout=True, drop_out=cfg.dropout_p)
            assert os.path.isfile(cfg.sr_checkpoint_path), \
                "=> no model found at '{}'".format(cfg.sr_checkpoint_path)
            print("=> loading model '{}'".format(cfg.sr_checkpoint_path))
            model_cp = pickle.load(open(cfg.sr_checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'DenseDepthNet':
            model = DenseDepthModel(pretrained=False)
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'ResnetUnet':
            model = ResnetUnetHybrid().cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'ResnetUnetPartialConv_v1': #Partial Conv without re-weithing of learned weight
            model = ResnetUnetHybridPartialConv_v1().cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'ResnetUnetPartialConv_v2': #original PartailConv of NVIDI script
            model = ResnetUnetHybridPartialConv_v2().cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'ResnetUnetPartialConv_v3':
            model = ResnetUnetHybridPartialConv_v3(pretrained=False).cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'UnetVGGPartialConv':
            model = PConvUNet().cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'ResnetUnetGatedConv':
            model = ResnetUnetHybridGatedConv(pretrained=False).cuda()
            assert os.path.isfile(cfg.checkpoint_path), \
                "=> no model found at '{}'".format(cfg.checkpoint_path)
            print("=> loading model '{}'".format(cfg.checkpoint_path))
            model_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_cp[cfg.mode], strict=False)

        elif cfg.network == 'SpatialRectifierDenseDepth':
            model = SpatialRectifierDenseDepth(depth_estimation_cnn_ckpt=cfg.checkpoint_path,
                                               sr_cnn_ckpt=cfg.sr_checkpoint_path)
            if cfg.full_checkpoint_path != None:
                assert os.path.isfile(cfg.full_checkpoint_path), \
                    "=> no model found at '{}'".format(cfg.full_checkpoint_path)
                print("=> loading model '{}'".format(cfg.full_checkpoint_path))
                model_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
                model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'SpatialRectifierResnetUnet':
            model = SpatialRectifierResnetUnet(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path, sr_cnn_ckpt=cfg.sr_checkpoint_path, mode='test', vps=cfg.vps, dataset=cfg.dataset, pad_mode=cfg.image_padding_mode)
            if cfg.full_checkpoint_path != None:
                assert os.path.isfile(cfg.full_checkpoint_path), \
                    "=> no model found at '{}'".format(cfg.full_checkpoint_path)
                print("=> loading model '{}'".format(cfg.full_checkpoint_path))
                model_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
                model.load_state_dict(model_cp['sr_depth_net'], strict=False)
        elif cfg.network == 'SpatialRectifierResnetUnetPartialConv_v2':
            model = SpatialRectifierResnetUnetPartialConv_v2(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path,
                                                             sr_cnn_ckpt=cfg.sr_checkpoint_path, mode='test', dataset=cfg.dataset,pad_mode=cfg.image_padding_mode)
            if cfg.full_checkpoint_path != None:
                assert os.path.isfile(cfg.full_checkpoint_path), \
                    "=> no model found at '{}'".format(cfg.full_checkpoint_path)
                print("=> loading model '{}'".format(cfg.full_checkpoint_path))
                model_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
                model.load_state_dict(model_cp[cfg.mode], strict=False)

        elif cfg.network == 'SpatialRectifierResnetUnetPartialConv_v3':
            model = SpatialRectifierResnetUnetPartialConv_v3(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path,
                                                             sr_cnn_ckpt=cfg.sr_checkpoint_path, mode='test', dataset=cfg.dataset)
            if cfg.full_checkpoint_path != None:
                assert os.path.isfile(cfg.full_checkpoint_path), \
                    "=> no model found at '{}'".format(cfg.full_checkpoint_path)
                print("=> loading model '{}'".format(cfg.full_checkpoint_path))
                model_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
                model.load_state_dict(model_cp[cfg.mode], strict=False)
        elif cfg.network == 'SpatialRectifierResnetUnetGatedConv':
            model = SpatialRectifierResnetUnetGatedConv(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path,
                                                        sr_cnn_ckpt=cfg.sr_checkpoint_path)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.full_checkpoint_path))
            # cfg.sr_checkpoint_path = 'results/sr_resnetunet_full/model-best_standard_r.pkl'
            # model_sr_cp = pickle.load(open(cfg.sr_checkpoint_path, "rb"))
            # model.warp_params_cnn.load_state_dict(model_sr_cp['sr_only'], strict=False)
            # print('loading sr checkpoint. again')
            #
            # path = os.path.join('results/sr_resnetunet_gatedconv_full/models/model-best-scannet_full_standard_roll:1.0_20220707_Ours.pkl')
            # model_cp = {cfg.mode: model.state_dict()}
            # pickle.dump(model_cp, open(path, 'wb'))
            # exit(0)

    print('* Image padding mode : {}'.format(cfg.image_padding_mode))

    if args.gpus == 2:
        model = torch.nn.DataParallel(model)  # make parallel
    elif args.gpus == 1:
        model.to(device)
    # from torchinfo import summary
    # # summary(model, (3, 240, 320))
    # summary(
    #     model,
    #     input_size=(1, 3, 240, 320),
    #     col_names=["output_size", "num_params"],
    # )
    # from thop import profile
    # inputs = torch.randn(1, 3, 240, 320).cuda()
    # flops, params = profile(model, (inputs,))
    # print('flops(GB): ', flops/1.e9, 'params: ', params)

    return model




def run_epoch(dataset, model, mode='train', epoch=0, cfg=None, warper=None):
    model.eval()
    batch_time = AverageMeter()

    if args.bin:
        error_metrics = BinAverageMeter(upper=90, lower=-90, step=10)
        loss_criteria = torch.nn.L1Loss()
    predTensor = torch.zeros((0, 1, cfg.imsize[0], cfg.imsize[1])).to('cpu')
    grndTensor = torch.zeros((0, 1, cfg.imsize[0], cfg.imsize[1])).to('cpu')
    maskTensor = torch.zeros((0, 1, cfg.imsize[0], cfg.imsize[1])).to('cpu')
    pred_gravTensor = torch.zeros((0, 3)).to('cpu')
    grnd_gravTensor = torch.zeros((0, 3)).to('cpu')
    elapsed_time_meter = AverageMeter()
    cpu_memory_meter = AverageMeter()

    opt = OpticalConverter(train_dataset='scannet', test_dataset=cfg.dataset)

    N = len(dataset)
    end = time.time()
    errors = []

    if args.save_video:
        if cfg.mode == 'sr_depth_net':
            video_writer = VideoWriter(os.path.join(cfg.cfg_dir, cfg.mode + '_'+cfg.dataset+'.mp4'), 10.0,(320*3, 240))
        elif cfg.mode == 'sr_only':
            video_writer = VideoWriter(os.path.join(cfg.cfg_dir, cfg.mode + '_' + cfg.dataset + '.mp4'), 10.0, (320 * 2, 240))


    with torch.no_grad():
        for iter, sample_batched in enumerate(dataset):
            for data_key, data_value in sample_batched.items():
                if torch.is_tensor(data_value):
                    sample_batched[data_key] = sample_batched[data_key].cuda()

            output_prediction = {'pred_depth': {}, 'I_g': {}}

            #apply random roll rotation for dataset that contains almost no roll rotation
            if cfg.dataset == 'scannet':
                sample_batched = data_augmentation(sample_batched, cfg, warper, epoch, iter, mode)


            if cfg.mode == 'sr_only':
                pred_gravity = model.forward(sample_batched['image'])
                #pred_gravity = pred_gravity[:] / torch.linalg.norm(pred_gravity[:])
                pred_gravity = F.normalize(pred_gravity, dim=-1, p=2)
                output_prediction['I_g'] = pred_gravity
                sample_batched_rectified = warper.warp_all_with_gravity_center_aligned(sample_batched, I_g=sample_batched['gravity'], #pred_gravity
                                                                                       I_a=sample_batched['aligned_directions'])
                if args.imshow or args.save_video:
                    for ii in range(sample_batched['image'].shape[0]):
                        frame_idx = iter * cfg.bs + ii
                        #print('pred_gravity = {}, aligned_dir = {}'.format(pred_gravity[ii], sample_batched['aligned_directions'][ii]))
                        # print('groundtruth_gravity = {}, aligned_dir = {}'.format(sample_batched['gravity'][ii],
                        #                                                    sample_batched['aligned_directions'][ii]))
                        output_rgb_img_rectified = np.uint8(sample_batched_rectified['image'][ii].detach().cpu().permute(1, 2, 0) * 255).copy()
                        output_rgb_img = np.uint8(sample_batched['image'][ii].detach().cpu().permute(1, 2, 0) * 255).copy()
                        gravity_target = np.array([0, 1., 0], dtype=np.float32)
                        centVec = (int(output_rgb_img.shape[1] / 2), int(output_rgb_img.shape[0] / 2))
                        #pred_g = pred_gravity[ii].to('cpu').detach().numpy().copy()
                        #pred_g = pred_g / np.linalg.norm(pred_g)
                        pred_g = sample_batched['gravity'][ii].detach().cpu().numpy().copy()
                        rotMat = rotation_matrix_from_vectors(pred_g, gravity_target)
                        rotVec, _ = cv2.Rodrigues(rotMat)
                        yaw = rotVec[1]
                        pitch = rotVec[0]
                        theta = rotVec[2]
                        # print('pitch = {}, yaw = {}, roll = {}'.format(math.degrees(pitch), math.degrees(yaw), math.degrees(theta)))
                        rollVec = (int(100 * math.sin(theta) + output_rgb_img.shape[1] / 2),
                                   int(100 * math.cos(theta) + output_rgb_img.shape[0] / 2))
                        rollVec_aligned = (int(output_rgb_img.shape[1] / 2), int(output_rgb_img.shape[0] / 2)+100)
                        cv2.arrowedLine(output_rgb_img, centVec, rollVec, (255, 255, 0), thickness=5)  # light blue
                        cv2.arrowedLine(output_rgb_img, centVec, rollVec_aligned, (255, 0, 255), thickness=5)  # yellow
                        cat_output_imgs = cv2.hconcat([output_rgb_img, output_rgb_img_rectified])
                        cv2.putText(cat_output_imgs, str(frame_idx), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                        cv2.imshow('Gravity_im', cat_output_imgs)
                        cv2.waitKey(100)
                        # if args.save_video:
                        #     video_writer.add_frame(cat_output_imgs)
                        # # # make gif video
                        # extrinsic_savepath = os.path.join(cfg.cfg_dir, 'gif', '%d.png' % (iter*sample_batched['image'].shape[0]+ii))
                        # if not os.path.exists(os.path.join(cfg.cfg_dir, 'gif')):
                        #     os.mkdir(os.path.join(cfg.cfg_dir, 'gif'))
                        # visualize_extrinsic(rgb_tensor=sample_batched['image'][ii], is_save=True, path=extrinsic_savepath, pred_g=sample_batched['gravity'][ii]) #output_prediction['I_g'][ii]
                        #saving_gravity_tensor_to_file(rgb_tensor=sample_batched['image'][ii],path=os.path.join('demo_dataset', 'gravity_ims', '%d_%d.png' % (iter, 0)), is_pred_g = True,pred_g = pred_gravity[ii], is_gt_g = False, gt_g = None, K = K)
                        # saving_rgb_tensor_to_file(rgb_tensor=sample_batched_rectified['image'][ii], path=os.path.join('demo_dataset', 'aligned_ims', '%d_%d.png' % (iter, 0)))
            elif cfg.mode == 'depth_net':
                if args.imu == True:
                    output_prediction = model.forward(sample_batched)
                    output_prediction['pred_depth'] = output_prediction['pred_depth'].detach().cpu()
                elif cfg.network == 'ResnetUnetPartialConv_v1' or cfg.network == 'ResnetUnetPartialConv_v2' or cfg.network == 'UnetVGGPartialConv' or cfg.network == 'ResnetUnetPartialConv_v3':
                    pred_depth, pred_mask = model.forward(sample_batched['image'], sample_batched['rgb_mask'])
                    output_prediction['pred_depth'] = pred_depth.detach().cpu()
                else:
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    pred_depth = model.forward(sample_batched['image'])

                    end_cuda.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_cuda.elapsed_time(end_cuda)
                    elapsed_time_meter.update(elapsed_time, sample_batched['image'].size(0))
                    output_prediction['pred_depth'] = pred_depth.detach().cpu()
                mask = sample_batched['mask'].detach().cpu()
                sample_batched['depth'] = sample_batched['depth'].detach().cpu()
                output_prediction['pred_depth'] = opt.scale_refine(output_prediction['pred_depth'])
                predTensor = torch.cat((predTensor, output_prediction['pred_depth']), dim=0)
                grndTensor = torch.cat((grndTensor, sample_batched['depth']), dim=0)
                if cfg.network == 'ResnetUnetPartialConv_v1' or cfg.network == 'ResnetUnetPartialConv_v2' or cfg.network == 'UnetVGGPartialConv' or cfg.network == 'ResnetUnetPartialConv_v3':
                    maskTensor = torch.cat((maskTensor, mask[:, 0, :, :].unsqueeze(1)), dim=0)
                else:
                    maskTensor = torch.cat((maskTensor, mask), dim=0)



                if args.bin:
                    for ii in range(sample_batched['image'].shape[0]):
                        rotMat = rotation_matrix_from_vectors(
                            sample_batched['gravity'][ii].to('cpu').detach().numpy().copy(),
                            sample_batched['aligned_directions'][ii].to('cpu').detach().numpy().copy())
                        rotVec, _ = cv2.Rodrigues(rotMat)
                        yaw = rotVec[1]
                        pitch = rotVec[0]
                        roll = rotVec[2]
                        #loss = loss_criteria(output_prediction['pred_depth'], sample_batched['depth'])
                        loss = abs_rel_loss(output_prediction['pred_depth'], sample_batched['depth'], mask)
                        error_metrics.update(val=loss.data.item(), x=math.degrees(roll))

                for ii in range(sample_batched['image'].shape[0]):
                    pred_depth_frame_rectified = output_prediction['pred_depth'][ii].squeeze()
                    timestamp = sample_batched['timestamp'][ii]
                    cv2.imwrite('./depth/' + str(timestamp) + '.png',
                                np.uint16(pred_depth_frame_rectified * 1000.0))
                    print('./depth/' + str(timestamp) + '.png')

                if args.imshow:
                    for ii in range(sample_batched['image'].shape[0]):
                        frame_idx = iter * cfg.bs + ii + 1
                        # if frame_idx != 370:
                        #     continue
                        if args.imu == True:
                            rgb_frame_input = output_prediction['W_I'][ii].detach().cpu().permute(1, 2, 0)[:, :, [2, 1, 0]]
                        else:
                            rgb_frame_input = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)[:, :, [2, 1, 0]]

                        pred_depth_frame_rectified = output_prediction['pred_depth'][ii].squeeze()
                        pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                        pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]

                        gt_depth_frame_rectified = sample_batched['depth'][ii].detach().cpu().squeeze()
                        gt_depth_frame_rectified = gt_depth_frame_rectified / torch.amax(gt_depth_frame_rectified)
                        gt_depth_frame_rectified_cmap = torch.Tensor(cm.jet(gt_depth_frame_rectified))[:, :, :3]

                        hcat_frames_rectified = torch.cat(
                            [rgb_frame_input, pred_depth_frame_rectified_cmap, gt_depth_frame_rectified_cmap],
                            axis=1).numpy().copy()
                        hcat_np = cv2.cvtColor(np.uint8(hcat_frames_rectified * 255), cv2.COLOR_BGR2RGB)

                        cv2.imwrite('./ScanNet/ResnetUnet_imu/'+str(frame_idx)+'.png',
                                    cv2.cvtColor(np.uint8(pred_depth_frame_rectified_cmap.numpy().copy() * 255),
                                                 cv2.COLOR_BGR2RGB))
                        cv2.imshow('frames', hcat_np)
                        cv2.waitKey(10)
                        if frame_idx > 300:
                            exit(0)

                mem = psutil.virtual_memory()
                #print('Memory consumption [{}] : {} GB/ {}GB'.format(iter, mem.used/1.e9, mem.total/1.e9))
                cpu_memory_meter.update(mem.used/1.e9, sample_batched['image'].size(0))





            elif cfg.mode == 'sr_depth_net':
                if cfg.network == 'SpatialRectifierResnetUnet':
                    # rgb_frame_augmented = sample_batched['image'][ID].detach().cpu().permute(1, 2, 0)[:, :, [2, 1, 0]]
                    # cv2.imwrite('./rgb_augmented.png',cv2.cvtColor(np.uint8(rgb_frame_augmented * 255), cv2.COLOR_BGR2RGB))
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()
                    pred_sample_batched = model.forward(sample_batched)
                    end_cuda.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_cuda.elapsed_time(end_cuda)
                    elapsed_time_meter.update(elapsed_time, sample_batched['image'].size(0))

                    mem = psutil.virtual_memory()
                    #print('Memory consumption [{}] : {} GB/ {}GB'.format(iter, mem.used / 1.e9, mem.total / 1.e9))
                    cpu_memory_meter.update(mem.used / 1.e9, sample_batched['image'].size(0))

                elif 'SpatialRectifierResnetUnetPartialConv' in cfg.network or 'SpatialRectifierResnetUnetGatedConv' in cfg.network:
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()
                    pred_sample_batched = model.forward(sample_batched)
                    end_cuda.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_cuda.elapsed_time(end_cuda)
                    elapsed_time_meter.update(elapsed_time, sample_batched['image'].size(0))
                mask = sample_batched['mask'].detach().cpu()
                output_prediction['pred_depth'] = pred_sample_batched['depth'].detach().cpu()
                output_prediction['I_g'] = pred_sample_batched['I_g'].detach().cpu()
                sample_batched['depth'] = sample_batched['depth'].detach().cpu()
                output_prediction['pred_depth'] = opt.scale_refine(output_prediction['pred_depth'])

                predTensor = torch.cat((predTensor, output_prediction['pred_depth']), dim=0)
                grndTensor = torch.cat((grndTensor, sample_batched['depth']), dim=0)
                maskTensor = torch.cat((maskTensor, mask), dim=0)
                pred_gravTensor = torch.cat((pred_gravTensor, output_prediction['I_g']), dim=0)
                grnd_gravTensor = torch.cat((grnd_gravTensor, sample_batched['gravity'].detach().cpu()), dim=0)


                for ii in range(sample_batched['image'].shape[0]):
                    frame_idx = iter * cfg.bs + ii + 1

                    pred_depth_frame_rectified = output_prediction['pred_depth'][ii].squeeze()
                    timestamp = sample_batched['timestamp'][ii]
                    cv2.imwrite('./depth/' + str(timestamp) + '.png', np.uint16(pred_depth_frame_rectified*1000.0))
                    print('./depth/' + str(timestamp) + '.png')


                    if args.bin:
                        rotMat = rotation_matrix_from_vectors(
                            sample_batched['gravity'][ii].to('cpu').detach().numpy().copy(),
                            sample_batched['aligned_directions'][ii].to('cpu').detach().numpy().copy())
                        rotVec, _ = cv2.Rodrigues(rotMat)
                        yaw = rotVec[1]
                        pitch = rotVec[0]
                        roll = rotVec[2]
                        loss = abs_rel_loss(output_prediction['pred_depth'], sample_batched['depth'], mask)
                        #loss = loss_criteria(output_prediction['I_g'], sample_batched['gravity'].detach().cpu())
                        error_metrics.update(val=loss.data.item(), x=math.degrees(roll))

                    rgb_frame = pred_sample_batched['W_I'][ii].detach().cpu().permute(1, 2, 0)[:, :,
                                [2, 1, 0]]
                    cv2.imwrite('./PConv/W_I/' + str(frame_idx) + '.png',
                                cv2.cvtColor(np.uint8(rgb_frame.numpy().copy() * 255), cv2.COLOR_BGR2RGB))
                    rgb_mask = pred_sample_batched['W_mask'][ii].detach().cpu().permute(1, 2, 0)[:, :,
                                [2, 1, 0]]
                    cv2.imwrite('./PConv/W_mask/' + str(frame_idx) + '.png',
                                cv2.cvtColor(np.uint8(rgb_mask.numpy().copy() * 255), cv2.COLOR_BGR2RGB))

                    print(pred_sample_batched['W_O_mask'][ii].shape)
                    rgb_masks = pred_sample_batched['W_O_mask'][ii].detach().cpu().permute(1, 2, 0)
                    cv2.imwrite('./PConv/W_O_mask/' + str(frame_idx) + '.png',
                                cv2.cvtColor(np.uint8(rgb_masks.numpy().copy() * 255), cv2.COLOR_BGR2RGB))

                    # rgb_frame_rectified = pred_sample_batched['W_I'][ii].detach().cpu().permute(1, 2, 0)[:, :,
                    #                       [2, 1, 0]]
                    # cv2.imwrite('./Padding//' + str(frame_idx) + '.png',
                    #             cv2.cvtColor(np.uint8(rgb_frame_rectified.numpy().copy() * 255), cv2.COLOR_BGR2RGB))
                    # extrinsic_savepath = './Padding/pose/' + str(frame_idx) + '.png'
                    # visualize_extrinsic(rgb_tensor=sample_batched['image'][ii], is_save=True, path=extrinsic_savepath,
                    #                     pred_g=sample_batched['gravity'][ii])  # output_prediction['I_g'][ii]
                    pred_depth_frame_rectified = opt.scale_refine(pred_sample_batched['W_O'])[ii].detach().cpu().squeeze() * rgb_mask[:, :, 0]
                    pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                    pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]
                    color_recitified_depth_map = cv2.cvtColor(np.uint8(pred_depth_frame_rectified_cmap * 255),
                                             cv2.COLOR_BGR2RGB) * rgb_mask.numpy().copy()
                    cv2.imwrite('./PConv/W_O/' + str(frame_idx) + '.png',color_recitified_depth_map)
                    # gt_depth_frame_rectified = sample_batched['depth'][ii].squeeze()
                    # gt_depth_frame_rectified = gt_depth_frame_rectified / torch.amax(gt_depth_frame_rectified)
                    # gt_depth_frame_rectified_cmap = torch.Tensor(cm.jet(gt_depth_frame_rectified))[:, :, :3]
                    # cv2.imwrite('./ScanNet/GT/' + str(frame_idx) + '.png',
                    #                 cv2.cvtColor(np.uint8(gt_depth_frame_rectified_cmap * 255),
                    #                              cv2.COLOR_BGR2RGB))

                    if args.save_video or args.imshow:
                        rgb_frame_augmented = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)[:, :,
                                              [2, 1, 0]]
                        #rgb_frame_augmented = rgb_frame_augmented * mask[ii].permute(1, 2, 0)

                        # gt = sample_batched['gravity'][ii].detach().cpu() #
                        pred_grav_augmented = torch.nn.functional.normalize(
                            output_prediction['I_g'][ii].unsqueeze(dim=0),
                            dim=1, eps=1e-6).numpy().copy()
                        rgb_frame_augmented = draw_gravity_dir(im=np.uint8(rgb_frame_augmented * 255).copy(),
                                                               grav=pred_grav_augmented,
                                                               align=np.array([0, 1., 0], dtype=np.float32), K=cfg.K)
                        # rgb_frame_input = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)[:,:,[2,1,0]]
                        rgb_frame_rectified = pred_sample_batched['W_I'][ii].detach().cpu().permute(1, 2, 0)[:, :,
                                              [2, 1, 0]]
                        pred_depth_frame_rectified = output_prediction['pred_depth'][
                            ii].detach().cpu().squeeze()  # pred_sample_batched['depth']
                        pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                        pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]
                        hcat_frames_rectified = torch.cat(
                            [rgb_frame_augmented, rgb_frame_rectified, pred_depth_frame_rectified_cmap],
                            axis=1).numpy().copy()

                        hcat_np = cv2.cvtColor(np.uint8(hcat_frames_rectified * 255), cv2.COLOR_BGR2RGB)
                        # cv2.imwrite(os.path.join(cfg.cfg_dir, 'gif', str(frame_idx)+'.png'), cv2.cvtColor(np.uint8(hcat_frames_rectified * 255), cv2.COLOR_BGR2RGB))
                        cv2.putText(hcat_np, str(frame_idx), (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 5,
                                    cv2.LINE_AA)

                        cv2.imshow('frames', hcat_np)
                        cv2.waitKey(10)

                        if args.save_video:
                            hcat_np = cv2.cvtColor(np.uint8(hcat_frames_rectified * 255), cv2.COLOR_BGR2RGB)
                            video_writer.add_frame(hcat_np)
                        if frame_idx > 300:
                            exit(0)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - iter))))

            # Log progress
            if iter % 50 == 0 and args.time == False:
                # Print to console
                print('{mode} Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      .format(epoch, iter, N, batch_time=batch_time, eta=eta, mode=mode))

            """clean up gpu memory"""
            torch.cuda.empty_cache()
            del output_prediction

    if args.time:
        return 1000. / elapsed_time_meter.avg
    elif args.bin:
        print(error_metrics.avg_values)
        return 0
    elif args.cpu:
        print('Total CPU use memory sum (GB) ', cpu_memory_meter.avg)
        return cpu_memory_meter.avg
    if not args.time and not args.bin:
        mean_errors = compute_errors(grndTensor, predTensor, maskTensor)  # np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*list(mean_errors)) + "\\\\")
        loss = nn.L1Loss()
        print('\n gravity evaluation')
        print(loss(pred_gravTensor, grnd_gravTensor))
        print('\n model Processing Time (msec): avg of {} frames'.format(cfg.bs))
        print("avg: {}   sum:{}  total_frame_size:{}".format(elapsed_time_meter.avg, elapsed_time_meter.sum,
                                                             elapsed_time_meter.count))
        print("avg: {} FPS".format(1000. / elapsed_time_meter.avg))
        print("\n-> Done!")
        return 0









def train(cfg, args):
    if cfg.dataset == 'scannet' or cfg.dataset == 'demo_dataset':
        K = np.array([[577.87061 * 0.5, 0., 319.87654 * 0.5], [0, 577.87061 * 0.5, 239.87603 * 0.5], [0., 0., 1.]],
                     dtype=np.float32)
    elif 'OurDataset' in cfg.dataset:
        K = np.array([[266.82119, 0., 319.87654 * 0.5], [0, 266.82119, 239.87603 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'OurDataset-limited':
        K = np.array([[266.82119, 0., 319.87654 * 0.5], [0, 266.82119, 239.87603 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei1rpy':
        K = np.array([[517.3 * 0.5, 0., 318.6 * 0.5], [0, 516.5 * 0.5, 255.3 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei2rpy':  # 520.9	521.0	325.1	249.7	0.2312	-0.7849	-0.0033	-0.0001	0.9172
        K = np.array([[520.9 * 0.5, 0., 325.1 * 0.5], [0, 521.0 * 0.5, 249.7 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei3rpy':  # 535.4	539.2	320.1	247.6	0	0	0	0	0
        K = np.array([[535.4 * 0.5, 0., 320.1 * 0.5], [0, 539.2 * 0.5, 247.6 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'NYUv2':
        K = np.array([[290.5, 0, 160], [0, 290.5, 120], [0, 0, 1]], dtype=np.float32)
    cfg.K = K
    cfg.bs = args.bs
    if args.time or args.bin or args.cpu:
        cfg.bs = 1
    cfg.vps = args.vps
    if args.dataset != 'None':
        cfg.dataset = args.dataset
    if args.mode == 'test':
        cfg.scale_mode = 'test'
    model = system_setup(cfg, args)
    dataloader_train, dataloader_test, dataloader_val = create_dataset_loader(cfg)
    # Step 5. Create warper input:
    warper = Warping2DOFAlignment(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    print('\n')
    print('Dataset[{}] : total images = {}'.format(cfg.dataset, 0))
    print('\n')
    with torch.no_grad():
        if args.time or args.cpu:
            sample_times_list = []
            num_trials = 5
            for time_i in range(num_trials):
                sample_times_list.append(run_epoch(dataloader_test, model, mode='test', epoch=0, cfg=cfg, warper=warper))
            print('\nTotal Process Eval [{} trials]: avg={}  median={}'.format(num_trials, np.mean(sample_times_list), np.median(sample_times_list)))
        else:
            run_epoch(dataloader_test, model, mode='test', epoch=0, cfg=cfg, warper=warper)
    print('***** test has done. *****')

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Tilted Depth Estimation via spatial rectifier based on Pose Distribution Bias')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='test')
    parser.add_argument('--bs',type=int, default=32)
    parser.add_argument('--vps', type=bool, default=False)
    parser.add_argument('--imu', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--gpuids', type=int, default=0, help='IDs of GPUs to use')
    parser.add_argument('--dataset', type=str, default='None', help='dataset to test')
    parser.add_argument('--imshow', type=bool, default=False, help='if display predicted depth images in depth_net mode')
    parser.add_argument('--save_video', type=bool, default=False, help='if save tilted video in depth_net or sr_only or sr_depth_net_full mode')
    parser.add_argument('--time', type=bool, default=False, help='process time eval')
    parser.add_argument('--cpu', type=bool, default=False, help='process cpu use rate in program')
    parser.add_argument('--bin', type=bool, default=False, help='output binned eval error')
    args = parser.parse_args()
    cfg = Config(args.cfg, create_tb_dirs=False)
    train(cfg, args)
