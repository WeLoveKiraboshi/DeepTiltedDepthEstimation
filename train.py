import numpy as np
import argparse
import time
import datetime
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import pickle
# for val
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchsummary import summary

from models.warping_2dof_alignment import Warping2DOFAlignment
from dataloader.data import create_dataset_loader, data_augmentation, OpticalConverter

from models.spatial_rectifier_networks import *
from models.DenseDepthNet import DenseDepthModel
from models.ResnetUnet import ResnetUnetHybrid
from models.ResnetUnetGatedConv import ResnetUnetHybridGatedConv
from models.ResnetUnetPartialConv_v1 import ResnetUnetHybridPartialConv_v1
from models.ResnetUnetPartialConv_v2 import ResnetUnetHybridPartialConv_v2
from models.ResnetUnetPartialConv_v3 import ResnetUnetHybridPartialConv_v3
from models.UnetVGGPartialConvolution import PConvUNet
from models.UnetPatrialConv import PConvUNet

from config_loader import Config
from utils.tb_logger import Logger
from utils.utils import AverageMeter, check_nan_ckpt
from loss.sr_loss import SRonly_Loss, SRonly_Loss_
from loss.DenseDepth_loss import DenseDepthLoss, DenseDepthLoss_
from loss.sr_DenseDepth_full_loss import SR_DenseDepth_Full_Loss
from utils.torch import to_cpu

from utils.save_tensor_im import saving_gravity_tensor_to_file, draw_gravity_dir
import matplotlib.pyplot as plt
from matplotlib import cm

torch.cuda.empty_cache()

def system_setup(cfg, args):
    """setup"""
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.cuda.memory_summary(device=None, abbreviated=False)

    if args.gpus == 2:
        gpus = (0, 1)
        device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
    elif args.gpus == 1:
        device = torch.device('cuda', index=int(args.gpuids)) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuids)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tb_logger = Logger(log_dir=cfg.tb_dir, network=cfg.network, name=None)

    if cfg.sr_checkpoint_path != None:
        assert '.p' in cfg.sr_checkpoint_path
        model_sr_cp = pickle.load(open(cfg.sr_checkpoint_path, "rb"))
    if cfg.checkpoint_path != None:
        assert '.p' in cfg.checkpoint_path
        model_depthnet_cp = pickle.load(open(cfg.checkpoint_path, "rb"))


    if cfg.network == 'SpatialRectifier':
        model = SpatialRectifier(in_channels=3, out_channel=3, is_dropout=True, drop_out=cfg.dropout_p)
        if cfg.sr_checkpoint_path != None and cfg.init_train==False:
            model.load_state_dict(model_sr_cp[cfg.mode], strict=False)
            print("=> loading SR model '{}'".format(cfg.sr_checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
    elif cfg.network == 'ResnetUnet':
        model = ResnetUnetHybrid(pretrained=True).cuda()
        if cfg.checkpoint_path != None and cfg.init_train==False:
            #model_depthnet_cp = pickle.load(open(cfg.checkpoint_path, "rb"))
            model.load_state_dict(model_depthnet_cp[cfg.mode], strict=False)
            print("=> loading DepthNet model '{}'".format(cfg.checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
    elif cfg.network == 'ResnetUnetPartialConv_v1':
        model = ResnetUnetHybridPartialConv_v1().cuda()
        if cfg.checkpoint_path != None and cfg.init_train == False:
            model.load_state_dict(model_depthnet_cp[cfg.mode], strict=False)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.full_checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
        #load_pretrained(model)
    elif cfg.network == 'ResnetUnetPartialConv_v2':
        model = ResnetUnetHybridPartialConv_v2(pretrained=True).cuda()
        if cfg.checkpoint_path != None and cfg.init_train == False:
            model.load_state_dict(model_depthnet_cp[cfg.mode], strict=False)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.full_checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
    elif cfg.network == 'ResnetUnetPartialConv_v3':
        model = ResnetUnetHybridPartialConv_v3(pretrained=True).cuda()
        if cfg.checkpoint_path != None and cfg.init_train == False:
            model.load_state_dict(model_depthnet_cp[cfg.mode], strict=False)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
    elif cfg.network == 'ResnetUnetGatedConv':
        model = ResnetUnetHybridGatedConv(pretrained=True).cuda()
        if cfg.checkpoint_path != None and cfg.init_train == False:
            model.load_state_dict(model_depthnet_cp[cfg.mode], strict=False)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.checkpoint_path))
            if cfg.start_epochs == 0:
                print('Error: config var cfg.start_epochs is not set as previous epoch size. cfg.start_epochs={}'.format(cfg.start_epochs))
                exit(0)
    elif cfg.network == 'UnetPartialConv':
        model = PConvUNet()
    elif cfg.network == 'SpatialRectifierResnetUnet':
        model = SpatialRectifierResnetUnet(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path, sr_cnn_ckpt=cfg.sr_checkpoint_path, mode='train', dataset=cfg.dataset, pad_mode=cfg.image_padding_mode)
        print("=> loading Full_SR_DepthNet model '{}'".format(cfg.checkpoint_path))
        if cfg.full_checkpoint_path != None:
            assert os.path.isfile(cfg.full_checkpoint_path), \
                "=> no model found at '{}'".format(cfg.full_checkpoint_path)
            print("=> loading model '{}'".format(cfg.full_checkpoint_path))
            model_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
            model.load_state_dict(model_cp['sr_depth_net'], strict=False)
            if cfg.init_train:
                print('cfg.init train = True !')
                exit(0)
    elif cfg.network == 'SpatialRectifierResnetUnetPartialConv_v2':
        model = SpatialRectifierResnetUnetPartialConv_v2(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path, sr_cnn_ckpt=cfg.sr_checkpoint_path,mode='train', dataset=cfg.dataset)
        print("=> loading Full_SR_DepthNet model '{}'".format(cfg.full_checkpoint_path))
    elif cfg.network == 'SpatialRectifierResnetUnetPartialConv_v3':
        model = SpatialRectifierResnetUnetPartialConv_v3(K=cfg.K, depth_estimation_cnn_ckpt=cfg.checkpoint_path, sr_cnn_ckpt=cfg.sr_checkpoint_path,mode='train', dataset=cfg.dataset)
        if cfg.full_checkpoint_path != None and cfg.init_train == False:
            model_full_cp = pickle.load(open(cfg.full_checkpoint_path, "rb"))
            model.load_state_dict(model_full_cp[cfg.mode], strict=False)
            print("=> loading Full_SR_DepthNet model '{}'".format(cfg.full_checkpoint_path))
    else:
        """network"""
        print('Unrecognized network is indicated. please re-check network setting')


    # from torchsummary import summary
    # if cfg.network == 'ResnetUnetPartialConv' or cfg.network == 'UnetPartialConv':
    #     summary(model, [(3,240,320), (3, 240, 320)])
    # else:
    #     summary(model, (3, 240, 320))

    print('* Image padding mode : {}'.format(cfg.image_padding_mode))


    torch.backends.cudnn.benchmark = True
    if args.gpus == 2:
        model = torch.nn.DataParallel(model)  # make parallel
    elif args.gpus == 1:
        model.to(device)


    if cfg.loss == "SRLoss":
        loss_criteria = SRonly_Loss(mode='optimize')
    elif cfg.loss == "DenseDepthLoss":
        loss_criteria = DenseDepthLoss(w_l1=0.1, w_ssim=1.0, w_grad=0.0)
    elif cfg.loss == "SRDenseDepthFullLoss":
        loss_criteria = SR_DenseDepth_Full_Loss(w_net=cfg.loss_net_w , w_pose=cfg.loss_pose_w , w_l1=0.1, w_ssim=1.0, w_grad=0.0, pose_mode='optimize')

    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)

    return model, optimizer, loss_criteria, tb_logger


# Step 5. Create warper input & set intrinsic param:
warper = Warping2DOFAlignment()

# Step 6. Learning loop
best_avg_error = None


def run_epoch(dataset, model, optimizer, loss_criteria, tb_logger, mode='train', epoch=0, K=None, args=None):
    global warper, best_avg_error
    if mode == 'train':
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    if cfg.mode == 'sr_depth_net':
        losses_sr = AverageMeter()
        losses_net = AverageMeter()
    N = len(dataset)
    end = time.time()

    opt = OpticalConverter(train_dataset='scannet', test_dataset=cfg.dataset)

    for iter, sample_batched in enumerate(dataset):
        for data_key, data_value in sample_batched.items():
            if torch.is_tensor(data_value):
                sample_batched[data_key] = sample_batched[data_key].cuda()

        output_prediction = {'pred_depth': {}, 'I_g': {}, 'I_a': {}}

        # apply random roll rotation
        if sample_batched['ga_split'] != 'no_ga':
            sample_batched = data_augmentation(sample_batched, cfg, warper, epoch, iter, mode)

        if cfg.mode == 'sr_only':
            # saving_gravity_tensor_to_file(sample_batched['image'][0], sample_batched['gravity'][0], '0.png')
            pred_gravity = model.forward(sample_batched['image'])
            output_prediction['I_g'] = pred_gravity
            loss = loss_criteria(output_prediction, sample_batched)
            losses.update(loss.data.item(), sample_batched['image'].size(0))
            if args.imshow:
                for ii in range(sample_batched['image'].shape[0]):
                    rgb_frame_augmented = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)[:, :, [2, 1, 0]]
                    pred_grav_augmented = torch.nn.functional.normalize(sample_batched['gravity'][ii].unsqueeze(dim=0),dim=1, eps=1e-6).to('cpu').detach().numpy().copy()
                    aligned_augmented = torch.nn.functional.normalize(sample_batched['aligned_directions'][ii].unsqueeze(dim=0), dim=1, eps=1e-6).to('cpu').detach().numpy().copy()
                    rgb_frame_augmented = draw_gravity_dir(im=np.uint8(rgb_frame_augmented * 255).copy(),grav=pred_grav_augmented, align=aligned_augmented, K=K)
                    plt.imshow(rgb_frame_augmented)
                    plt.show()
        elif cfg.mode == 'depth_net':
            if cfg.network == 'ResnetUnetPartialConv_v1' or cfg.network == 'ResnetUnetPartialConv_v2' or cfg.network == 'UnetPartialConv' or cfg.network == 'ResnetUnetPartialConv_v3':
                pred_depth, pred_mask = model.forward(sample_batched['image'],sample_batched['rgb_mask'])
                output_prediction['pred_depth'] = pred_depth
                if args.imshow:
                    rgb_frame_augmented = sample_batched['image'][0].detach().cpu().permute(1, 2, 0)
                    rgb_mask_augmented = sample_batched['mask'][0].detach().cpu().permute(1, 2, 0)
                    pred_mask_frame_rectified = pred_mask[0].detach().cpu().squeeze()
                    pred_mask_frame_rectified = pred_mask_frame_rectified / torch.amax(pred_mask_frame_rectified)
                    pred_mask_frame_rectified_cmap = torch.Tensor(cm.jet(pred_mask_frame_rectified))[:, :, :3]

                    pred_mask_frame_rectified = pred_mask[0].detach().cpu().squeeze()
                    pred_mask_frame_rectified = pred_mask_frame_rectified / torch.amax(pred_mask_frame_rectified)
                    pred_mask_frame_rectified_cmap = torch.Tensor(cm.jet(pred_mask_frame_rectified))[:, :, :3]

                    pred_depth_frame_rectified = output_prediction['pred_depth'][0].detach().cpu().squeeze()
                    pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                    pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]

                    hcat_frames_rectified = torch.cat(
                        [rgb_frame_augmented, rgb_mask_augmented, pred_depth_frame_rectified_cmap], axis=1)
                    plt.imshow(hcat_frames_rectified)
                    plt.show()
            else:
                pred_depth = model.forward(sample_batched['image'])
                output_prediction['pred_depth'] = pred_depth
                #print('pred_depth = {}, gt_depth = {}, image = {}, mask = {}'.format(output_prediction['pred_depth'].shape, sample_batched['depth'].shape,sample_batched['image'].shape,sample_batched['mask'].shape))
                #output_prediction['pred_depth'] = opt.scale_refine(output_prediction['pred_depth'])
                if args.imshow:
                    for ii in range(sample_batched['image'].shape[0]):
                        rgb_frame_input = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)
                        pred_depth_frame_rectified = output_prediction['pred_depth'][ii].detach().cpu().squeeze()
                        pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                        pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]

                        gt_depth_frame_rectified = sample_batched['depth'][ii].detach().cpu().squeeze()
                        gt_depth_frame_rectified = gt_depth_frame_rectified / torch.amax(gt_depth_frame_rectified)
                        gt_depth_frame_rectified_cmap = torch.Tensor(cm.jet(gt_depth_frame_rectified))[:, :, :3]

                        hcat_frames_rectified = torch.cat(
                            [rgb_frame_input, pred_depth_frame_rectified_cmap, gt_depth_frame_rectified_cmap],
                            axis=1).numpy().copy()
                        plt.imshow(hcat_frames_rectified)
                        plt.show()
                        break

            loss = loss_criteria(output_prediction, sample_batched)
            losses.update(loss.data.item(), sample_batched['image'].size(0))
        elif cfg.mode == 'sr_depth_net':
            pred_sample_batched = model.forward(sample_batched)

            if cfg.loss == 'SRDenseDepthFullLoss':
                loss, sr_loss, net_loss = loss_criteria(pred_sample_batched, sample_batched)
                losses.update(loss.data.item(), sample_batched['image'].size(0))
                losses_sr.update(sr_loss.data.item(), sample_batched['image'].size(0))
                losses_net.update(net_loss.data.item(), sample_batched['image'].size(0))
                #print('sr loss : {}  net loss : {}'.format(sr_loss.data.item(), net_loss.data.item()))
            elif cfg.loss == 'SRLoss':
                loss = loss_criteria(pred_sample_batched, sample_batched)
                losses.update(loss.data.item(), sample_batched['image'].size(0))
            else:
                loss = loss_criteria(pred_sample_batched, sample_batched)
                losses.update(loss.data.item(), sample_batched['image'].size(0))

            if args.imshow:
                for ii in range(sample_batched['image'].shape[0]):
                    # pred_g = output_prediction['I_g']
                    # gt_g = sample_batched['gravity']
                    # a = sample_batched['aligned_directions']
                    # pred_inner = torch.bmm(pred_g.unsqueeze(1), a.unsqueeze(2)).squeeze()
                    # pred_mul_norms = torch.linalg.norm(pred_g, dim=1) * torch.linalg.norm(a, dim=1)
                    # pred_vector_angles = torch.acos(pred_inner / pred_mul_norms)
                    # gt_inner = torch.bmm(gt_g.unsqueeze(1), a.unsqueeze(2)).squeeze()
                    # gt_mul_norms = torch.linalg.norm(gt_g, dim=1) * torch.linalg.norm(a, dim=1)
                    # gt_vector_angles = torch.acos(gt_inner / gt_mul_norms)
                    # print('scane: {}, pred: {}, gt: {}'.format(
                    #     'scsns/scene/' + str(int(sample_batched['scene'][ii].detach().cpu())),
                    #     torch.rad2deg(pred_vector_angles)[ii], torch.rad2deg(gt_vector_angles)[ii]))

                    rgb_frame_augmented = sample_batched['image'][ii].detach().cpu().permute(1, 2, 0)[:,:,[2,1,0]]
                    pred_grav_augmented = torch.nn.functional.normalize(sample_batched['gravity'][ii].unsqueeze(dim=0), dim=1, eps=1e-6).to('cpu').detach().numpy().copy()
                    aligned_augmented = torch.nn.functional.normalize(sample_batched['aligned_directions'][ii].unsqueeze(dim=0),dim=1, eps=1e-6).to('cpu').detach().numpy().copy()
                    rgb_frame_augmented = draw_gravity_dir(im=np.uint8(rgb_frame_augmented*255).copy(), grav=pred_grav_augmented, align=aligned_augmented, K=K)
                    rgb_frame_rectified = pred_sample_batched['W_I'][ii].detach().cpu().permute(1, 2, 0)[:,:,[2,1,0]]

                    mask_frame_augmented = sample_batched['mask'][ii].detach().cpu() #.permute(1, 2, 0)
                    output_prediction_pred_depth_opt_scaled = opt.scale_refine(pred_sample_batched['depth'][ii])
                    pred_depth_frame_rectified = output_prediction_pred_depth_opt_scaled.detach().cpu() * mask_frame_augmented
                    pred_depth_frame_rectified = pred_depth_frame_rectified.squeeze()
                    pred_depth_frame_rectified = pred_depth_frame_rectified / torch.amax(pred_depth_frame_rectified)
                    pred_depth_frame_rectified_cmap = torch.Tensor(cm.jet(pred_depth_frame_rectified))[:, :, :3]

                    sample_batched_opt_scaled = opt.scale_refine(sample_batched['depth'][ii])
                    gt_depth_frame = sample_batched_opt_scaled.detach().cpu().squeeze()
                    gt_depth_frame = gt_depth_frame / torch.amax(gt_depth_frame)
                    gt_depth_frame_cmap = torch.Tensor(cm.jet(gt_depth_frame))[:, :, :3]


                    hcat_frames_rectified = torch.cat(
                        [rgb_frame_augmented, rgb_frame_rectified, pred_depth_frame_rectified_cmap, gt_depth_frame_cmap], axis=1)  #pred_depth_frame_rectified_cmap
                    plt.imshow(hcat_frames_rectified)
                    plt.show()


        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            del loss
            # if cfg.mode == 'sr_depth_net':
            #     del sr_loss, net_loss
            optimizer.step()


            #print(model.state_dict()['warp_params_cnn.warping_params_output.3.weight']) #conv1.weight
            # check if model contains nan value
            if check_nan_ckpt(model):
                for name, param in model.named_parameters():
                    if torch.sum(torch.isnan(param.data)):
                        print(name)
                latest_model_path = os.path.join(cfg.model_dir, 'model-latest.pkl')
                model_cp = pickle.load(open(latest_model_path, "rb"))
                model.load_state_dict(model_cp['model'])
                optimizer.load_state_dict(model_cp['optimizer'])
                print('Getting Nan iter:[{}]  reloading model from last checkpoint {}'.format(iter,
                                                                                              latest_model_path))




        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - iter))))

        # Log progress
        niter = epoch * N + iter
        if iter % 50 == 0:
            # Print to console
            print('{mode} Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'ETA {eta}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, iter, N, batch_time=batch_time, loss=losses, eta=eta, mode=mode))
        if iter % cfg.tb_log_interval == 0:
            tb_logger.scalar_summary(cfg.loss + '_' + mode, losses.val, niter)
            if cfg.mode == 'sr_depth_net':
                tb_logger.scalar_summary('sr_' + mode, losses_sr.val, niter)
                tb_logger.scalar_summary('depth_net_' + mode, losses_net.val, niter)
            if iter == 0 and mode == 'train' and cfg.mode != 'sr_only':
                tb_logger.LogProgressImage(model, sample_batched, epoch)

        if cfg.save_model_interval > 0 and mode == 'train' and niter % cfg.save_model_interval == 0:   #save latest model
            # if cfg.mode != 'sr_depth_net':
            #     with to_cpu(model):
            #         model_path = '%s/iter_%04d.pkl' % (cfg.model_dir, niter)
            #         model_cp = {cfg.mode: model.state_dict()}
            #         pickle.dump(model_cp, open(model_path, 'wb'))
            # else:
            with to_cpu(model):  # save latest model, instead of saving every iter
                model_path = '%s/model-latest.pkl' % (cfg.model_dir)
                model_cp = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                pickle.dump(model_cp, open(model_path, 'wb'))
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        del sample_batched, output_prediction

    if mode == 'val':  # check the best model for every val epochs
        # save the best checkpoint (except train_SR_only as we don't evaluate it)
        current_avg_error = losses.avg  # np.median(total_normal_errors)
        if best_avg_error is None:
            best_avg_error = current_avg_error
            print('Best Avg error in validation: %f, saving best checkpoint epoch %d, iter %d' % (
                best_avg_error, epoch, iter))
            path = os.path.join(cfg.model_dir, 'model-best_' + cfg.train_dataset_split_id + '.pkl')
            model_cp = {cfg.mode: model.state_dict()}
            pickle.dump(model_cp, open(path, 'wb'))
        else:
            if current_avg_error < best_avg_error:
                best_avg_error = current_avg_error
                print('Best Avg error in validation: %f, saving the best checkpoint, epoch %d, iter %d' % (
                    best_avg_error, epoch, iter))
                path = os.path.join(cfg.model_dir, 'model-best_' + cfg.train_dataset_split_id + '.pkl')
                model_cp = {cfg.mode: model.state_dict()}
                pickle.dump(model_cp, open(path, 'wb'))



def train(cfg, args):
    if cfg.dataset == 'scannet' or cfg.dataset == 'demo_dataset':
        K = np.array([[577.87061 * 0.5, 0., 319.87654 * 0.5], [0, 577.87061 * 0.5, 239.87603 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'OurDataset':
        K = np.array([266.82119, 0., 319.87654 * 0.5], [0, 266.82119, 239.87603 * 0.5], [0., 0., 1.], dtype=np.float32)
    elif cfg.dataset == 'OurDataset-limited':
        K = np.array([[266.82119, 0., 319.87654 * 0.5], [0, 266.82119, 239.87603 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei1rpy':
        K = np.array([[517.3 * 0.5, 0., 318.6 * 0.5], [0, 516.5 * 0.5, 255.3 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei2rpy':  # 520.9	521.0	325.1	249.7	0.2312	-0.7849	-0.0033	-0.0001	0.9172
        K = np.array([[520.9 * 0.5, 0., 325.1 * 0.5], [0, 521.0 * 0.5, 249.7 * 0.5], [0., 0., 1.]], dtype=np.float32)
    elif cfg.dataset == 'TUMrgbd_frei3rpy':  # 535.4	539.2	320.1	247.6	0	0	0	0	0
        K = np.array([[535.4 * 0.5, 0., 320.1 * 0.5], [0, 539.2 * 0.5, 247.6 * 0.5], [0., 0., 1.]], dtype=np.float32)
    cfg.K = K
    model, optimizer, loss_criteria, tb_logger = system_setup(cfg, args)
    dataloader_train, dataloader_test, dataloader_val = create_dataset_loader(cfg)


    for epoch in range(cfg.start_epochs, cfg.start_epochs+cfg.epochs, 1):
        run_epoch(dataloader_train, model, optimizer, loss_criteria, tb_logger, mode='train', epoch=epoch, K=K, args=args)
        with torch.no_grad():
            run_epoch(dataloader_val, model, optimizer, loss_criteria, tb_logger, mode='val', epoch=epoch, K=K, args=args)

    print('Finished Training epoch:{}. model saved '.format(cfg.epochs))

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Project-Kajita Multi camera switching via tobbi sensor using VAE')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--gpuids', type=int, default=0, help='IDs of GPUs to use')
    parser.add_argument('--imshow', type=bool, default=False, help='is imshow')
    #parser.add_argument('--start-epoch', type=int, default=0, help='start epochs. please use when you train model with any ckpt file')
    args = parser.parse_args()
    cfg = Config(args.cfg, create_tb_dirs=True)
    train(cfg, args)
