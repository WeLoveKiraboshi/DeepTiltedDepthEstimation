import numpy as np
import argparse
import time
import datetime
import os
import cv2
import sys
sys.path.append(os.getcwd())
import pickle
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn

from tqdm import tqdm
from lu_vp_detect import VPDetection

length_thresh = 60
principal_point = None
focal_length = 266.82119 * 2
seed = 1337


def testLineSegmentDetector(dataset_dir, file_path, vps_save_path, idx, args):
    print(vps_save_path)
    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
    vps = vpd.find_vps(file_path)
    vps = np.vstack([vps, -vps]).astype(np.float32) #get 6 * 3 dir
    # vps_2D = vpd.vps_2D
    # print(vps_2D)
    save_im_path = './LSD_ims/'+str(idx)+'.png'
    if args.save_im:
        vpd.create_debug_VP_image(show_image=args.imshow, save_image=save_im_path)
    if args.save_nvp:
        np.save(vps_save_path, vps)

    return 0



def test(args):
#'/home/yukisaito/TUMdataset/rgbd_dataset_freiburg1_rpy/'
    dataset_dir = '/home/yukisaito/mydataset/NFOV/pitch/seq2/'
    associate_path = '/home/yukisaito/mydataset/NFOV/pitch/seq2/associate.txt'
    if not os.path.exists(os.path.join(dataset_dir, 'vps')):
        os.mkdir(os.path.join(dataset_dir, 'vps'))
    f = open(associate_path)
    line = f.readline()
    data_info = [[], [], []]  # 0;rgb 1;depth 2;pose
    while line:
        rgbpath = line.strip().split()[1]
        depthpath = line.strip().split()[3]
        data_info[0].append(rgbpath)
        data_info[1].append(depthpath)
        line = f.readline()
    f.close()
    idx_list = [i for i in range(0, len(data_info[0]), 1)]
    for idx in idx_list:
        # data = np.load(os.path.join(dataset_dir, 'vps', str(idx)+'.npy'))
        # print(data)
        # print(data.shape)
        # exit(0)
        rgb_file_path = os.path.join(dataset_dir, data_info[0][idx])
        vps_file_path = os.path.join(dataset_dir, data_info[0][idx].replace('rgb', 'vps').replace('png', 'npy'))
        # if idx > 90:
        testLineSegmentDetector(dataset_dir, rgb_file_path, vps_file_path, idx, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tilted Depth Estimation via spatial rectifier based on Pose Distribution Bias')
    parser.add_argument('--imshow', type=bool, default=False)
    parser.add_argument('--save_nvp', type=bool, default=True)
    parser.add_argument('--save_im', type=bool, default=True)
    args = parser.parse_args()
    test(args)
