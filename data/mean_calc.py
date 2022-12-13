import numpy as np
import os
import cv2
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm







def data_loader(datadir, pkl_file):
    data_info = pickle.load(open(pkl_file, 'rb'))
    modelist = ['train', 'test', 'val']

    mean_r_list = []
    mean_g_list = []
    mean_b_list = []
    for mode in modelist:
        for idx in tqdm(range(len(data_info[mode][0])), desc='Progress'):
            rgb_path_sampled = data_info[mode][0][idx]
            rgb_path = os.path.join(datadir, rgb_path_sampled)
            color_img = cv2.resize(cv2.imread(rgb_path, cv2.IMREAD_COLOR), (320, 240), interpolation=cv2.INTER_AREA)
            mean_b = color_img[:, :, 0].mean()
            mean_g = color_img[:, :, 1].mean()
            mean_r = color_img[:, :, 2].mean()
            mean_b_list.append(mean_b)
            mean_g_list.append(mean_g)
            mean_r_list.append(mean_r)
        print(mode + 'loading has ended.')
    print('Mean value of train test val : pkl = {}'.format(pkl_file))
    print('mean b : {} mean g : {} mean r : {}'.format(np.mean(np.array(mean_b_list)), np.mean(np.array(mean_g_list)), np.mean(np.array(mean_r_list))))








if __name__ == '__main__':
    train_test_split = './my_scannet_standard_train_test_val_split.pkl'
    datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
    data_loader(datadir, train_test_split)






