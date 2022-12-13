import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm


train_test_split= './rectified_2dofa_scannet.pkl' #'./my_scannet_standard_train_test_val_split.pkl'

data_info = pickle.load(open(train_test_split, 'rb'))

print(data_info)


train_data = data_info['train']
rgb_train_data = data_info['train']
test_data = data_info['test']
val_data = data_info['val']

print('train_len = {}'.format(len(val_data[0])))


datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
modelist = ['train', 'test', 'val']
max_depth = 0
WriterList = {'train': [[], [], []], 'test': [[], [], []], 'val': [[], [], []]} # rgb and depth and pose
for mode in modelist:
    for idx in tqdm(range(len(data_info[mode][0])), desc='Progress'):
        rgb_path_sampled = data_info[mode][0][idx]
        depth_path_sampled = data_info[mode][1][idx]
        pose_path_sampled = data_info[mode][2][idx]
        gravity_path_sampled = pose_path_sampled.replace('pose', 'gravity-dir')
        rgb_path = os.path.join(datadir, rgb_path_sampled)
        depth_path = os.path.join(datadir, depth_path_sampled)
        pose_path = os.path.join(datadir, pose_path_sampled)
        gravity_path = os.path.join(datadir, gravity_path_sampled)
        P = np.loadtxt(gravity_path)
        if np.any(np.isinf(P)):
            print('Gravity Dir contains inf : {}'.format(pose_path_sampled))
        if not os.path.exists(gravity_path):
            print('File does not exist... {}'.format(gravity_path))


        # depth_img = cv2.resize(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32), (320, 240), interpolation=cv2.INTER_AREA)
        # max_depth_candidate = np.amax(depth_img)
        #
        # if max_depth < max_depth_candidate:
        #     max_depth = max_depth_candidate
        # if idx % 1000 == 0:
        #    print('[{}] iter:{} max_depth:{}'.format(mode, idx, max_depth))
    print(mode + 'has ended... ', max_depth)
print('max depth = ', max_depth)


