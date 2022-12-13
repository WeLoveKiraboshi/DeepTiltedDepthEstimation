import numpy as np
import pickle
import os

train_test_split='./scannet_standard_train_test_val_split.pkl'
write_file_name = './my_scannet_standard_train_test_val_split.pkl'
data_info = pickle.load(open(train_test_split, 'rb'))

train_data = data_info['train']
rgb_train_data = data_info['train']
test_data = data_info['test']
val_data = data_info['val']


datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
modelist = ['train', 'test', 'val']

WriterList = {'train': [[], [], []], 'test': [[], [], []], 'val': [[], [], []]} # rgb and depth and pose
for mode in modelist:
    for sample in data_info[mode][0]:
        subdir = sample[:12]
        orig_filename = sample[13:]
        id = int(orig_filename[6:12])
        #print('subdir = {}, id = {}'.format(subdir, id))
        rgb_path_sampled = os.path.join(subdir, 'color', str(id)+'.jpg')
        depth_path_sampled = os.path.join(subdir, 'depth', str(id)+'.png')
        pose_path_sampled = os.path.join(subdir, 'pose', str(id)+'.txt')
        rgb_path = os.path.join(datadir, rgb_path_sampled)
        depth_path = os.path.join(datadir, depth_path_sampled)
        pose_path = os.path.join(datadir, pose_path_sampled)
        if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_path):
            WriterList[mode][0].append(rgb_path_sampled)
            WriterList[mode][1].append(depth_path_sampled)
            WriterList[mode][2].append(pose_path_sampled)
        else:
            pass
            # print('file does not exists... {}'.format(rgb_path))
            # exit(0)



import pickle
with open(write_file_name, 'wb') as handle:
    pickle.dump(WriterList, handle, protocol=pickle.HIGHEST_PROTOCOL)
