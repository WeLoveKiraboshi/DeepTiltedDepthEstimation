import numpy as np
import pickle
import os

train_test_split='./full_2dofa_scannet.pkl'
write_file_name = './my_full_2dofa_scannet.pkl'
data_info = pickle.load(open(train_test_split, 'rb'))

train_data = data_info['train']
test_data = data_info['test']




exit(0)
datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
modelist = ['train', 'test']
e_mode_list = ['e2', '-e2']


WriterList = {'train': {'with_ga': {'e2': [[], [], []], '-e2': [[], [], []]}, 'no_ga': [[], [], []]}, 'test': {'with_ga': {'e2': [[], [], []], '-e2': [[], [], []]}, 'no_ga': [[], [], []]}} # rgb and depth and pose
for mode in modelist:
    for e_mode in e_mode_list:
        for ga_file in data_info[mode]['with_ga'][e_mode]:
            subdir = ga_file[:12]
            #print('subdir', subdir)
            orig_filename =ga_file[13:]
            #print('filename', orig_filename)
            id = int(orig_filename[6:12])
            #print('id', id)
            rgb_path_sampled = os.path.join(subdir, 'color', str(id) + '.jpg')
            depth_path_sampled = os.path.join(subdir, 'depth', str(id) + '.png')
            pose_path_sampled = os.path.join(subdir, 'pose', str(id) + '.txt')
            rgb_path = os.path.join(datadir, rgb_path_sampled)
            depth_path = os.path.join(datadir, depth_path_sampled)
            pose_path = os.path.join(datadir, pose_path_sampled)
            if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_path):
                WriterList[mode]['with_ga'][e_mode][0].append(rgb_path_sampled)
                WriterList[mode]['with_ga'][e_mode][1].append(depth_path_sampled)
                WriterList[mode]['with_ga'][e_mode][2].append(pose_path_sampled)
        print('record... with_ga_split')

    for noga_file in data_info[mode]['no_ga']:
        subdir = noga_file[:12]
        orig_filename = noga_file[13:]
        id = int(orig_filename[6:12])
        ##
        rgb_path_sampled = os.path.join(subdir, 'color', str(id) + '.jpg')
        depth_path_sampled = os.path.join(subdir, 'depth', str(id) + '.png')
        pose_path_sampled = os.path.join(subdir, 'pose', str(id) + '.txt')
        rgb_path = os.path.join(datadir, rgb_path_sampled)
        depth_path = os.path.join(datadir, depth_path_sampled)
        pose_path = os.path.join(datadir, pose_path_sampled)
        if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_path):
            WriterList[mode]['no_ga'][0].append(rgb_path_sampled)
            WriterList[mode]['no_ga'][1].append(depth_path_sampled)
            WriterList[mode]['no_ga'][2].append(pose_path_sampled)
    print('record... no_ga_split')





#import pickle
#with open(write_file_name, 'wb') as handle:
#    pickle.dump(WriterList, handle, protocol=pickle.HIGHEST_PROTOCOL)
