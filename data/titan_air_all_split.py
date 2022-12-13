import numpy as np
import pickle
import os




datadir = '/media/yukisaito/ssd2/Titan-air'
write_file_name = './titan_air_hospital_easy'
modelist = ['pitch']
seqlist = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6']

WriterList = {'test': [[], [], []]} # rgb and depth and pose
for mode in modelist:
    for seq in seqlist:
        SEQ_DIR = os.path.join(datadir, mode, seq)
        associate_path = os.path.join(SEQ_DIR, 'associate.txt')
        f = open(associate_path)
        line = f.readline()
        while line:
            rgbpath = line.strip().split()[1]
            depthpath = line.strip().split()[3]
            rgb_path_sampled = os.path.join(SEQ_DIR, rgbpath)
            depth_path_sampled = os.path.join(SEQ_DIR, depthpath)
            pose_path_sampled = rgb_path_sampled.replace('rgb', 'gravity-dir').replace('png', 'txt')
            print(rgb_path_sampled, depth_path_sampled)
            if os.path.exists(rgb_path_sampled) and os.path.exists(depth_path_sampled) and os.path.exists(pose_path_sampled):
                WriterList['test'][0].append(rgb_path_sampled)
                WriterList['test'][1].append(depth_path_sampled)
                WriterList['test'][2].append(pose_path_sampled)
            line = f.readline()
        f.close()

        #print('subdir = {}, id = {}'.format(subdir, id))
        # rgb_path_sampled = os.path.join(subdir, 'color', str(id)+'.jpg')
        # depth_path_sampled = os.path.join(subdir, 'depth', str(id)+'.png')
        # pose_path_sampled = os.path.join(subdir, 'gravity-dir', str(id)+'.txt')
        # rgb_path = os.path.join(datadir, rgb_path_sampled)
        # depth_path = os.path.join(datadir, depth_path_sampled)
        # pose_path = os.path.join(datadir, pose_path_sampled)
        # if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_path):
        #     WriterList[mode][0].append(rgb_path_sampled)
        #     WriterList[mode][1].append(depth_path_sampled)
        #     WriterList[mode][2].append(pose_path_sampled)
        # else:
        #     pass
        #     # print('file does not exists... {}'.format(rgb_path))
        #     # exit(0)



import pickle
with open(write_file_name, 'wb') as handle:
    pickle.dump(WriterList, handle, protocol=pickle.HIGHEST_PROTOCOL)
