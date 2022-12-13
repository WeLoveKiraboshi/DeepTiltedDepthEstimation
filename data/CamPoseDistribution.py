import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm
import math
import glob

train_test_split='./my_scannet_standard_train_test_val_split.pkl'
write_file_name = './my_scannet_standard_train_test_val_split.pkl'
data_info = pickle.load(open(train_test_split, 'rb'))

train_data = data_info['train']
rgb_train_data = data_info['train']
test_data = data_info['test']
val_data = data_info['val']


datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
modelist = ['train', 'test', 'val']
max_depth = 0

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / numpy.linalg.norm(vec1)).reshape(3), (vec2 / numpy.linalg.norm(vec2)).reshape(3)
    v = numpy.cross(a, b)
    if any(v): #if not all zeros then 
        c = numpy.dot(a, b)
        s = numpy.linalg.norm(v)
        kmat = numpy.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return numpy.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return numpy.eye(3) #cross of all zeros only occurs on identical directions

def rotation_matrix_from_vectors_v2(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rterm_3 = kmat.dot(kmat) * ((1 - c) / (s ** 2))
    if ~np.any(np.isnan(Rterm_3)):
        rotation_matrix = np.eye(3) + kmat + Rterm_3
    else:
        rotation_matrix = np.eye(3) + kmat
    return rotation_matrix
    
def calc_pose_dist():
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
            g = np.loadtxt(gravity_path) 
            g = g / np.linalg.norm(g)
            #rpy calc
            R = rotation_matrix_from_vectors_v2(g, np.array([0,1,0], dtype=np.float32))
            if np.isnan(R).any() or np.isinf(R).any():
                print(f'{rgb_path_sampled} contain inf or nan value')
                continue
            rotVec, _ = cv2.Rodrigues(R)
            yaw = rotVec[1]
            pitch = rotVec[0]
            theta = rotVec[2]
            print('pitch = {}, yaw = {}, roll = {}'.format(math.degrees(pitch), math.degrees(yaw), math.degrees(theta)))
        print(mode + ' has ended... ', max_depth)

def normalize(num, lower=0, upper=360, b=False):
    from math import floor, ceil
    res = num
    if not b:
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                             (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res
    
pitch_list = []
yaw_list = []
theta_list = []        
 
def scene_iter(idx, output_path, subdir):
    global pitch_list, yaw_list, theta_list
    init_pose = np.loadtxt(os.path.join(output_path, 'pose', '0.txt'))
    file_list = sorted(glob.glob(os.path.join(output_path, 'pose', '*.txt')))
    for filename in file_list:
        pose = np.loadtxt(filename) #cam to world
        inv_pose = np.linalg.inv(pose) # world to cam
        R = pose @ np.linalg.inv(init_pose)
        if np.isnan(R).any() or np.isinf(R).any():
                #print(f'{filename} contain inf or nan value')
                continue
        rotVec, _ = cv2.Rodrigues(R[:3, :3])
        yaw = normalize(math.degrees(rotVec[1]), lower=-180, upper=180, b=True)
        pitch = normalize(math.degrees(rotVec[0]), lower=-180, upper=180, b=True)
        theta = normalize(math.degrees(rotVec[2]), lower=-180, upper=180, b=True)
        #print('pitch = {}, yaw = {}, roll = {}'.format(pitch, yaw, theta))
        pitch_list.append(pitch)
        yaw_list.append(yaw)
        theta_list.append(theta)
    if idx == 706:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        np_pitch_array = np.array(pitch_list)
        np_yaw_array = np.array(yaw_list)
        np_theta_array = np.array(theta_list)
        plt.hist([np_pitch_array, np_yaw_array, np_theta_array], bins=20, range=(-180, 180), stacked=True,label=['roll', 'pitch', 'yaw'])
        plt.legend()
        ax.set_xlabel('rotation angle [degree]')
        ax.set_ylabel('sample num')
        plt.savefig("./ScanNetv2_hist_camerapose.png", bbox_inches = "tight")
        exit(0)

        
if __name__ == '__main__':
    global error_datalist, infpose_datalist
    data_dir = '/media/yukisaito/ssd2/ScanNetv2/scans'
    data_idxs = 707 #807 #707

    print('===========================================')
    for idx in range(0, data_idxs): #707 ~ 806
        for sub_idx in range(10):
            # if sub_idx == 0 or sub_idx == 1 or sub_idx == 2:
            #     continue
            subdir = "scene"+str(idx).zfill(4)+'_'+str(sub_idx).zfill(2) 
            output_path = os.path.join(data_dir, subdir)
            if os.path.exists(output_path):
                print('Data: {} is Loading....'.format(subdir))
                #apply floor Plane Detection in this scene
                scene_iter(idx, output_path, subdir)
