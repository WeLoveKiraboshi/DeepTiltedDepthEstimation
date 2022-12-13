import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm
import math
import glob
import csv

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

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
 
def scene_iter(datadir, subdir, gravity_path):
    global pitch_list, yaw_list, theta_list

    grav = np.loadtxt(gravity_path).astype(np.float32)
    align = np.array([0, 1., 0], dtype=np.float32)
    rotMat = rotation_matrix_from_vectors(align, grav)
    rotVec, _ = cv2.Rodrigues(rotMat)
    yaw = rotVec[1]
    pitch = rotVec[0]
    theta = rotVec[2]

    yaw = normalize(math.degrees(rotVec[1]), lower=-180, upper=180, b=True)
    pitch = normalize(math.degrees(rotVec[0]), lower=-180, upper=180, b=True)
    theta = normalize(math.degrees(rotVec[2]), lower=-180, upper=180, b=True)
    pitch_list.append(pitch)
    yaw_list.append(yaw)
    theta_list.append(theta)



        
if __name__ == '__main__':
    train_test_split = './my_dataset_pitch_split' #'./my_scannet_standard_train_test_val_split.pkl'
    data_info = pickle.load(open(train_test_split, 'rb'))

    datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'

    for mode in data_info.keys():
        for sample in tqdm(data_info[mode][0]):
            print(sample)
            subdir = sample[:12]
            if 'color' in sample:
                pose_path_sampled = os.path.join(datadir, sample.replace('color', 'gravity-dir').replace('jpg', 'txt'))
            elif 'rgb' in sample:
                pose_path_sampled = os.path.join(datadir, sample.replace('rgb', 'gravity-dir').replace('png', 'txt'))
            else:
                pose_path_sampled = sample
            scene_iter(datadir, subdir, pose_path_sampled)
    age_list = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    np_pitch_array = np.array(pitch_list)
    np_yaw_array = np.array(yaw_list)
    np_roll_array = np.array(theta_list)
    roll_hist, roll_data = np.histogram(np_roll_array, bins=age_list)
    pitch_hist, pitch_data = np.histogram(np_pitch_array, bins=age_list)
    print('roll = ', roll_hist)
    print('pitch = ', pitch_hist)










