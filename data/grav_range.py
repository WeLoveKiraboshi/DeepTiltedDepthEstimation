import os, sys
import random, time, copy
import argparse
import torch

import numpy as np
import torch
import glob
import math
import cv2

def rotation_matrix_from_vectors(vec1, vec2):
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
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



#my_scannet_standard_train_test_val_split.pkl
#my_full_2dofa_scannet.pkl

if __name__ == '__main__':
	pkl_file = './my_scannet_standard_train_test_val_split.pkl'
	data_info = pickle.load(open(pkl_file, 'rb'))
	modelist = ['train']  # 'test', 'val'

	pitch_list = []
	roll_list = []
	for mode in modelist:
		for idx in tqdm(range(len(data_info[mode][0])), desc='Progress'):
			pose_path_sampled = data_info[mode][2][idx]
			gravity_path_sampled = pose_path_sampled.replace('pose', 'gravity-dir')
			gravity_path = os.path.join(datadir, gravity_path_sampled)
			grav = np.loadtxt(path)
			Rot_normal = rotation_matrix_from_vectors(np.array([0, 1, 0]), grav)
			rotVec, _ = cv2.Rodrigues(Rot_normal)
			yaw = rotVec[1]
			pitch = rotVec[0]
			theta = rotVec[2]
			# print('Calculated from floor detection Gravity pitch = {}, yaw = {}, roll = {}'.format(
			# 	math.degrees(pitch),
			# 	math.degrees(yaw),
			# 	math.degrees(theta)))
			pitch_list.append(math.degrees(pitch))
			roll_list.append(math.degrees(theta))

		print('--- {} range pitch ----'.format(mode))
		pitch_list_array = np.array(pitch_list)
		roll_list_array = np.array(roll_list)
		print('pitch : max_angle = {}  deg.    min_angle = {} deg.'.format(np.amax(pitch_list_array),
																		   np.amin(pitch_list_array)))
		print(' roll : max_angle = {}  deg.    min_angle = {} deg.'.format(np.amax(roll_list_array),
																		   np.amin(roll_list_array)))

