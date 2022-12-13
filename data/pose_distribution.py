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






def data_loader(datadir, pkl_file):
    data_info = pickle.load(open(pkl_file, 'rb'))
    modelist = ['train']#'test', 'val'

    gravity_list = []
    for mode in modelist:
        for idx in tqdm(range(len(data_info[mode][0])), desc='Progress'):
            # rgb_path_sampled = data_info[mode][0][idx]
            # depth_path_sampled = data_info[mode][1][idx]
            pose_path_sampled = data_info[mode][2][idx]
            gravity_path_sampled = pose_path_sampled.replace('pose', 'gravity-dir')
            # rgb_path = os.path.join(datadir, rgb_path_sampled)
            # depth_path = os.path.join(datadir, depth_path_sampled)
            # pose_path = os.path.join(datadir, pose_path_sampled)
            gravity_path = os.path.join(datadir, gravity_path_sampled)
            gravity_array = np.loadtxt(gravity_path, dtype=np.float32)
            gravity_tensor = torch.tensor(gravity_array, dtype=torch.float32)
            gravity_tensor = F.normalize(gravity_tensor, dim=-1, p=2)
            if torch.any(np.isinf(gravity_tensor)):
                print('Gravity Dir contains inf : {}'.format(gravity_path))
            gravity_list.append(gravity_tensor)
            if idx > 10:
                break
        print(mode + 'loading has ended.')
    gravity_tensor_all = torch.cat(gravity_list).reshape(len(gravity_list), *gravity_list[0].shape)
    #print(gravity_tensor_all.shape)
    m = torch.distributions.von_mises.VonMises(loc=gravity_tensor_all,concentration=1)
    print(m.sample())


def distributions_to_directions(x):
  """Convert spherical distributions from the DirectionNet to directions."""
  distribution_pred = spherical_normalization(x)
  expectation = spherical_expectation(distribution_pred)
  expectation_normalized = tf.nn.l2_normalize(expectation, axis=-1)
  return expectation_normalized, expectation, distribution_pred










if __name__ == '__main__':
    train_test_split = './my_scannet_standard_train_test_val_split.pkl'
    datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
    data_loader(datadir, train_test_split)






