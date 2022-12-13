import skimage.io as sio
import torch
import numpy as np
import argparse
import os
import time

import cv2
import math

def limit_angle(x):
    #limit angle in [-pi, pi]
    if x > math.pi or x < -math.pi:
        x = x - int(x / math.pi) * math.pi
    return x


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
    Rterm_3 = kmat.dot(kmat) * ((1 - c) / (s ** 2))
    if ~np.any(np.isnan(Rterm_3)):
        rotation_matrix = np.eye(3) + kmat + Rterm_3
    else:
        rotation_matrix = np.eye(3) + kmat
    return rotation_matrix
def _convert_R_mat_to_vec(R_mat):
    """
    Use rodiriguez formula to convert R matrix to R vector with 3 DoF
    :return:
    """
    phi = np.arccos((np.trace(R_mat) - 1) / 2)

    R_vec = np.array([R_mat[2][1] - R_mat[1][2], R_mat[0][2] - R_mat[2][0], R_mat[1][0] - R_mat[0][1]])

    R_vec = R_vec * (phi / (2 * np.sin(phi)))

    # print("-------- R mat to vec-----------")
    # print(cv2.Rodrigues(R_mat)[0])
    # print("-------- R mat to vec-----------")

    return R_vec




def saving_gravity_tensor_to_file(rgb_tensor, path, is_pred_g=True,
                                  pred_g=None, is_gt_g = False, gt_g=None, K=np.eye(3), H=None):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255).copy()
    gravity_target = np.array([0, 1., 0], dtype=np.float32)
    centVec = (int(output_rgb_img.shape[1] / 2), int(output_rgb_img.shape[0] / 2))
    if is_pred_g:
        pred_g = pred_g.to('cpu').detach().numpy().copy()
        pred_g = pred_g / np.linalg.norm(pred_g)

        rotMat = rotation_matrix_from_vectors(pred_g, gravity_target)
        rotVec, _ = cv2.Rodrigues(rotMat)
        #rotVec = _convert_R_mat_to_vec(rotMat)
        yaw = rotVec[1]
        pitch = rotVec[0]
        theta = rotVec[2]

        #print('pitch = {}, yaw = {}, roll = {}'.format(math.degrees(pitch), math.degrees(yaw), math.degrees(theta)))
        rollVec = (int(100 * math.sin(theta) + output_rgb_img.shape[1] / 2),
                   int(100 * math.cos(theta) + output_rgb_img.shape[0] / 2))

        # u = (K[0,0]*pred_g[0]+K[0,2]*pred_g[2])/pred_g[2]
        # v = (K[1,1]*pred_g[1]+K[1,2]*pred_g[2])/pred_g[2]
        # rollVec = (u, v)
        # diffVec = np.array([rollVec[0]-centVec[0], rollVec[1]-centVec[1]])
        # diffVec = 100 * diffVec / np.linalg.norm(np.array(diffVec))
        # rollVec = (int(centVec[0]+diffVec[0]), int(centVec[1]+diffVec[1]))
        cv2.arrowedLine(output_rgb_img, centVec, rollVec, (255, 255, 0), thickness=5) #yellow
    if is_gt_g:
        gt_g = gt_g.to('cpu').detach().numpy().copy()
        gt_g = gt_g / np.linalg.norm(gt_g)
        u = (K[0, 0] * gt_g[0] + K[0, 2] * gt_g[2]) / gt_g[2]
        v = (K[1, 1] * gt_g[1] + K[1, 2] * gt_g[2]) / gt_g[2]
        rollVec = (u, v)
        diffVec = np.array([rollVec[0] - centVec[0], rollVec[1] - centVec[1]])
        diffVec = 100 * diffVec / np.linalg.norm(np.array(diffVec))
        rollVec = (int(centVec[0] + diffVec[0]), int(centVec[1] + diffVec[1]))
        cv2.arrowedLine(output_rgb_img, centVec, rollVec, (255, 0, 0), thickness=5)
    if is_pred_g or is_gt_g:
        if path != None:
            sio.imsave(path, output_rgb_img)
        else:
            cv2.imshow('Gravity_im', output_rgb_img)
            cv2.wairKey(10)
            return output_rgb_img


def draw_gravity_dir(im=None, grav=np.array([0, 1., 0], dtype=np.float32), align=np.array([0, 1., 0], dtype=np.float32), K=np.eye(3)):
    if len(im.shape) != 3 or im.shape[2] != 3:
        print('Shape invariant in draw_gravity_dir. {}'.format(im.shape))
        return
    if torch.is_tensor(im) or torch.is_tensor(grav) or torch.is_tensor(align):
        print('Invarid input of Torch Tensor in draw_gravity_dir')
        return
    rotMat = rotation_matrix_from_vectors(grav, align)
    rotVec, _ = cv2.Rodrigues(rotMat)
    yaw = rotVec[1]
    pitch = rotVec[0]
    theta = rotVec[2]
    #print('pitch = {}, yaw = {}, roll = {}'.format(math.degrees(pitch), math.degrees(yaw), math.degrees(theta)))
    rollVec = (int(100 * math.sin(theta) + im.shape[1] / 2),
               int(100 * math.cos(theta) + im.shape[0] / 2))
    centVec = (int(im.shape[1] / 2), int(im.shape[0] / 2))
    rollVec_aligned = (int(im.shape[1] / 2), int(im.shape[0] / 2) + 100)
    cv2.arrowedLine(im, centVec, rollVec, (255, 255, 0), thickness=5)  # light blue
    cv2.arrowedLine(im, centVec, rollVec_aligned, (255, 0, 255), thickness=5)  # yellow
    return torch.from_numpy(im.astype(np.float32) / 255).clone()



def saving_rgb_tensor_to_file(rgb_tensor, path):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255)
    if path != None:
        sio.imsave(path, output_rgb_img)
    else:
        return output_rgb_img


def saving_normal_tensor_to_file(normal_tensor, path):
    normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)
    output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)
    if path != None:
        sio.imsave(path, output_normal_img)
    else:
        return output_normal_img



def visualize_gravity_dir(rgb_tensor, path, is_pred_g=True, pred_g=None, is_gt_g = False, gt_g=None, K=np.eye(3), H=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pytransform3d import rotations as pr
    from pytransform3d import transformations as pt
    from pytransform3d.transform_manager import TransformManager

    random_state = np.random.RandomState(0)

    ee2robot = pt.transform_from_pq(
        np.hstack((np.array([0.4, -0.3, 0.5]),
                   pr.random_quaternion(random_state))))
    cam2robot = pt.transform_from_pq(
        np.hstack((np.array([0.0, 0.0, 0.8]), pr.q_id)))
    object2cam = pt.transform_from(
        pr.active_matrix_from_intrinsic_euler_xyz(np.array([0.0, 0.0, -0.5])),
        np.array([0.5, 0.1, 0.1]))

    tm = TransformManager()
    tm.add_transform("end-effector", "robot", ee2robot)
    tm.add_transform("camera", "robot", cam2robot)
    tm.add_transform("object", "camera", object2cam)

    ee2object = tm.get_transform("end-effector", "object")

    ax = tm.plot_frames_in("robot", s=0.1)
    ax.set_xlim((-0.25, 0.75))
    ax.set_ylim((-0.5, 0.5))
    ax.set_zlim((0.0, 1.0))
    plt.show()