import numpy as np
import os
from numba import jit
import numpy as np
import cv2
from scipy import optimize
from utils import _convert_R_mat_to_vec, _convert_R_vec_to_mat



class AlignmentDirectionOptimization:
    def __init__(self, data, near_thr=3):
        self.data = data
        self.near_thr = near_thr
        # self.num_imgs = len(images)
        # self.K = self.images[0].K
        # self.Rt = []
        # for im in self.images:
        #     Rt = np.concatenate([im.R, im.T], 1) #Rt is the mat(3, 4)
        #     self.Rt.append(Rt)

    @jit
    def near(p, pntList, d0):
        cnt = 0
        counter = 0
        for pj in pntList:
            counter += 1
            dist = np.linalg.norm(p - pj)
            if dist < d0:
                cnt += 1 - dist / d0
        # print('finish calc near vector within the sphere plots')
        return cnt


    @staticmethod
    def compute_residuals(x, data, frame, g, near_thr):
        """
        :param x: np array - variables to optimize
        0:5 (K -> u0, v0, alpha, beta, gamma)
        Then 3 entries of R ,followed by 3 entries for t for each img
        If radial dist is True, then followed by k1, k2 for each img
        x = [alpha, gamma, u0, beta, v0,  r1_1, r2_1, r3_1, t1_1, t2_1, t3_1, r1_2, r2_2, r3_2, t1_2, t2_2, t3_2......]
        :param img_hc: List of lists of all actual img pts
        :param world_hc: np array of rows of world pts
        :param radial_dist: bool, whether to take radial distortion into account
        :return:
        """
        score = AlignmentDirectionOptimization.near(x, data, near_thr)
        return -np.array([score, score, score])
        # num_corners = images[0].rows * images[0].cols
        # num_imgs = len(images)
        # K = np.zeros((3, 3))
        # #  Build K
        # K[0][0] = x[0]
        # K[0][1] = x[1]
        # K[0][2] = x[2]
        # K[1][1] = x[3]
        # K[1][2] = x[4]
        # K[2][2] = 1
        #
        # residual_sum = 0
        # img_hc_list = []
        # proj_crd_list = []
        #
        # for i in range(num_imgs):
        #     Rt_vec = x[5 + i * 6: 5 + (i + 1) * 6]
        #     R = _convert_R_vec_to_mat(Rt_vec[0:3])
        #     Rt = np.hstack((R, Rt_vec[3:].reshape(3, 1)))
        #     P = np.matmul(K, Rt)  # shape = 3, 4
        #     # Compute projections per image, per corner
        #     # world_hc = <num_corners> rows of x, y, z, w
        #     # image = list of nd arrays. Each nd array has rows of [x, y, z]
        #
        #     img_hc = np.concatenate([images[i].im_pts, np.ones((num_corners, 1))], 1).T  # 3, n_pts
        #     world_hc = np.concatenate([images[i].plane_pts, np.zeros((num_corners, 1)), np.ones((num_corners, 1))],
        #                               1).T  # 4, n_pts
        #
        #     proj_crd = np.matmul(P, world_hc)  # Proj_crd shape = # 3, n_pts
        #
        #     proj_crd = proj_crd / proj_crd[2, :]  # normalizing last crd
        #
        #
        #     #save result and make mat which contains all info
        #     proj_crd_list.append(proj_crd)
        #     img_hc_list.append(img_hc)
        #
        #     #diff = img_hc - proj_crd
        #     #ABS = np.mean(np.abs(diff))
        #     #RMSE = np.sqrt(np.mean(np.power(diff, 2)))
        #     #print('Image[1] : ReproError  abs = {},  rmse = {}'.format(ABS, RMSE))
        #
        # # compute residual
        # img_hc = np.array(img_hc_list)
        # proj_crd = np.array(proj_crd_list)
        # residual = img_hc.ravel() - proj_crd.ravel()  # 3, n_pts
        # return residual



    def _optimize(self, data, frame, g):
        print("---------------------------------------")
        print("solve optimized aligned dir vec")
        print('this may takes several seconds for computation')

        ##param : x = [x,y,z]
        num_params = 3
        x_init = np.zeros(num_params)
        x_init[0] = 0
        x_init[1] = 1
        x_init[2] = 0

        sol = optimize.least_squares(AlignmentDirectionOptimization.compute_residuals, x_init, args=([data, frame, g, self.near_thr]), method='lm',
                                     xtol=1e-15, ftol=1e-15)
        print(" Optimize ---x:{}, y:{}, z:{}".format(sol.x[0],sol.x[1],sol.x[2]))
        print("-----------------------------------------------------------------------------------------")

    @staticmethod
    def objective(x, *args):
        data=args[0]
        frame=args[1]
        g=args[2]
        near_thr=args[3]
        score = AlignmentDirectionOptimization.near(x, data, near_thr)


        return -score

    @staticmethod
    def norm_cons(x):
        return np.linalg.norm(x)-1

    def optimize(self, frame, g):
        print("---------------------------------------")
        print("solve optimized aligned dir vec")
        print('this may takes several seconds for computation')

        ##param : x = [x,y,z]
        num_params = 3
        x_init = np.zeros(num_params)
        x_init[0] = 0
        x_init[1] = 1
        x_init[2] = 0
        cons = (
            {'type': 'eq', 'fun': AlignmentDirectionOptimization.norm_cons},
        )

        result = optimize.minimize(AlignmentDirectionOptimization.objective, x0=x_init, constraints=cons, args=(self.data, frame, g, self.near_thr), method="SLSQP")
        x = result['x'][0]
        y = result['x'][1]
        z = result['x'][2]

        # sol = optimize.least_squares(AlignmentDirectionOptimization.compute_residuals, x_init, args=([data, frame, g, self.near_thr]), method='lm',
        #                              xtol=1e-15, ftol=1e-15)
        print(" Optimize ---x:{}, y:{}, z:{}".format(x, y, z))
        print('norm = {}'.format(np.linalg.norm(result['x'])))
        # print("-----------------------------------------------------------------------------------------")





