import torch
import numpy as np


class Warping2DOFAlignment:
    def __init__(self, fx=577.87061*0.5, fy=577.87061*0.5, cx=319.87654*0.5, cy=239.87603*0.5, mode='train', dataset='scannet'):
        self.device = 'cuda:0'
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.W = 320 #np.ceil(2 * cx).astype(int)
        self.H = 240 #np.ceil(2 * cy).astype(int)
        assert self.W == 320 and self.H == 240
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        self.YY, self.XX = np.meshgrid(np.linspace(0, self.H - 1, self.H), np.linspace(0, self.W - 1, self.W))
        self.corners_points = np.array([[0, 0, 1], [self.W - 1, 0, 1], [0, self.H - 1, 1], [self.W - 1, self.H - 1, 1]]).transpose()
        
        self.K = torch.tensor(self.K, dtype=torch.float).to(self.device)
        self.K_inv = torch.tensor(self.K_inv, dtype=torch.float).to(self.device)
        self.YY = torch.tensor(self.YY, dtype=torch.float).to(self.device)
        self.XX = torch.tensor(self.XX, dtype=torch.float).to(self.device)
        self.corners_points = torch.tensor(self.corners_points, dtype=torch.float).to(self.device)
        self.I3 = torch.tensor(np.identity(3), dtype=torch.float).to(self.device)

        self.template_principle_dirs = torch.tensor([[0., 1., 0.], [0., -1., 0.], [1., 0., 0.], [-1., 0., 0.]], dtype=torch.float).to(self.device)
        self.template_principle_dirs = self.template_principle_dirs.transpose(1, 0)
        self.mode = mode
        self.dataset = dataset
        if 'OurDataset_roll' in self.dataset:
            self.min_scale_sigma = 0.6
            self.max_scale_sigma = 2.2
        elif 'OurDataset_pitch' in self.dataset:
            self.min_scale_sigma = 0.8
            self.max_scale_sigma = 2.2
        else:
            self.min_scale_sigma = 0.8
            self.max_scale_sigma = 2.2

        self.IMG_MEAN = torch.tensor([97.9909, 113.3113, 126.4953], dtype=torch.float).to(self.device)


    def _skewsymm(self, x):
        if x.shape[0] == 1:
            return torch.tensor([[0.0, -x[0, 2], x[0, 1]], [x[0, 2], 0.0, -x[0, 0]], [-x[0, 1], x[0, 0], 0.0]], dtype=torch.float).to(
                self.device)
        else:
            return torch.tensor([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]], dtype=torch.float).to(
                self.device)

    def _build_homography(self, I_g, I_a, derotate_flag=False):
        if derotate_flag: #I_g = (16, 3)  #extract roll rotation matrix
            I_g[:, 2] = 0
        Cg_R_C = torch.zeros((I_g.shape[0], 3, 3), device=I_g.device, dtype=torch.float)
        skewsymm_I_a = torch.zeros((I_g.shape[0], 3, 3), device=I_g.device, dtype=torch.float)
        I_g = I_g.view(I_g.shape[0], 3, 1)
        I_a = I_a.view(I_a.shape[0], 1, 3)
        for i in range(I_g.shape[0]):
            skewsymm_I_a[i] = -self._skewsymm(I_a[i].view(-1))
        Cg_q_C = skewsymm_I_a @ I_g
        dot_Ig_warped_direction = I_a @ I_g
        norm_Cg_q_C = Cg_q_C.clone().norm(dim=1)
        Cg_q4_C = torch.cuda.FloatTensor(I_a.shape[0], 1).fill_(0)

        for i in range(Cg_R_C.shape[0]):
            Cg_q4_C[i] = torch.cos(0.5 * torch.atan2(norm_Cg_q_C[i, 0], dot_Ig_warped_direction[i, 0, 0]))
            if norm_Cg_q_C[i] < 1e-4 or Cg_q4_C[i].abs() < 1e-4:
                Cg_R_C[i] = self.I3
            Cg_q_C[i] = Cg_q_C[i].div((2.0 * Cg_q4_C[i]))
            skewsymm_Cg_q_C = self._skewsymm(Cg_q_C[i].view(-1))
            Cg_R_C[i] = self.I3 + 2. * Cg_q4_C[i] * skewsymm_Cg_q_C + \
                                      2. * skewsymm_Cg_q_C @ skewsymm_Cg_q_C  #Rotation Mat transforming I_g => I_a
            # if derotate_flag:
            #     Cg_R_C[i] = self.de_rotate_matrix(Cg_R_C[i])
        Cg_H_C = self.K @ Cg_R_C @ self.K_inv
        C_R_Cg = Cg_R_C.permute(0, 2, 1)
        Cg_H_C_inv = self.K @ C_R_Cg @ self.K_inv
        return Cg_H_C, Cg_R_C, Cg_H_C_inv

    def _convert_R_vec_to_mat(self, pitch_angle, yaw_angle, roll_angle):
        """
        Function to convert R vector computed using Rodriguez formula back to a mtrix
        R_vec = [wx, wy, wz]
        :return:
        """
        roll = torch.from_numpy(roll_angle.astype(np.float32)).clone() #torch.randn(roll_angle, requires_grad=True)
        yaw = torch.from_numpy(yaw_angle.astype(np.float32)).clone() #torch.Tensor(yaw_angle) #torch.randn(yaw_angle, requires_grad=True)
        pitch = torch.from_numpy(pitch_angle.astype(np.float32)).clone() #torch.Tensor(pitch_angle) #torch.randn(pitch_angle, requires_grad=True)

        tensor_0 = torch.zeros(1)
        tensor_1 = torch.ones(1)

        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RY, RX)
        R = torch.mm(R, RZ)
        return R

    def _convert_R_mat_to_vec(self, R_mat):
        """
        Use rodiriguez formula to convert R matrix to R vector with 3 DoF
        :return:
        yaw=atan2(R(2,1),R(1,1));
        pitch=atan2(-R(3,1),sqrt(R(3,2)^2+R(3,3)^2)));
        roll=atan2(R(3,2),R(3,3))
        """
        phi = torch.arccos((torch.trace(R_mat) - 1) / 2)
        print(phi)
        R_vec = torch.Tensor([R_mat[2][1] - R_mat[1][2], R_mat[0][2] - R_mat[2][0], R_mat[1][0] - R_mat[0][1]]).to(self.device)
        R_vec = R_vec * (phi / (2 * torch.sin(phi)))
        return R_vec

    def de_rotate_matrix(self, Rc):
        Rc_np = Rc.detach().cpu().numpy()
        import cv2
        rotVec, _ = cv2.Rodrigues(Rc_np)
        yaw = np.array([0.], dtype=np.float32) #rotVec[1]#[0]
        pitch = rotVec[0]#[0] np.array([0.], dtype=np.float32)
        roll = rotVec[2]#[0]
        #print('yaw = {}  pitch = {}  theta = {}'.format(yaw, pitch, roll))
        Rc_rp = self._convert_R_vec_to_mat(pitch, yaw, roll)
        return Rc_rp



    def image_sampler_forward_inverse(self, I_g, I_a, warped_corners_points):
        Cg_H_C, Cg_R_C, Cg_H_C_inv = self._build_homography(I_g, I_a,derotate_flag=False)
        Cg_H_C = Cg_H_C.type(torch.float)
        Cg_H_C_inv = Cg_H_C_inv.type(torch.float)
        C_R_Cg = Cg_R_C.permute(0, 2, 1)

        grid_sampler = torch.cuda.FloatTensor(I_g.shape[0], self.H, self.W, 2).fill_(0)
        inv_grid_sampler = torch.cuda.FloatTensor(I_g.shape[0], self.H, self.W, 2).fill_(0)
        C_R_Cg_ret = torch.zeros_like(C_R_Cg)

        for i in range(I_g.shape[0]):
            Cg_corners_points = Cg_H_C[i] @ self.corners_points # Need proper broadcast w/ batchsize as input
            Cg_corners_points_projection = Cg_corners_points[0:2].clone() / Cg_corners_points[2].clone()

            px_max = torch.max(Cg_corners_points_projection[0])
            px_min = torch.min(Cg_corners_points_projection[0])
            py_max = torch.max(Cg_corners_points_projection[1])
            py_min = torch.min(Cg_corners_points_projection[1])
            #print('Xmin = {}   Xmax={}   Ymin={}   Ymax={}'.format(px_min, px_max, py_min, py_max))

            h_max = py_max - py_min
            w_max = px_max - px_min
            scale_sigma = w_max.detach() / h_max.detach()
            #print(scale_sigma)
            if scale_sigma < self.min_scale_sigma or scale_sigma > self.max_scale_sigma: #scale_sigma < 0.8 or scale_sigma > 2.2: #TUMfrei1rpy 0.6, 2.2
                #print('Protector detects exceeded transformation. {}', i)
                C_R_Cg_ret[i] = self.I3
                grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (self.XX - self.cx), (self.W, self.H)).t()
                grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (self.YY - self.cy), (self.W, self.H)).t()
                inv_grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (self.XX - self.cx),
                                                             (self.W, self.H)).t()
                inv_grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (self.YY - self.cy),
                                                             (self.W, self.H)).t()
                continue

            warped_corners_points_i = torch.cat([warped_corners_points[i], torch.ones(1, 4).to(self.device)], dim=0)
            Cg_corners_points_original = Cg_H_C[i] @ warped_corners_points_i  # Need proper broadcast w/ batchsize as input (3*3) (2*4)
            Cg_corners_points_projection_original = Cg_corners_points_original[0:2].clone() / \
                                                    Cg_corners_points_original[2].clone()

            h_max_original = torch.max(Cg_corners_points_projection_original[1, :]) - torch.min(
                Cg_corners_points_projection_original[1, :])
            w_max_original = torch.max(Cg_corners_points_projection_original[0, :]) - torch.min(
                Cg_corners_points_projection_original[0, :])

            #print('w ={}, h={}    orig_w={}, orig_h={}'.format(w_max, h_max, w_max_original, h_max_original))
            if self.mode != 'test':
                if w_max > 4 * h_max / 3:
                    kw = self.W / w_max.clone()
                    kh = self.H / (3 * w_max.clone() / 4)
                else:
                    kh = self.H / (h_max.clone())
                    kw = self.W / (4 * h_max.clone() / 3)
            else:
                if w_max > 4 * h_max / 3:
                    kw = self.W / w_max_original.clone()
                    kh = self.H / (3 * w_max_original.clone() / 4)
                else:
                    kh = self.H / (h_max_original.clone())
                    kw = self.W / (4 * h_max_original.clone() / 3)

            C1p = Cg_H_C_inv[i] @ torch.reshape(torch.cat((1./kw * (self.XX) + px_min,
                                                           1./kh * (self.YY) + py_min,
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))

            C1p_x = C1p[0].clone() / C1p[2].clone()
            C1p_y = C1p[1].clone() / C1p[2].clone()
            grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (C1p_x - self.cx), (self.W, self.H)).t()
            grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (C1p_y - self.cy), (self.W, self.H)).t()

            inv_C1p = Cg_H_C[i] @ torch.reshape(torch.cat((self.XX,
                                                           self.YY,
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))
            inv_C1p_projection = inv_C1p[0:2, :].clone() / inv_C1p[2, :].clone()
            inv_C1p_x = kw * (inv_C1p_projection[0, :] - px_min)
            inv_C1p_y = kh * (inv_C1p_projection[1, :] - py_min)
            inv_grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (inv_C1p_x - self.cx), (self.W, self.H)).t()
            inv_grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (inv_C1p_y - self.cy), (self.W, self.H)).t()
            C_R_Cg_ret[i] = C_R_Cg[i]

        return C_R_Cg_ret, grid_sampler, inv_grid_sampler



    def warp_all_with_gravity_center_aligned(self, x, I_g,  I_a,  image_border='zeros'): # Z -> depth
        x_i = x['image']
        x_m = x['rgb_mask']
        x_n_m = x['mask']
        x_n = x['depth']
        C_g = x['gravity']
        C_g = C_g.view(C_g.shape[0], 3, 1)
        C_a = x['aligned_directions']
        C_a = C_a.view(C_a.shape[0], 3, 1)
        w_x = {'image': [], 'mask': [], 'gravity': [], 'aligned_directions': [], 'depth':[], 'Cg_H_C_inv':[], 'corners_points':[]}

        x_m = x_m.view(x_m.shape[0], x_m.shape[1], x_m.shape[2], x_m.shape[3])

        Cg_H_C, Cg_R_C, Cg_H_C_inv = self._build_homography(I_g, I_a)
        Cg_H_C = Cg_H_C.type(torch.float)
        Cg_H_C_inv = Cg_H_C_inv.type(torch.float)
        Cg_corners_points_list = torch.zeros(x_i.shape[0], 2, 4, dtype=torch.float).to(self.device)

        grid_sampler = torch.cuda.FloatTensor(x_i.shape[0], self.H, self.W, 2).fill_(0)
        assert x_i.shape[0] == I_g.shape[0]
        for i in range(x_i.shape[0]): #loop for batch nums
            Cg_corners_points = Cg_H_C[i] @ self.corners_points # Need proper broadcast w/ batchsize as input
            Cg_corners_points = Cg_corners_points[0:2].clone() / Cg_corners_points[2].clone()
            h_max = torch.max(Cg_corners_points[1, :]) - torch.min(Cg_corners_points[1, :])
            w_max = torch.max(Cg_corners_points[0, :]) - torch.min(Cg_corners_points[0, :])

            if w_max > 4 * h_max / 3:
                kw = self.W / w_max
                kh = self.H / (3 * w_max / 4)
            else:
                kh = self.H / (h_max)
                kw = self.W / (4 * h_max / 3)

            C1p = Cg_H_C_inv[i] @ torch.reshape(torch.cat((1./kw * (self.XX) + torch.min(Cg_corners_points[0, :]),
                                                           1./kh * (self.YY) + torch.min(Cg_corners_points[1, :]),
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))
            C1p = C1p.div(C1p[2, :])
            C1p[0, :] = 1. / (self.W / 2) * (C1p[0, :] - self.cx)
            C1p[1, :] = 1. / (self.H / 2) * (C1p[1, :] - self.cy)
            grid_sampler[i, :, :, 0] = torch.reshape(C1p[0, :], (self.W, self.H)).t()
            grid_sampler[i, :, :, 1] = torch.reshape(C1p[1, :], (self.W, self.H)).t()
            #print(' list  = {}    pts = {}'.format(Cg_corners_points_list.shape, Cg_corners_points.shape))
            Cg_corners_points_list[i] = Cg_corners_points

        if image_border == 'mean':
            y_i = torch.nn.functional.grid_sample(x_i, grid_sampler, padding_mode='zeros', mode='bilinear')
            y_m = torch.nn.functional.grid_sample(x_m, grid_sampler, padding_mode='zeros', mode='nearest')
            y_m = y_m.view(x_m.shape[0], x_m.shape[1], x_m.shape[2], x_m.shape[3])
            for i_ch in range(3):
                y_i[:, i_ch, :, :] = y_i[:, i_ch, :, :].masked_fill(y_m[:, i_ch, :, :] == 0, self.IMG_MEAN[i_ch]/255)
        else:
            y_i = torch.nn.functional.grid_sample(x_i, grid_sampler, padding_mode=image_border, mode='bilinear')
            y_m = torch.nn.functional.grid_sample(x_m, grid_sampler, padding_mode='zeros', mode='nearest')
            y_m = y_m.view(x_m.shape[0], x_m.shape[1], x_m.shape[2], x_m.shape[3])


        y_n_m = torch.nn.functional.grid_sample(x_n_m, grid_sampler, padding_mode='zeros', mode='nearest')
        y_n_m = y_n_m.view(x_n_m.shape[0], x_n_m.shape[1], x_n_m.shape[2], x_n_m.shape[3])


        y_n = torch.nn.functional.grid_sample(x_n, grid_sampler, padding_mode='zeros', mode='bilinear')
        z_n = Cg_R_C[:, 2, :].unsqueeze(2).bmm(y_n.view(x_n.shape[0], x_n.shape[1], x_n.shape[2] * x_n.shape[3]))
        z_n = z_n[:, 2, :].view(x_n.shape[0], x_n.shape[1], x_n.shape[2],  x_n.shape[3])

        #y_visible = torch.nn.functional.grid_sample(torch.ones_like(x_m), grid_sampler, padding_mode='zeros', mode='bilinear')

        w_x['image'] = y_i
        w_x['rgb_mask'] = y_m
        w_x['mask'] = y_n_m
        w_x['depth'] = z_n
        w_x['gravity'] = Cg_R_C.bmm(C_g).view(C_g.shape[0], 3)
        #w_x['visible_mask'] = y_visible

        # Computing the supervised aligned direction
        w_x['aligned_directions'] = x['aligned_directions']
        w_x['Cg_H_C_inv'] = Cg_H_C_inv
        w_x['corners_points'] = Cg_corners_points_list

        w_x['scene'] = x['scene']

        return w_x