import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import figure, draw, pause


def to_homogeneous(A):
    """Convert a stack of inhomogeneous vectors to a homogeneous
       representation.
    """
    A = np.atleast_2d(A)

    N = A.shape[0]
    A_hom = np.hstack((A, np.ones((N,1))))

    return A_hom


def make_axis_publishable(ax, major_x, major_y, major_z):
    # [t.set_va('center') for t in ax.get_yticklabels()]
    # [t.set_ha('left') for t in ax.get_yticklabels()]
    # [t.set_va('center') for t in ax.get_xticklabels()]
    # [t.set_ha('right') for t in ax.get_xticklabels()]
    # [t.set_va('center') for t in ax.get_zticklabels()]
    # [t.set_ha('left') for t in ax.get_zticklabels()]

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.zaxis.set_major_locator(MultipleLocator(major_z))

def visualize_extrinsic(rgb_tensor=None, is_save=False, path=None, pred_g=None, K=None):
    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    make_axis_publishable(ax, 10, 10, 10)

    ax.set_title('Camera-Centric system')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # From StackOverflow: https://stackoverflow.com/questions/39408794/python-3d-pyramid
    v = np.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]])
    #v = to_homogeneous(v)

    n = 256
    x = np.linspace(-4, 4, n)
    y = np.linspace(-4, 4, n)
    X, Y = np.meshgrid(x, y)
    E = np.eye(4)

    gravity_target = np.array([0, 1., 0], dtype=np.float32).reshape(3,1)
    pred_g = pred_g.to('cpu').detach().numpy().copy().reshape(3,1)
    pred_g = pred_g / np.linalg.norm(pred_g)
    R = rotation_matrix_from_vectors(gravity_target, pred_g)
    E[0:3, 0:3] = R
    E_inv = np.linalg.inv(E)
    E_inv = E_inv[:3]  # 3 * 4
    v_new = np.dot(R,v.T).T #np.dot(v, E_inv.T)

    verts = [[v_new[0], v_new[1], v_new[4]], [v_new[0], v_new[3], v_new[4]],
            [v_new[2], v_new[1], v_new[4]], [v_new[2], v_new[3], v_new[4]],
            [v_new[0], v_new[1], v_new[2], v_new[3]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    #pred gravity direction
    linesegment = [[0., 0., 0.], [pred_g[0,0], pred_g[1,0], pred_g[2,0]]]
    ax.add_collection3d(Poly3DCollection(linesegment, facecolors='cyan', linewidths=3, edgecolors='y', alpha=.25))
    ax.view_init(azim=-90, elev=-77)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255).copy()
    from PIL import Image

    image = Image.fromarray(output_rgb_img[:,:,::-1])
    #image = image[200:-180, 200:-180, :]
    im = OffsetImage(image, zoom=0.45)
    x, y = -0.1, 0.1
    ab = AnnotationBbox(im, (x, y), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)
    #ax.invert_xaxis()
    if is_save:
        if path != None:
            fig.savefig(path)
        else:
            return fig


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


def compute_R_from_vectors():
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
                    2. * skewsymm_Cg_q_C @ skewsymm_Cg_q_C
    Cg_H_C = self.K @ Cg_R_C @ self.K_inv
    C_R_Cg = Cg_R_C.permute(0, 2, 1)
