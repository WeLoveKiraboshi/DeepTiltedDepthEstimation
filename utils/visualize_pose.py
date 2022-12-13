import cv2
import numpy as np
import test
import os
import pickle


import time
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def random_point( r=1 ):
    ct = 2*np.random.rand() - 1
    st = np.sqrt( 1 - ct**2 )
    phi = 2* np.pi *  np.random.rand()
    x = r * st * np.cos( phi)
    y = r * st * np.sin( phi)
    z = r * ct
    return np.array( [x, y, z ] )

@jit
def near( p, pntList, d0 ):
    cnt=0
    counter = 0
    for pj in pntList:
        counter += 1
        dist=np.linalg.norm( p - pj )
        if dist < d0:
            cnt += 1 - dist/d0
    #print('finish calc near vector within the sphere plots')
    return cnt


def visualize_density(data):
    pointList = np.array([random_point(10.05) for i in range(10)])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)

    # create the sphere surface
    XX = 10 * np.outer(np.cos(u), np.sin(v))
    YY = 10 * np.outer(np.sin(u), np.sin(v))
    ZZ = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    WW = XX.copy()
    for i in range(len(XX)):
        for j in range(len(XX[0])):
            x = XX[i, j]
            y = YY[i, j]
            z = ZZ[i, j]
            WW[i, j] = near(np.array([x, y, z]), pointList, 3)
            WW = WW / np.amax(WW)
            myheatmap = WW

    # ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
    ax.plot_surface(XX, YY, ZZ, cstride=1, rstride=1, facecolors=cm.jet(myheatmap))
    plt.show()

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def vis(data):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    xi, yi, zi = sample_spherical(100)
    xi = data[:1000, 0]
    yi = data[:1000, 1]
    zi = data[:1000, 2]
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'auto'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
    plt.show()


def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, length = 1.2,
              arrow_length_ratio = 0.2)



def vis2(data):
    # Creating the theta and phi values.
    theta = np.linspace(0, np.pi, 100, endpoint=True)
    phi = np.linspace(0, np.pi * 2, 100, endpoint=True)

    # Creating the coordinate grid for the unit sphere.
    X = np.outer(np.sin(theta), np.cos(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.cos(theta), np.ones(100))

    print(data)
    WW = X.copy()
    for i in range(len(X)):
        for j in range(len(X[0])):
            x = X[i, j]
            y = Y[i, j]
            z = Z[i, j]
            WW[i, j] = near(np.array([x, y, z]), data[0:, :], 3)
            print('i = {}, j = {}'.format(i, j))
    WW = WW / np.amax(WW)

    # Creat the plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.plot_surface(X, Y, Z, facecolors=cm.jet(WW), alpha=0.22, linewidth=1,cstride=1, rstride=1) #

    loc = [0, 0, 0]
    u = [2, 0, 0]
    v = [0, 2, 0]
    w = [0, 0, 2]
    visual_vector_3d(ax, loc, u, "red") # x axis
    visual_vector_3d(ax, loc, v, "blue") # y axis
    visual_vector_3d(ax, loc, w, "green") # z axis


    xi = data[:, 0]
    yi = data[:, 1]
    zi = data[:, 2]
    #ax.scatter(xi, yi, zi, s=100, c='r', zorder=1)

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(WW)
    plt.colorbar(m)

    # Show the plot.
    plt.show()



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


def read_data_rectified_2dofa_pkl(datadir='/media/yukisaito/ssd2/ScanNetv2/scans', train_test_split='data/rectified_2dofa_scannet.pkl'):
    data_info = pickle.load(open(train_test_split, 'rb'))
    train_data = data_info['train']
    test_data = data_info['test']
    val_data = data_info['val']

    data = []
    init_grav = np.array([0.0, 0.0, -1.0]).reshape(3, 1)
    print('loading pose info... from {}'.format(train_test_split))
    max_iter = 100

    for iter, e2_comp in enumerate(train_data['e2']):
        subdir = e2_comp[:12]
        frame_idx = int(e2_comp[19:25])
        impath = os.path.join(datadir, subdir, 'color', str(frame_idx) + '.jpg')
        depthpath = os.path.join(datadir, subdir, 'depth', str(frame_idx) + '.jpg')
        posepath = os.path.join(datadir, subdir, 'pose', str(frame_idx) + '.txt')
        pose_array = np.loadtxt(posepath, dtype=np.float32)[:3, :3]
        gravity_array = np.linalg.inv(pose_array) @ init_grav
        gravity_array = gravity_array / np.linalg.norm(gravity_array)
        data.append(gravity_array)
        if iter > max_iter:
            break


    # for _e2_comp in train_data['-e2']:
    #     print(_e2_comp)

    # for idx in range(total_idx):
    #     impath = os.path.join(subdir, 'color', str(idx)+'.jpg')
    #     depthpath = os.path.join(subdir, 'depth', str(idx)+'.jpg')
    #     posepath = os.path.join(subdir, 'pose', str(idx)+'.txt')
    #
    #     #cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
    #     pose = np.loadtxt(posepath)[:-1, :-1]
    #     grav_dir = pose @ np.array([0,1,0])
    #     data.append(grav_dir)
    print('loading pose info done ...')

    return np.array(data)


if __name__ == '__main__':
    datadir = '/media/yukisaito/ssd2/ScanNetv2/scans'
    train_test_split = 'data/rectified_2dofa_scannet.pkl'
    dataset = read_data_rectified_2dofa_pkl(datadir, train_test_split)
    vis2(dataset)
    #visualize_density(dataset)