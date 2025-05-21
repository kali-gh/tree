import numpy as np


def get_2d_dummy():
    """
    Builds a dummy 2d problem and returns mesh data for it

    :return: meshes for x1, x2, y
    """
    x1, x2 = np.mgrid[-2: 2: 0.1, -2: 2: 0.1]

    def model(x1, x2):
        y = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x2[i, j] > 0:
                    y[i, j] = 2
                else:
                    pass
        return y


    y = model(x1,x2)

    return x1, x2, y

def get_2d_dummy_v2():
    """
    Builds a dummy 2d problem and returns mesh data for it

    :return: meshes for x1, x2, y
    """
    x1, x2 = np.mgrid[-2: 2: 0.1, -2: 2: 0.1]

    def model(x1, x2):
        y = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                if x2[i, j] > -1:
                    y[ i, j] = 0.5
                if x2[i, j] > 0:
                    if x1[i, j] > 0:
                        y[i, j] = 1
                    else:
                        y[i, j] = 2
                else:
                    pass
        return y


    y = model(x1,x2)

    return x1, x2, y

def get_2d_mesh_to_numpy(x1, x2, y):
    """
    Converts data for a 2d mesh to an array

    :param x1: 2d x1 mesh
    :param x2: 2d x2 mesh
    :param y: 2d response
    :return: numpy array Nx3 with responses y for each point x1[i,j] and x2[i,j]
    """

    data = []

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            data.append([x1[i,j], x2[i,j], y[i,j]])

    return np.array(data)
