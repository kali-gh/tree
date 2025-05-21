import matplotlib.pyplot as plt

def plot_2d(x1, x2, y):

    output = plt.subplot(111, projection = '3d')   # 3d projection
    output.plot_surface(x1, x2, y, rstride = 2, cstride = 1, cmap = plt.cm.Blues_r)
    output.set_xlabel('x')                         # axis label
    output.set_xlabel('y')
    output.set_xlabel('z')

    plt.show()