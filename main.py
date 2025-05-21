import numpy as np

import libs_data

from libs_data import get_2d_mesh_to_numpy

# params
MAX_DEPTH = 2

# build the dummy data
x1, x2, y = libs_data.get_2d_dummy_v2()
Xy = get_2d_mesh_to_numpy(x1,x2,y)

# split the tree
from libs import build_tree_nodes
nodes = build_tree_nodes(Xy, max_depth = MAX_DEPTH)

# builds the binary tree
from tree import Tree
t = Tree(nodes)
arms = t._bt.get_arms()

# calculate predictions and losses
from libs import predict
from libs import add_arm_indices_to_Xy, get_arm_stats, predict_at_cx
Xya = add_arm_indices_to_Xy(Xy, t)
stats = get_arm_stats(Xya, t)
y_pred = predict(Xya, t)

y_pred_mesh = np.zeros(x1.shape)
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        cx = [x1[i,j], x2[i,j]]
        y_pred_mesh[i, j] = predict_at_cx(Xya, cx, t)


# plotting
import matplotlib.pyplot as plt
from matplotlib import cm


print(f'Outputing plots to y_actual.png and y_pred.png based on max depth : {MAX_DEPTH}')
output = plt.subplot(111, projection = '3d')   # 3d projection
output.plot_surface(x1, x2, y_pred_mesh, rstride = 2, cstride = 1, cmap =  plt.cm.Blues_r)
output.set_xlabel('x')                         # axis label
output.set_xlabel('y')
output.set_xlabel('z')
plt.savefig('y_actual.png')

output = plt.subplot(111, projection = '3d')   # 3d projection
output.plot_surface(x1, x2, y_pred_mesh, rstride = 2, cstride = 1, cmap =  cm.coolwarm)
output.set_xlabel('x')                         # axis label
output.set_xlabel('y')
output.set_xlabel('z')
plt.savefig(f'y_pred_{MAX_DEPTH}.png')


