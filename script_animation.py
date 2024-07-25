from matplotlib.animation import FuncAnimation

from skelet_pose import *
from skelet_animation import *
from skelet_pose import *

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('animation', html='jshtml')


def update(num, skelet, anim, sizes, quiver):
    start_num = skelet.nodes[-1].children[0]
    skeleton_vec = append_by_num_anim(skelet, sizes, start_num, np.array([1, 0, 0]), np.array([0, 0, 0]), num, anim)

    X, Y, Z, U, V, W = [], [], [], [], [], []
    for vec in skeleton_vec:
        X.append(vec[0])
        Y.append(vec[1])
        Z.append(vec[2])
        U.append(vec[3])
        V.append(vec[4])
        W.append(vec[5])

    quiver.set_segments([[[x, y, z], [x + u, y + v, z + w]] for x, y, z, u, v, w in zip(X, Y, Z, U, V, W)])
    return quiver,


animation_path = "m_78_Motion_out/m_78_Motion.gltf"
pose_path = "model_low.glb"

skelet = skeleton(pose_path)
animation = get_quat(animation_path)
sizes = get_sizes(skelet)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Vector Field')

quiver = ax.quiver([], [], [], [], [], [])

n_frames = len(animation["CC_Base_Hip"]["times"])

anim = FuncAnimation(fig, update, frames=n_frames, fargs=(skelet, animation, sizes, quiver), interval=100, blit=False)

plt.show()

