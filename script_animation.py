import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.animation import FuncAnimation

from skelet_animation import *
from skelet_pose import *

rc('animation', html='jshtml')


animation_path = "m_74_Motion_out/m_74_Motion.gltf"
pose_path = "model_low.glb"

skelet = skeleton(pose_path)
animation = get_quat(animation_path)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Vector Field')

quiver = ax.quiver([], [], [], [], [], [], pivot='tail')

n_frames = len(animation["CC_Base_Hip"]["times"])

anim = FuncAnimation(fig, update, frames=n_frames, fargs=(skelet, animation, quiver), interval=100, blit=False)

plt.show()
