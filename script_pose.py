import matplotlib.pyplot as plt
import numpy as np

from skelet_pose import *

path_to_glb = "CharAdoptisLow.glb"
skelet = skeleton(path_to_glb)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Vector Field')

data = append_by_num(skelet, len(skelet.nodes) - 1, np.array([1, 0, 0]), np.array([0, 0, 0]))

for i in range(len(skelet.nodes)):
    print("       ")
    print(i)
    print(skelet.nodes[i].name)
    for j in range(len(skelet.nodes[i].children)):
        print(skelet.nodes[i].children[j])

Quiver = [[], [], [], [], [], []]
for dat in data:
    for i in range(len(Quiver)):
        Quiver[i].append(dat[i])

quiver = ax.quiver(Quiver[0], Quiver[1], Quiver[2], Quiver[3], Quiver[4], Quiver[5])

plt.show()
