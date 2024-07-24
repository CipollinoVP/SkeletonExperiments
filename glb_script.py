import json

from pygltflib import GLTF2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation


class node:
    def __init__(self, gltf_node):
        self.translation = gltf_node.translation if gltf_node.translation else [0, 0, 0]
        self.rotation = gltf_node.rotation if gltf_node.rotation else [0, 0, 0, 1]
        self.scale = gltf_node.scale if gltf_node.scale else [1, 1, 1]
        self.children = gltf_node.children if gltf_node.children else []
        self.name = gltf_node.name


class skeleton:
    def __init__(self, glb_file_path):
        self.nodes = []
        gltf = GLTF2().load(glb_file_path)
        for node_index, node_it in enumerate(gltf.nodes):
            self.nodes.append(node(node_it))


def get_sizes(skelet):
    sizes = {}
    for nod in skelet.nodes:
        if nod.translation and nod.scale:
            size = np.linalg.norm(np.array(nod.translation) * np.array(node.scale))
        else:
            size = 0
        sizes[nod.name] = size
    return sizes


def append_by_num(skelet, sizes, num, prev_vec, start):
    nod = skelet.nodes[num]
    name = nod["name"]

    quat = Rotation.from_quat(nod.rotation)
    vector = quat.apply(prev_vec)

    shift = sizes[name] * vector
    final_point = start + shift
    vecs = [[start[0], start[1], start[2], shift[0], shift[1], shift[2]]]

    if nod.children:
        for j in nod.children:
            vecs += append_by_num(skelet, sizes, j, vector, final_point)
    return vecs


path_to_glb = "CharAdoptisLow.glb"
skelet = skeleton(path_to_glb)
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

n_frames = len(anim["CC_Base_Hip"]) // 4

ani = FuncAnimation(fig, update, frames=n_frames, fargs=(data, anim, sizes, quiver), interval=100, blit=False)
