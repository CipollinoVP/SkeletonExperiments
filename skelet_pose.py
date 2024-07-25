import numpy as np
from pygltflib import GLTF2
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
            size = np.linalg.norm(np.array(nod.translation) * np.array(nod.scale))
        else:
            size = 0
        sizes[nod.name] = size
    return sizes


def append_by_num(skelet, sizes, num, prev_vec, start):
    nod = skelet.nodes[num]
    name = nod.name

    quat = Rotation.from_quat(nod.rotation)
    vector = quat.apply(prev_vec)

    shift = sizes[name] * vector
    final_point = start + shift
    vecs = [[start[0], start[1], start[2], shift[0], shift[1], shift[2]]]

    if nod.children:
        for j in nod.children:
            vecs += append_by_num(skelet, sizes, j, vector, final_point)
    return vecs
