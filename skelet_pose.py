import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation


class node:
    def __init__(self, gltf_node):
        translation = gltf_node.translation if gltf_node.translation else [0, 0, 0]
        scale = gltf_node.scale if gltf_node.scale else [1, 1, 1]
        if translation and scale:
            self.size = np.linalg.norm(np.array(translation) * np.array(scale))
        else:
            self.size = 0
        self.rotation = gltf_node.rotation if gltf_node.rotation else [0, 0, 0, 1]
        self.children = gltf_node.children if gltf_node.children else []
        self.name = gltf_node.name
        self.father = -1
        self.coord = [0, 0, 0]

    def refresh(self, skelet, start_vec, start_coord):
        quat = Rotation.from_quat(self.rotation)
        vector = quat.apply(start_vec)
        self.coord = start_coord + vector
        for i in self.children:
            skelet.nodes[i].refresh(skelet, vector, self.coord)


class skeleton:
    def __init__(self, glb_file_path):
        self.nodes = []
        self.base_node = -1
        gltf = GLTF2().load(glb_file_path)
        for node_index, node_it in enumerate(gltf.nodes):
            self.nodes.append(node(node_it))
        for j in range(len(self.nodes)):
            for i in self.nodes[j].children:
                self.nodes[i].father = j
        for i in range(len(self.nodes)):
            if self.nodes[i].father == -1:
                self.base_node = i
                break

    def refresh(self):
        start_vec = np.array([1, 0, 0])
        start_coord = np.array([0, 0, 0])
        self.nodes[self.base_node].refresh(self, start_vec, start_coord)

    def get_by_name(self, name: str):
        for nod in self.nodes:
            if name == nod.name:
                return nod
        return 0


def get_sizes(skelet):
    sizes = {}
    for nod in skelet.nodes:
        if nod.translation and nod.scale:
            size = np.linalg.norm(np.array(nod.translation) * np.array(nod.scale))
        else:
            size = 0
        sizes[nod.name] = size
    return sizes


def append_by_num(skelet, num, prev_vec, start):
    nod = skelet.nodes[num]

    quat = Rotation.from_quat(nod.rotation)
    vector = quat.apply(prev_vec)

    shift = nod.size * vector
    final_point = start + shift
    vecs = [[start[0], start[1], start[2], shift[0], shift[1], shift[2]]]

    if nod.children:
        for j in nod.children:
            vecs += append_by_num(skelet, j, vector, final_point)
    return vecs
