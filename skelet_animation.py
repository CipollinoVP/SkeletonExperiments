from skelet_pose import *

from pygltflib import GLTF2

import numpy as np
from scipy.spatial.transform import Rotation


def append_by_num_anim(skelet, sizes, num, prev_vec, start, i, animation):
    nod = skelet.nodes[num]
    name = nod.name

    if name in animation:
        rotation = animation[name]["rotations"][4 * i: 4 * i + 4]
        quat = Rotation.from_quat(rotation)
    else:
        quat = Rotation.from_quat(nod.rotation)
    vector = quat.apply(prev_vec)

    shift = sizes[name] * vector
    final_point = start + shift
    vecs = [[start[0], start[1], start[2], shift[0], shift[1], shift[2]]]

    if nod.children:
        for j in nod.children:
            vecs += append_by_num_anim(skelet, sizes, j, vector, final_point, i, animation)
    return vecs


def get_data_from_accessor(gltf, accessor):
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # Считывание бинарных данных
    data = gltf.load_file_uri(buffer.uri)

    # Извлечение данных в зависимости от типа компонента
    dtype = None
    if accessor.componentType == 5126:  # FLOAT
        dtype = np.float32
    elif accessor.componentType == 5123:  # UNSIGNED SHORT
        dtype = np.uint16
    elif accessor.componentType == 5122:  # SHORT
        dtype = np.int16
    elif accessor.componentType == 5121:  # UNSIGNED BYTE
        dtype = np.uint8
    elif accessor.componentType == 5120:  # BYTE
        dtype = np.int8

    # Учитывание смещений и количества элементов
    byte_offset = buffer_view.byteOffset + (accessor.byteOffset or 0)
    count = accessor.count
    component_size = np.dtype(dtype).itemsize
    num_components = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    }[accessor.type]
    byte_stride = buffer_view.byteStride or component_size * num_components

    # Извлечение и преобразование данных
    array = np.frombuffer(data, dtype=dtype, count=count * num_components, offset=byte_offset)
    return array.reshape((count, num_components))


def get_quat(gltf_path):
    gltf = GLTF2().load(gltf_path)
    animation = gltf.animations[0]

    out_dict = {}

    for channel in animation.channels:
        sampler = animation.samplers[channel.sampler]

        # Извлечение времени (входные данные) и кватернионов (выходные данные)
        input_accessor = gltf.accessors[sampler.input]
        output_accessor = gltf.accessors[sampler.output]

        # Извлечение данных из буферов
        input_data = get_data_from_accessor(gltf, input_accessor)
        output_data = get_data_from_accessor(gltf, output_accessor)

        node_id = channel.target.node
        # Получение имени узла (кости)
        node_name = gltf.nodes[node_id].name

        target_path = channel.target.path
        if target_path == "rotation":
            out_dict[node_name] = {"times": [], "rotations": []}
            for time, value in zip(input_data, output_data):
                out_dict[node_name]["times"].append(time)
                out_dict[node_name]["rotations"].append(value)

    return out_dict
