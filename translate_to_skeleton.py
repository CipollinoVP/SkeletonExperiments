from datafile import *
from skelet_pose import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_vec(data, num):
    return np.array([data[num][1], data[num][2], data[num][0]])


def initial_pose(input_skelet: skeleton, mediapipe_data):
    skelet = input_skelet
    for keys, values in start_quat.items():
        node_id = skelet.get_by_name(keys)
        v1 = get_vec(mediapipe_data, values[1]) - get_vec(mediapipe_data, values[0])
        v2 = get_vec(mediapipe_data, values[2]) - get_vec(mediapipe_data, values[1])
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Находим ось поворота (векторное произведение) и угол поворота
        axis = np.cross(v1_norm, v2_norm)
        angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))

        # Создаем кватернион
        if np.linalg.norm(axis) == 0:
            # Векторы параллельны, поворот не требуется
            quaternion = np.array([1, 0, 0, 0])
        else:
            axis = axis / np.linalg.norm(axis)  # Нормализуем ось
            quaternion = R.from_rotvec(axis * angle).as_quat()

        skelet.nodes[node_id].rotation = quaternion

    return skelet


def get_video_vecs(data):
    vecs = []
    bones = list(point_bones.keys())
    nums = list(point_bones.values())
    for i in range(len(bones)):
        for j in range(i + 1, len(bones)):
            if isinstance(nums[i], int):
                vec1 = get_vec(data, nums[i])
            else:
                vec10 = get_vec(data, nums[i][0])
                vec11 = get_vec(data, nums[i][1])
                vec1 = (vec10 + vec11) / 2

            if isinstance(nums[j], int):
                vec2 = get_vec(data, nums[j])
            else:
                vec20 = get_vec(data, nums[j][0])
                vec21 = get_vec(data, nums[j][1])
                vec2 = (vec20 + vec21) / 2

            vec = vec1 - vec2
            vec = vec / np.linalg.norm(vec)
            vecs.append([bones[i], bones[j], vec])

    return vecs


def objective_function(vecs, skelet: skeleton):
    max = 0.
    for item in vecs:
        coord1 = skelet.nodes[skelet.get_by_name(item[0])].coord
        coord2 = skelet.nodes[skelet.get_by_name(item[1])].coord
        local_vec = coord1 - coord2
        local_vec = local_vec / np.linalg.norm(local_vec)
        local_objective = np.linalg.norm(item[2] - local_vec)
        if local_objective > max:
            max = local_objective

    return max


def quat_from_angles(angles):
    quat = np.zeros(4)
    quat[0] = np.cos(angles[0])
    quat[1] = np.sin(angles[0]) * np.cos(angles[1])
    quat[2] = np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
    quat[3] = np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2])
    return quat


def angles_from_quat(quat):
    angles = np.zeros(3)

    # Угол theta_1
    angles[0] = np.arccos(quat[0])

    # sin(theta_1) для вычисления theta_2 и theta_3
    sin_theta1 = np.sqrt(1 - quat[0] ** 2)

    if sin_theta1 < 1e-10:
        # Когда sin(theta_1) близок к нулю, theta_2 и theta_3 неопределены,
        # можем зафиксировать theta_2 и theta_3 на нуле.
        angles[1] = 0.0
        angles[2] = 0.0
    else:
        # Угол theta_2
        angles[1] = np.arccos(quat[1] / sin_theta1)

        # Угол theta_3
        angles[2] = np.arctan2(quat[3], quat[2])

    return angles


def descent_for_one(n_bones, d_eps, vecs, input_skelet: skeleton):
    skelet = input_skelet
    prev = 2.1
    func = objective_function(vecs, skelet)
    while prev > func:
        gradient = np.array([0., 0., 0.])
        angles = angles_from_quat(skelet.nodes[n_bones].rotation)

        for i in range(3):
            angles1 = angles
            angles1[i] += d_eps
            skelet1 = skelet
            skelet1.nodes[n_bones].rotation = quat_from_angles(angles1)
            skelet1.refresh()
            func1 = objective_function(vecs, skelet1)
            gradient[i] = (func1 - func) / d_eps

        angles = angles - 0.001 * gradient
        prev = func
        skelet.nodes[n_bones].rotation = quat_from_angles(angles)
        skelet.refresh()
        func = objective_function(vecs, skelet)
    return skelet


def search_minimize_bones(d_eps, vecs, skelet: skeleton):
    origin_val = objective_function(vecs, skelet)
    for name in bones_in_task:
        skelet1 = skelet
        nod_i = skelet1.get_by_name(name)
        skelet1.nodes[nod_i].rotation[0] += d_eps
        skelet1.refresh()
        if objective_function(vecs, skelet1) != origin_val:
            print(skelet.nodes[nod_i].name)
            return nod_i
    return -1


def minimization(d_eps, vecs, input_skelet: skeleton):
    skelet = input_skelet
    prev = 2.1
    func = objective_function(vecs, skelet)
    while func < prev:
        nod_i = search_minimize_bones(d_eps, vecs, skelet)
        skelet = descent_for_one(nod_i, d_eps, vecs, skelet)
        prev = func
        func = objective_function(vecs, skelet)

    return skelet


pose_data = np.load('pose_data.npy', allow_pickle=True)

path_to_glb = "CharAdoptisLow.glb"
skelet = skeleton(path_to_glb)

skelet.refresh()

vecs = get_video_vecs(pose_data[0])

skelet = initial_pose(skelet, pose_data[0])
skelet = minimization(0.00000001, vecs, skelet)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-100, 100])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Vector Field')

data = append_by_num(skelet, len(skelet.nodes) - 1, np.array([-1, 0, 0]), np.array([0, 0, 0]))

Quiver = [[], [], [], [], [], []]
for dat in data:
    for i in range(len(Quiver)):
        Quiver[i].append(dat[i])

quiver = ax.quiver(Quiver[0], Quiver[1], Quiver[2], Quiver[3], Quiver[4], Quiver[5])

plt.show()

'''

quat = np.array([0.5, 0.5, -0.5, 0.5])
angles = angles_from_quat(quat)
quat1 = quat_from_angles(angles)
print(quat1)
'''
