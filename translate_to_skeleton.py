from datafile import *
from skelet_pose import *
import numpy as np


def get_video_vecs(data):
    vecs = []
    bones = list(point_bones.keys())
    nums = list(point_bones.values())
    for i in range(len(bones)):
        for j in range(i + 1, len(bones)):
            if isinstance(nums[i], int):
                vec1 = np.array([data[nums[i]][0], data[nums[i]][1], data[nums[i]][2]])
            else:
                vec10 = np.array([data[nums[i][0]][0], data[nums[i][0]][1], data[nums[i][0]][2]])
                vec11 = np.array([data[nums[i][1]][0], data[nums[i][1]][1], data[nums[i][1]][2]])
                vec1 = (vec10 + vec11) / 2

            if isinstance(nums[j], int):
                vec2 = np.array([data[nums[j]][0], data[nums[j]][1], data[nums[j]][2]])
            else:
                vec20 = np.array([data[nums[j][0]][0], data[nums[j][0]][1], data[nums[j][0]][2]])
                vec21 = np.array([data[nums[j][1]][0], data[nums[j][1]][1], data[nums[j][1]][2]])
                vec2 = (vec20 + vec21) / 2

            vec = vec1 - vec2
            vec = vec / np.linalg.norm(vec)
            vecs.append([bones[i], bones[j], vec])

    return vecs


def objective_function(vecs, skelet: skeleton):
    min = 2.
    for item in vecs:
        coord1 = skelet.get_by_name(item[0]).coord
        coord2 = skelet.get_by_name(item[1]).coord
        local_vec = coord1 - coord2
        local_vec = local_vec / np.linalg.norm(local_vec)
        local_objective = np.linalg.norm(item[2] - local_vec)
        if local_objective < min:
            min = local_objective

    return min


pose_data = np.load('pose_data.npy', allow_pickle=True)

path_to_glb = "CharAdoptisLow.glb"
skelet = skeleton(path_to_glb)

skelet.refresh()

vecs = get_video_vecs(pose_data[0])

print(objective_function(vecs, skelet))
