import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation

# Пример соединений между точками
connections = [
    # Соединения на голове
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (7, 8), (9, 10),

    # Верхняя часть тела
    (11, 12), (11, 23), (12, 24),

    # Руки
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

    # Нижняя часть тела
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (31, 32)
]

def update(num, pose_data, lines):
    segments = []
    for start, end in connections:
        start_point = pose_data[num][start][0:3]
        end_point = pose_data[num][end][0:3]
        segments.append([start_point, end_point])
    # Update the segments in the Line3DCollection
    lines.set_segments(segments)
    return lines,

pose_data = np.load('pose_data.npy', allow_pickle=True)

# Настройка графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Vector Field')

# Create an initial set of segments as a placeholder
initial_segments = [[[0, 0, 0], [0, 0, 0]]]
lines = Line3DCollection(initial_segments, linewidths=2)
ax.add_collection3d(lines)

n_frames = len(pose_data)

# Анимация
anim = FuncAnimation(fig, update, frames=n_frames, fargs=(pose_data, lines), interval=100, blit=False)

plt.show()