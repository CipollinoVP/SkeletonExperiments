import matplotlib.pyplot as plt
import numpy as np

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

pose_data = np.load('pose_data.npy', allow_pickle=True)

# Настройка графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отрисовка соединений с помощью quiver
for start, end in connections:
    start_point = pose_data[0][start]
    end_point = pose_data[0][end]

    ax.quiver(start_point[1], start_point[2], start_point[0],
              end_point[1] - start_point[1],
              end_point[2] - start_point[2],
              end_point[0] - start_point[0],
              color='black')

# Настройка осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
