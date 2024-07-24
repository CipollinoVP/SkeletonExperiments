import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pygltflib import GLTF2


# Загрузка GLB файла
def load_gltf(filename):
    gltf = GLTF2().load(filename)
    return gltf


# Извлечение данных о скелете и анимации
def extract_skeleton_and_animation(gltf):
    nodes = gltf.nodes
    animations = gltf.animations

    # Для простоты, извлекаем первую анимацию и первый канал
    animation = animations[0]
    channel = animation.channels[0]
    sampler = animation.samplers[channel.sampler]

    times = np.array(sampler.input)
    translations = np.array(sampler.output).reshape(-1, len(nodes), 3)

    return nodes, times, translations


# Визуализация с использованием matplotlib
def animate_skeleton(nodes, times, translations):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    scatter = ax.scatter([], [])

    def update(frame):
        coords = translations[frame]
        scatter.set_offsets(coords)
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)
    plt.show()


# Основная функция
def main():
    filename = 'model_low.glb'  # Замените на ваш GLB файл
    gltf = load_gltf(filename)
    nodes, times, translations = extract_skeleton_and_animation(gltf)
    animate_skeleton(nodes, times, translations)


if __name__ == '__main__':
    main()