import cv2
import mediapipe as mp
import numpy as np

# Инициализация необходимых компонентов Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils


# Функция для извлечения координат позы из видео
def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose_landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра для получения позы
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
            pose_landmarks_list.append(pose_landmarks)
        else:
            # Если поза не обнаружена, добавляем пустой список
            pose_landmarks_list.append([])

    cap.release()
    return np.array(pose_landmarks_list)


# Путь к вашему видеофайлу
video_path = './videos/nothing.mp4'
pose_data = extract_pose_from_video(video_path)

# Сохранение numpy массива в файл, если это необходимо
np.save('pose_data.npy', pose_data)

print("Позы успешно извлечены и сохранены в виде numpy массив")

