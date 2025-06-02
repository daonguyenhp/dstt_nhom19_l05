import cv2
import os
import numpy as np

def load_faces(folder_path, size=(100, 100)):
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]  # lấy khuôn mặt đầu tiên
                face_crop = gray[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, size)

                faces.append(face_crop.flatten())
                labels.append(os.path.splitext(filename)[0])
            else:
                print(f"⚠️ Không tìm thấy khuôn mặt trong {filename}, bỏ qua.")

    if len(faces) < 2:
        raise ValueError("❌ Cần ít nhất 2 khuôn mặt hợp lệ trong thư mục faces/ để dùng PCA.")

    return np.array(faces), labels
