import cv2
import numpy as np
from pca import project_face

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        return face.flatten()
    return None


def recognize_face(face_vector, faces, labels, mean_face_vector, eigenvectors, k, threshold=4000):
    proj = project_face(face_vector, mean_face_vector, eigenvectors, k)
    min_dist = float('inf')
    identity = "Không nhận diện được"

    for i, known_face in enumerate(faces):
        known_proj = project_face(known_face, mean_face_vector, eigenvectors, k)
        dist = np.linalg.norm(proj - known_proj)
        if dist < min_dist:
            min_dist = dist
            identity = labels[i]

    # Áp dụng threshold
    if min_dist > threshold:
        identity = "Không nhận diện được"

    return identity

