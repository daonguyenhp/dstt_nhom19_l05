import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
from data_loader import load_faces
from pca import mean_face, center_faces, covariance_matrix, eigen_decomposition
from face_recognition import detect_face, recognize_face

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load dữ liệu training
        self.faces, self.labels = load_faces('faces')
        self.mean_face_vector = mean_face(self.faces)
        centered_faces = center_faces(self.faces, self.mean_face_vector)
        cov_matrix = covariance_matrix(centered_faces)
        eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)
        self.eigenvectors = eigenvectors
        self.k = 50

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_snapshot = Button(window, text="Chụp ảnh", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.label_info = Label(window, text="Thông tin đối tượng:", font=("Arial", 16))
        self.label_info.pack()

        self.current_frame = None  # để lưu khung hình mới nhất
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.current_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ viền xanh quanh mặt

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def snapshot(self):
        if self.current_frame is not None:
            frame = self.current_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]  # lấy khuôn mặt đầu tiên

                # --- CẮT VÙNG KHUÔN MẶT TRONG VIỀN XANH ---
                face_crop_color = frame[y:y + h, x:x + w]  # giữ màu

                # Resize nếu muốn (ví dụ chuẩn hóa 100x100)
                face_crop_color = cv2.resize(face_crop_color, (100, 100))

                # Nếu bạn dùng PCA grayscale, convert về gray:
                face_crop_gray = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2GRAY)
                face_vector = face_crop_gray.flatten()

                # Nhận diện PCA
                prediction = recognize_face(face_vector, self.faces, self.labels, self.mean_face_vector,
                                            self.eigenvectors, self.k, threshold=4000)

                self.label_info.config(text=f"Thông tin: {prediction}")

                # --- Hiển thị vùng mặt cắt lên canvas ---
                face_crop_rgb = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(face_crop_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # --- LƯU FILE CHỈ VÙNG MẶT ---
                cv2.imwrite("captured_face.png", cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB))

            else:
                self.label_info.config(text="Không tìm thấy khuôn mặt!")

    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    FaceRecognitionApp(tk.Tk(), "Ứng dụng nhận diện khuôn mặt PCA")


