
# 📸 Ứng Dụng Nhận Diện Khuôn Mặt bằng PCA [NHÓM 19 - L05]

## Tác giả:
1. Nguyễn Đăng Gia Đạo
2. Bùi Trung Hải
3. Hà Ngọc Khải
4. Chu Trọng Khánh
5. Văn Nguyễn Hoài Oanh
6. Phạm Viết Trường
7. Ngô Diệu Vy
   
## 🧠 Giới thiệu

Đây là một ứng dụng nhận diện khuôn mặt thời gian thực sử dụng **Principal Component Analysis (PCA)** để so sánh khuôn mặt từ webcam với tập dữ liệu mẫu đã lưu. Giao diện đồ họa được xây dựng bằng Tkinter và thư viện xử lý ảnh sử dụng OpenCV.

---

## 🗂 Cấu trúc thư mục

```
├── app.py                # Giao diện ứng dụng chính
├── face_recognition.py  # Phát hiện và nhận diện khuôn mặt
├── data_loader.py       # Tải dữ liệu khuôn mặt từ thư mục
├── pca.py               # Triển khai PCA: tính vector trung bình, eigenfaces...
├── faces/               # Thư mục chứa ảnh khuôn mặt để huấn luyện
└── captured_face.png    # Ảnh khuôn mặt chụp gần nhất (được lưu sau khi nhận diện)
```

---

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện cần thiết
```bash
pip install numpy opencv-python Pillow
```

### 2. Chuẩn bị dữ liệu
- Tạo thư mục `faces/` trong cùng thư mục với `app.py`.
- Thêm các ảnh khuôn mặt vào `faces/`. Tên file sẽ được dùng làm nhãn (label), ví dụ:
  ```
  faces/
  ├── alice.jpg
  ├── bob.png
  ```

> **Lưu ý:** Ảnh phải có khuôn mặt rõ ràng để hệ thống phát hiện được. Cần ít nhất 2 khuôn mặt khác nhau.

### 3. Chạy ứng dụng
- **Cách chạy trên PyCharm**:
  1. Mở PyCharm và tạo một dự án Python mới.
  2. Đặt các file mã nguồn vào thư mục dự án.
  3. Cài đặt các thư viện yêu cầu bằng cách vào **File > Settings > Project: [Tên dự án] > Python Interpreter**, sau đó tìm và cài đặt `numpy`, `opencv-python`, và `Pillow`.
  4. Chạy file `app.py` bằng cách nhấn nút **Run** hoặc **Shift + F10** trong PyCharm.

- Giao diện Tkinter sẽ hiện lên.
- Nhấn nút **"Chụp ảnh"** để hệ thống chụp khung hình và thực hiện nhận diện.
- Kết quả sẽ hiển thị ở dưới khung hình.

---

## ⚙️ Chi tiết kỹ thuật

- **Nhận diện khuôn mặt**: Dùng `cv2.CascadeClassifier` với mô hình `haarcascade_frontalface_default.xml`.
- **Tiền xử lý**:
  - Resize ảnh về kích thước chuẩn `100x100`.
  - Chuyển sang ảnh grayscale.
- **Giảm chiều PCA**:
  - Tính trung bình các vector ảnh.
  - Chuẩn hóa ảnh bằng cách trừ trung bình.
  - Tính ma trận hiệp phương sai và thực hiện phân rã trị riêng.
- **So sánh ảnh**: Dựng ảnh mới vào không gian eigenfaces và đo khoảng cách Euclidean với các mẫu đã biết.
- **Ngưỡng phân biệt**: `threshold=4000` (có thể điều chỉnh để tăng/giảm độ nhạy).

---

## 📸 Output

- Ảnh khuôn mặt đã chụp sẽ được lưu tự động vào file `captured_face.png`.
- Nếu không tìm thấy khuôn mặt, hệ thống sẽ hiển thị cảnh báo.

---

## ✅ Tác giả & Đóng góp

Nếu bạn muốn đóng góp:
1. Fork dự án
2. Tạo branch mới: `git checkout -b feature/ten-chuc-nang`
3. Commit: `git commit -m "Thêm chức năng"`
4. Push lên: `git push origin feature/ten-chuc-nang`
5. Tạo Pull Request

---

## 🧪 Phát triển thêm (gợi ý)
- Lưu kết quả nhận diện kèm timestamp.
- Thêm chức năng đăng ký khuôn mặt mới.
- Huấn luyện nâng cao với mô hình học sâu như FaceNet, dlib...
