import cv2
import os

# Xác định đường dẫn đến file cascade
cascade_path = os.path.join(os.getcwd(), 'cascades', 'haarcascade_frontalface_default.xml')

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Không tìm thấy file Haar Cascade tại: {cascade_path}")

# Khởi tạo bộ nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cascade_path)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
