from ultralytics import YOLO
import cv2
import time

# Load model
print("📥 Loading YOLOv11 nano model...")
model = YOLO('yolo11n.pt')
print("✅ Model loaded!")

# Buka webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting webcam... (tekan 'q' untuk stop)")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Jalankan deteksi
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Gambar hasil deteksi
    annotated_frame = results[0].plot()
    
    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Tampilin FPS di video
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam stopped!")