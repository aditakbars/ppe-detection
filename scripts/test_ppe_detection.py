from ultralytics import YOLO
import cv2
import time

print("="*60)
print("🏭 PPE DETECTION - Real-Time Testing")
print("="*60)

# Load trained model
model_path = "../models/ppe_model_v1/ppe_best_model_3epoch.pt"
print(f"\n📥 Loading model: {model_path}")
model = YOLO(model_path)
print("✅ Model loaded!")

# Class names yang kita peduliin
important_classes = ['helmet', 'no_helmet', 'glove', 'no_glove']

# Buka webcam
cap = cv2.VideoCapture(0)
print("\n🎥 Starting webcam...")
print("💡 Tips: Coba pakai/lepas topi buat simulasi helm!")
print("⚠️  Tekan 'q' untuk stop\n")

# Untuk hitung FPS
prev_time = time.time()

# Counter violations
violation_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Jalankan deteksi
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Gambar hasil deteksi
    annotated_frame = results[0].plot()
    
    # Check violations
    detected_objects = results[0].boxes
    has_violation = False
    violation_text = []
    
    for box in detected_objects:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        
        # Cek kalau ada no_helmet atau no_glove
        if class_name == 'no_helmet':
            has_violation = True
            violation_text.append(f"⚠️ NO HELMET! ({confidence:.0%})")
        elif class_name == 'no_glove':
            has_violation = True
            violation_text.append(f"⚠️ NO GLOVES! ({confidence:.0%})")
    
    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Tampilin FPS
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilin violation warning (kalau ada)
    if has_violation:
        violation_count += 1
        y_pos = 70
        for vtext in violation_text:
            cv2.putText(annotated_frame, vtext, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 40
        
        # Tambahin border merah
        cv2.rectangle(annotated_frame, (0, 0), 
                     (annotated_frame.shape[1], annotated_frame.shape[0]), 
                     (0, 0, 255), 10)
    
    # Tampilin violation counter
    cv2.putText(annotated_frame, f'Violations: {violation_count}', (10, annotated_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('PPE Detection - Safety Monitoring', annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Webcam stopped!")
print(f"📊 Total violations detected: {violation_count}")
print("="*60)