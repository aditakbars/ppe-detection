# ppe_monitor_fully_optimized.py
from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from threading import Thread
from queue import Queue, Empty

load_dotenv()

# ========== OPTIMIZED CONFIG ==========
MODEL_PT_PATH = "../models/ppe_model_v1/ppe_best_model_10epoch.pt"
USE_OPENVINO = True
IMG_SIZE = 544        # Turunkan dari 416 ke 320 untuk Intel UHD. Coba 256 jika perlu lebih cepat.
CONF_DEFAULT = 0.45   # Sedikit turunkan untuk speed
# FRAME_SKIP dihapus, karena kita akan memproses secepatnya di thread terpisah.

# Quality vs Speed balance
IOU_THRESHOLD = 0.5   # NMS threshold
MAX_DETECTIONS = 100  # Limit detections per frame

# Screenshot settings
SCREENSHOT_COOLDOWN = 5
MIN_VIOLATION_DURATION = 1

# Folders
violations_folder = "../PPE-DETECTION/violations"
os.makedirs(violations_folder, exist_ok=True)

# ========== THREADED FRAME READER ==========
class VideoCapture:
    """Threaded video capture untuk menghindari I/O blocking"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if isinstance(src, str) and src.startswith('rtsp'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) # Buffer sedikit lebih banyak untuk stream
        self.q = Queue(maxsize=2)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True # Hentikan jika stream berakhir
                    continue
                self.q.put(frame)
            else:
                time.sleep(0.01) # Tunggu sebentar jika queue penuh
                
    def read(self):
        try:
            return self.q.get(timeout=1) # Tunggu hingga 1 detik untuk frame baru
        except Empty:
            return None
        
    def release(self):
        self.stopped = True
        self.cap.release()

# ========== MODEL LOADING (Sama seperti kodemu, tidak diubah) ==========
print(f"[1/4] Loading PyTorch model: {MODEL_PT_PATH}")
model = YOLO(MODEL_PT_PATH)
print("✓ PyTorch model loaded")

openvino_model_path = None
if USE_OPENVINO:
    try:
        print("[2/4] Exporting to OpenVINO (one-time process)...")
        ov_path = model.export(format="openvino", dynamic=False, half=True, imgsz=IMG_SIZE)
        openvino_model_path = str(ov_path)
        print(f"✓ OpenVINO exported: {openvino_model_path}")
    except Exception as e:
        print(f"⚠️ OpenVINO export failed: {e}")
        openvino_model_path = None

use_ov = False
inference_device = 'cpu'
if openvino_model_path:
    try:
        print("[3/4] Loading OpenVINO model...")
        model = YOLO(openvino_model_path, task='detect')
        print("[4/4] Testing OpenVINO inference...")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        test_devices = ['AUTO', 'GPU', 'CPU']
        device_success = False
        for dev in test_devices:
            try:
                print(f"   Trying device: {dev}...")
                _ = model.predict(dummy, device=dev, verbose=False, imgsz=IMG_SIZE, conf=CONF_DEFAULT, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS)
                inference_device = dev
                device_success = True
                print(f"   ✓ Success with device: {dev}")
                break
            except Exception as e:
                print(f"   ✗ Failed with {dev}: {str(e)[:50]}...")
        if device_success:
            print(f"✓ OpenVINO model ready on device: {inference_device}")
            use_ov = True
        else:
            raise Exception("All devices failed")
    except Exception as e:
        print(f"⚠️ OpenVINO initialization failed: {e}. Falling back to PyTorch CPU...")
        model = YOLO(MODEL_PT_PATH)
        inference_device = 'cpu'
else:
    print("[3/4] Using PyTorch model on CPU")
    inference_device = 'cpu'

ALL_CLASSES = list(model.names.values())
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']
print(f"\n📋 Monitoring classes: {ALL_CLASSES}")
print(f"⚠️  Violation classes: {VIOLATION_CLASSES}\n")


# ========== VIOLATION TRACKER (Sama seperti kodemu, tidak diubah) ==========
class ViolationTracker:
    def __init__(self):
        self.last_screenshot_time = {}
        self.violation_start_time = {}
        self.current_violations = set()
        self.total_screenshots = 0
    def update_violations(self, detected):
        current, now = set(detected), time.time()
        new, resolved = current - self.current_violations, self.current_violations - current
        for v in new: self.violation_start_time[v] = now
        for v in resolved: self.violation_start_time.pop(v, None); self.last_screenshot_time.pop(v, None)
        self.current_violations = current
        return new, resolved
    def should_screenshot(self, v, now):
        if v in self.violation_start_time and now - self.violation_start_time[v] < MIN_VIOLATION_DURATION: return False, "duration_too_short"
        if v in self.last_screenshot_time and now - self.last_screenshot_time[v] < SCREENSHOT_COOLDOWN: return False, "cooldown"
        return True, "ok"
    def record_screenshot(self, v):
        self.last_screenshot_time[v] = time.time()
        self.total_screenshots += 1

# ========== INFERENCE WORKER THREAD ==========
frame_queue = Queue(maxsize=2)
results_queue = Queue(maxsize=2)
inference_stopped = False

def inference_worker(model, device, use_ov, confidence):
    """Fungsi yang berjalan di thread terpisah untuk inferensi"""
    while not inference_stopped:
        try:
            original_frame, frame_resized = frame_queue.get(timeout=1)
        except Empty:
            continue

        try:
            results = model.predict(
                frame_resized, 
                device=device,
                conf=confidence, 
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                max_det=MAX_DETECTIONS,
                verbose=False,
                half=use_ov
            )
            # Taruh frame ASLI dan hasilnya ke queue
            results_queue.put((original_frame, frame_resized, results))
        except Exception as e:
            print(f"⚠️ Inference error in worker: {e}")
            continue

# ========== MAIN APPLICATION ==========
def main():
    global inference_stopped, confidence_threshold
    
    # Inisialisasi
    confidence_threshold = CONF_DEFAULT
    tracker = ViolationTracker()
    
    # Setup Camera
    camera_source = os.getenv('CCTV_URL') if os.getenv('CAMERA_SOURCE', '').lower() == 'cctv' else 0
    print(f"📹 Connecting to: {camera_source}")
    cap = VideoCapture(camera_source).start()
    time.sleep(2.0)
    
    # Start Inference Thread
    inference_thread = Thread(
        target=inference_worker, 
        args=(model, inference_device, use_ov, confidence_threshold), 
        daemon=True
    )
    inference_thread.start()

    cv2.namedWindow('PPE Detection - Safety Monitoring', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('PPE Detection - Safety Monitoring', 1280, 720)

    print("\n" + "="*60 + "\n🚀 STARTING MONITORING\n" + "="*60)
    print(f"Inference Device: {inference_device} {'(OpenVINO)' if use_ov else '(PyTorch)'}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("="*60 + "\n")

    # Variabel untuk statistik
    prev_time = time.time()
    fps_samples = []
    frame_id = 0
    
    try:
        while True:
            # 1. Baca frame dari kamera (non-blocking)
            original_frame = cap.read()
            if original_frame is None:
                print("Camera stream ended or failed. Exiting.")
                break
            
            # 2. Pre-process dan kirim ke thread inferensi jika queue tidak penuh
            if not frame_queue.full():
                h, w = original_frame.shape[:2]
                scale = IMG_SIZE / max(w, h)
                frame_resized = cv2.resize(original_frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                frame_queue.put((original_frame, frame_resized))

            # 3. Ambil hasil dari thread inferensi (jika ada)
            try:
                processed_original_frame, display_frame, results = results_queue.get_nowait()
                frame_id += 1
                
                # 4. Gambar hasil deteksi
                detected_violations = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf, cls_id = float(box.conf[0]), int(box.cls[0])
                    class_name = model.names[cls_id]
                    color = (0, 0, 255) if class_name in VIOLATION_CLASSES else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, y1-th-4), (x1+tw, y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    if class_name in VIOLATION_CLASSES: detected_violations.append(class_name)
                
                # 5. Logika screenshot (menggunakan frame asli berkualitas tinggi)
                tracker.update_violations(detected_violations)
                now = time.time()
                for v in tracker.current_violations:
                    if tracker.should_screenshot(v, now)[0]:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fn = f"{violations_folder}/violation_{v}_{ts}.jpg"
                        cv2.imwrite(fn, processed_original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        tracker.record_screenshot(v)
                        print(f"📸 Saved: {fn}")

                # Hitung FPS berdasarkan seberapa cepat kita bisa menampilkan hasil
                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
                prev_time = now
                fps_samples.append(fps)
                if len(fps_samples) > 30: fps_samples.pop(0)
                avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0

                # UI Overlay
                fps_color = (0, 255, 0) if avg_fps >= 25 else (0, 255, 255) if avg_fps >= 15 else (0, 0, 255)
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                cv2.putText(display_frame, f"Conf: {confidence_threshold:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if tracker.current_violations:
                    cv2.putText(display_frame, f"ALERT: {len(tracker.current_violations)} Violations", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('PPE Detection - Safety Monitoring', display_frame)

            except Empty:
                # Jika tidak ada hasil baru, jangan lakukan apa-apa.
                # Ini akan membuat loop berjalan cepat dan responsif terhadap input keyboard.
                pass

            # Keyboard controls
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            # Tambahkan kontrol lain jika perlu

    finally:
        print("Stopping threads and cleaning up...")
        inference_stopped = True
        inference_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()