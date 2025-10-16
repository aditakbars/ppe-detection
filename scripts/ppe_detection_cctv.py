# ppe_monitor_openvino_fallback.py
from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ---------- CONFIG ----------
MODEL_PT_PATH = "../models/ppe_model_v1/ppe_best_model_10epoch.pt"
USE_OPENVINO = True   # coba pakai OpenVINO jika tersedia
IMG_SIZE = 416        # gunakan 320-416 untuk laptop pas-pasan
FRAME_SKIP = 2        # proses 1 dari (FRAME_SKIP+1)
CONF_DEFAULT = 0.5
# --------------------------------

# Screenshot settings (sama seperti kode kamu)
SCREENSHOT_COOLDOWN = 5
MIN_VIOLATION_DURATION = 1

# folders
violations_folder = "../PPE-DETECTION/violations"
os.makedirs(violations_folder, exist_ok=True)

# Load model (initial PyTorch .pt)
print(f"Loading PyTorch model: {MODEL_PT_PATH}")
model = YOLO(MODEL_PT_PATH)
print("PyTorch model loaded.")

# Try to export to OpenVINO (if requested and not already exported)
openvino_model_path = None
if USE_OPENVINO:
    try:
        print("Attempting to export model to OpenVINO format (this may take a while)...")
        # export returns path to exported model directory / file
        ov_path = model.export(format="openvino", dynamic=True, half=True)
        # ultralytics export may return a path string or Path-like
        openvino_model_path = str(ov_path)
        print("OpenVINO export finished:", openvino_model_path)
    except Exception as e:
        print("⚠️ OpenVINO export failed or OpenVINO not installed:", e)
        openvino_model_path = None

# Try to load OpenVINO model via ultralytics wrapper (if export succeeded)
use_ov_runtime = False
if openvino_model_path:
    try:
        print("Trying to use exported OpenVINO model for inference...")
        ov_model = YOLO(openvino_model_path)  # Ultralytics supports predict on exported models
        # quick predict dry-run to validate
        _ = ov_model.predict(source=None, conf=CONF_DEFAULT, imgsz=IMG_SIZE)
        print("✅ OpenVINO-backed model ready.")
        use_ov_runtime = True
        model = ov_model  # replace model with the OpenVINO-loaded one
    except Exception as e:
        print("⚠️ Loading exported OpenVINO model failed — will fallback to PyTorch/CPU. Error:", e)
        use_ov_runtime = False

# Print classes
ALL_CLASSES = list(model.names.values())
print("Monitoring classes:", ALL_CLASSES)
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']

# Camera selection (sama seperti sebelumnya; env var CAMERA_SOURCE)
def get_camera_source():
    camera_source = os.getenv('CAMERA_SOURCE', 'webcam').lower()
    if camera_source == 'cctv':
        return os.getenv('CCTV_URL')
    if camera_source == 'webcam':
        return 0
    if camera_source.isdigit():
        return int(camera_source)
    return camera_source

camera_source = get_camera_source()
if camera_source is None:
    print("Camera source invalid. Check .env")
    exit(1)

# VideoCapture
cap = cv2.VideoCapture(camera_source)
if isinstance(camera_source, str) and camera_source.startswith('rtsp'):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
if not cap.isOpened():
    print("Cannot open camera/stream")
    exit(1)

# tracker simplified (copy of your ViolationTracker but minimal for brevity)
class ViolationTracker:
    def __init__(self):
        self.last_screenshot_time = {}
        self.violation_start_time = {}
        self.current_violations = set()
        self.total_screenshots = 0
    def update_violations(self, detected):
        current = set(detected)
        new = current - self.current_violations
        resolved = self.current_violations - current
        now = time.time()
        for v in new:
            self.violation_start_time[v] = now
        for v in resolved:
            self.violation_start_time.pop(v, None)
            self.last_screenshot_time.pop(v, None)
        self.current_violations = current
        return new, resolved
    def should_screenshot(self, v, now):
        if v in self.violation_start_time:
            if now - self.violation_start_time[v] < MIN_VIOLATION_DURATION:
                return False, "duration_too_short"
        if v in self.last_screenshot_time:
            if now - self.last_screenshot_time[v] < SCREENSHOT_COOLDOWN:
                return False, "cooldown"
        return True, "ok"
    def record_screenshot(self, v):
        self.last_screenshot_time[v] = time.time()
        self.total_screenshots += 1

tracker = ViolationTracker()

# Display window
cv2.namedWindow('PPE Detection - Safety Monitoring', cv2.WINDOW_NORMAL)

frame_id = 0
confidence_threshold = CONF_DEFAULT
prev_time = time.time()
fps = 0.0
total_detections = 0
violation_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(camera_source)
            continue

        frame_id += 1
        do_infer = (FRAME_SKIP == 0) or (frame_id % (FRAME_SKIP + 1) == 1)

        # Resize for inference to IMG_SIZE (maintain aspect)
        h, w = frame.shape[:2]
        scale = 1.0
        if max(w, h) > IMG_SIZE:
            scale = IMG_SIZE / max(w, h)
            frame_in = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_in = frame.copy()

        annotated_frame = frame_in.copy()

        if do_infer:
            # If using OpenVINO-backed model, ultralytics will use optimized runtime
            # Otherwise model will run on CPU (PyTorch). We pass imgsz and conf for speed.
            results = model.predict(frame_in, conf=confidence_threshold, imgsz=IMG_SIZE, verbose=False)
            # draw output
            try:
                annotated_frame = results[0].plot()
            except Exception:
                annotated_frame = frame_in.copy()
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if conf < confidence_threshold:
                            continue
                        name = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
                        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(annotated_frame, f"{name} {conf:.2f}", (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # analyze detections
            detected_violations = []
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                if class_name in VIOLATION_CLASSES:
                    detected_violations.append(class_name)
                total_detections += 1

            new_v, res_v = tracker.update_violations(detected_violations)
            if new_v:
                print("New:", new_v)
            if res_v:
                print("Resolved:", res_v)

            # screenshot logic
            now = time.time()
            for v in tracker.current_violations:
                ok, reason = tracker.should_screenshot(v, now)
                if ok:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = f"{violations_folder}/violation_{v}_{ts}.jpg"
                    cv2.imwrite(fn, annotated_frame)
                    tracker.record_screenshot(v)
                    violation_count += 1
                    print("Saved", fn)

        # UI overlays simplified
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated_frame, f"Conf: {confidence_threshold:.2f}", (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # show scaled to screen if needed
        screen_max_w, screen_max_h = 1280, 720
        h2, w2 = annotated_frame.shape[:2]
        scale_w = screen_max_w / w2
        scale_h = screen_max_h / h2
        scale_display = min(scale_w, scale_h, 1.0)
        if scale_display < 1.0:
            disp = cv2.resize(annotated_frame, (int(w2*scale_display), int(h2*scale_display)))
        else:
            disp = annotated_frame

        cv2.imshow('PPE Detection - Safety Monitoring', disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            fn = f"{violations_folder}/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, annotated_frame)
            print("Manual saved", fn)
        elif k in (ord('+'), ord('=')):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print("Conf ->", confidence_threshold)
        elif k in (ord('-'), ord('_')):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print("Conf ->", confidence_threshold)

except KeyboardInterrupt:
    print("Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Session summary:")
    print("Frames:", frame_id, "Detections:", total_detections, "Screenshots:", tracker.total_screenshots,
          "Violations saved:", violation_count)
