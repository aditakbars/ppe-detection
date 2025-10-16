# ppe_screenshot_only.py
from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables (optional)
load_dotenv()

# CONFIG
model_path = "../models/ppe_model_v1/ppe_best_model_10epoch.pt"
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']

# Screenshot policy
SCREENSHOT_COOLDOWN = 5        # seconds between screenshots for same violation
MIN_VIOLATION_DURATION = 1.0   # seconds violation must persist before screenshot
SAVE_CROPS = True              # also save cropped bbox images per violation class
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# Folders
violations_folder = "../PPE-DETECTION/violations"
crops_folder = os.path.join(violations_folder, "crops")
os.makedirs(violations_folder, exist_ok=True)
if SAVE_CROPS:
    os.makedirs(crops_folder, exist_ok=True)
    for cls in VIOLATION_CLASSES:
        os.makedirs(os.path.join(crops_folder, cls), exist_ok=True)

# Load model
print(f"Loading model: {model_path}")
model = YOLO(model_path)
print("Model loaded.")
ALL_CLASS_NAMES = list(model.names.values())
print("Model classes:", ALL_CLASS_NAMES)

# Helper: get camera source from env or default to webcam
def get_camera_source():
    source = os.getenv("CAMERA_SOURCE", "webcam").lower()
    if source == "webcam":
        return 0
    if source == "cctv":
        url = os.getenv("CCTV_URL")
        return url
    if source.isdigit():
        return int(source)
    return source

camera_source = get_camera_source()
if camera_source is None:
    print("Camera source not configured. Set CAMERA_SOURCE or ensure .env present.")
    exit(1)

# Open stream
cap = cv2.VideoCapture(camera_source)
if isinstance(camera_source, str) and camera_source.startswith("rtsp"):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Cannot open camera/source:", camera_source)
    exit(1)

# Simple Violation tracker (same logic, minimal)
class ViolationTracker:
    def __init__(self):
        self.last_screenshot_time = {}    # {violation_type: timestamp}
        self.violation_start_time = {}    # {violation_type: timestamp}
        self.current_violations = set()
        self.total_screenshots = 0

    def update_violations(self, detected_violations):
        current_set = set(detected_violations)
        new_violations = current_set - self.current_violations
        resolved_violations = self.current_violations - current_set

        now = time.time()
        for v in new_violations:
            self.violation_start_time[v] = now

        for v in resolved_violations:
            self.violation_start_time.pop(v, None)
            self.last_screenshot_time.pop(v, None)

        self.current_violations = current_set
        return new_violations, resolved_violations

    def should_screenshot(self, violation_type, current_time):
        # check duration
        start = self.violation_start_time.get(violation_type)
        if start is None:
            return False, "no_start_time"
        if (current_time - start) < MIN_VIOLATION_DURATION:
            return False, "duration_too_short"

        # check cooldown
        last = self.last_screenshot_time.get(violation_type)
        if last is not None and (current_time - last) < SCREENSHOT_COOLDOWN:
            return False, "cooldown"

        return True, "ok"

    def record_screenshot(self, violation_type):
        self.last_screenshot_time[violation_type] = time.time()
        self.total_screenshots += 1

tracker = ViolationTracker()

# Minimal UI settings
window_name = "CCTV - PPE Screenshot Collector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Run loop
confidence_threshold = 0.5
frame_count = 0
total_detections = 0
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # try reconnect once
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(camera_source)
            time.sleep(0.5)
            continue

        frame_count += 1
        now = time.time()

        # Run detection (single-frame detect)
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        annotated = results[0].plot()

        # collect detected violation classes and also keep their boxes for crops
        detected_violations = []
        violation_boxes = []  # list of (class_name, xyxy)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            total_detections += 1
            if cls_name in VIOLATION_CLASSES:
                detected_violations.append(cls_name)
                # xyxy as ints
                xyxy = tuple(map(int, box.xyxy[0].tolist()))
                violation_boxes.append((cls_name, xyxy))

        # update tracker state
        new_v, resolved_v = tracker.update_violations(detected_violations)

        # For each currently active violation, decide whether to screenshot
        for v in sorted(tracker.current_violations):
            ok, reason = tracker.should_screenshot(v, now)
            if not ok:
                continue

            # Save full annotated frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = os.path.join(violations_folder, f"violation_{v}_{ts}.jpg")
            cv2.imwrite(fname, annotated)

            # Save crops for matching boxes of this class (if enabled)
            if SAVE_CROPS:
                # find all boxes for this violation in the current frame
                idx = 0
                for cls_name, xyxy in violation_boxes:
                    if cls_name != v:
                        continue
                    x1, y1, x2, y2 = xyxy
                    # clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_ts = f"{ts}_{idx}"
                    crop_name = os.path.join(crops_folder, cls_name, f"crop_{cls_name}_{crop_ts}.jpg")
                    cv2.imwrite(crop_name, crop)
                    idx += 1

            tracker.record_screenshot(v)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Screenshot saved for '{v}': {fname}")

        # Resize annotated frame for display (keep aspect)
        h, w = annotated.shape[:2]
        scale_w = MAX_DISPLAY_WIDTH / w
        scale_h = MAX_DISPLAY_HEIGHT / h
        scale = min(scale_w, scale_h, 1.0)
        if scale < 1.0:
            annotated_disp = cv2.resize(annotated, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            annotated_disp = annotated

        # Small timestamp overlay (optional)
        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_disp, ts_text, (10, annotated_disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(window_name, annotated_disp)

        # minimal key handling: q to quit, s manual save
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping.")
            break
        elif key == ord('s'):
            # manual save full frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = os.path.join(violations_folder, f"manual_{ts}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"Manual screenshot saved: {fname}")

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Session summary:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections processed (approx): {total_detections}")
    print(f"  Total screenshots saved: {tracker.total_screenshots}")
    print("Done.")
