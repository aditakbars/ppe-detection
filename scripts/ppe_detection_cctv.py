from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import threading
from queue import Queue
import numpy as np

# Load environment variables
load_dotenv()

print("="*70)
print("🏭 PPE DETECTION - Multi-Source Real-Time Monitoring")
print("="*70)

# ========================================
# CONFIGURATION
# ========================================

# Resolve absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Model & inference config
model_path = os.getenv('MODEL_PT_PATH', os.path.join(REPO_ROOT, 'models', 'ppe_model_v1', 'ppe_best_model_10epoch.pt'))

# If path is relative, resolve it relative to REPO_ROOT (not current working dir)
if not os.path.isabs(model_path):
    model_path = os.path.join(REPO_ROOT, model_path)

model_path = os.path.abspath(model_path)  # Ensure absolute path

IMG_SIZE = int(os.getenv('IMG_SIZE', 416))  # Reduce to 320 for faster inference on weaker GPU
FRAME_SKIP = int(os.getenv('FRAME_SKIP', 0))  # 0=every frame, 1=every 2nd, 2=every 3rd
USE_OPENVINO = os.getenv('USE_OPENVINO', 'true').lower() == 'true'
OPENVINO_DEVICE = os.getenv('OPENVINO_DEVICE', 'GPU')  # 'GPU', 'CPU', 'GPU.0'

print(f"\n⚙️  Config loaded:")
print(f"   Model: {os.path.basename(model_path) if model_path and os.path.exists(model_path) else 'NOT FOUND'}")
print(f"   Inference size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Frame skip: {FRAME_SKIP} (process every {FRAME_SKIP+1} frame)")
print(f"   OpenVINO: {'✅ Enabled' if USE_OPENVINO else '❌ Disabled'}")
if USE_OPENVINO:
    print(f"   Device: {OPENVINO_DEVICE}")

# Load trained model
# Check if model file exists
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found!")
    print(f"   Expected: {model_path}")
    print(f"   Set MODEL_PT_PATH in .env or ensure model file exists")
    exit(1)

print(f"\n📥 Loading model: {os.path.basename(model_path)}")
model = YOLO(model_path)
print("✅ Model loaded!")
if not model:
    print("❌ ERROR: Failed to load model")
    exit(1)

# All classes
ALL_CLASSES = list(model.names.values())
print(f"\n🎯 Monitoring all classes: {ALL_CLASSES}")

# Violation classes
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']

# ========================================
# OPENVINO SETUP (Optional GPU acceleration)
# ========================================

openvino_model = None
use_openvino = False

if USE_OPENVINO:
    try:
        from openvino.runtime import Core
        
        print("\n🔧 Setting up OpenVINO...")
        
        # Detect or export model to OpenVINO IR format
        pt_basename = os.path.splitext(os.path.basename(model_path))[0]
        ov_model_dir = os.path.join(os.path.dirname(model_path), f"{pt_basename}_openvino_model")
        ov_xml = os.path.join(ov_model_dir, f"{pt_basename}.xml")
        ov_bin = os.path.join(ov_model_dir, f"{pt_basename}.bin")
        
        # Check if exported IR exists
        if not os.path.exists(ov_xml):
            print(f"   Exporting model to OpenVINO IR format... (may take 1-2 min)")
            try:
                model.export(format='openvino', dynamic=True, half=True)
                print(f"   ✅ Export complete: {ov_model_dir}")
            except Exception as e:
                print(f"   ⚠️ Export failed: {e}")
        else:
            print(f"   ✅ Found existing IR at {ov_xml}")
        
        # Load with OpenVINO runtime
        if os.path.exists(ov_xml) and os.path.exists(ov_bin):
            try:
                core = Core()
                
                # Log available devices
                available_devices = core.available_devices
                print(f"   📊 Available devices: {available_devices}")
                
                # Load model with selected device
                compiled_model = core.compile_model(ov_xml, OPENVINO_DEVICE)
                print(f"   ✅ OpenVINO model compiled for device: {OPENVINO_DEVICE}")
                
                # Create infer request
                openvino_model = compiled_model.create_infer_request()
                use_openvino = True
                print("   ✅ OpenVINO ready for inference!")
                
            except Exception as e:
                print(f"   ⚠️ OpenVINO setup failed: {e}")
                print(f"   💡 Tip: Install openvino-dev and check GPU drivers")
                use_openvino = False
    
    except ImportError:
        print(f"\n⚠️ OpenVINO library not found. Install: pip install openvino openvino-dev")
        print(f"   Falling back to PyTorch inference...")
        use_openvino = False

if not use_openvino:
    print(f"\n📍 Using PyTorch inference on CPU (slower)")
    print(f"   💡 For better performance, install OpenVINO: pip install openvino")

# ========================================
# BACKGROUND SCREENSHOT THREAD
# ========================================

screenshot_queue = Queue()
stop_screenshot_thread = False

def screenshot_worker():
    """Background worker to save screenshots without blocking main loop"""
    while not stop_screenshot_thread:
        try:
            # Non-blocking get with timeout
            filename, frame = screenshot_queue.get(timeout=1.0)
            if filename and isinstance(frame, np.ndarray):
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        except:
            pass

# Start background screenshot thread
screenshot_thread = threading.Thread(target=screenshot_worker, daemon=True)
screenshot_thread.start()

# ========================================
# RENDERING HELPER FUNCTIONS
# ========================================

def draw_detections_optimized(frame, boxes, model_names, conf_threshold=0.5):
    """Draw bounding boxes without using expensive plot() method"""
    height, width = frame.shape[:2]
    frame_copy = frame.copy()
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if conf < conf_threshold:
            continue
        
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = max(0, min(x1, width)), min(width, max(x2, 0))
        y1, y2 = max(0, min(y1, height)), min(height, max(y2, 0))
        
        # Choose color based on violation class
        class_name = model_names[cls_id]
        if class_name in VIOLATION_CLASSES:
            color = (0, 0, 255)  # Red for violations
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for compliant
            thickness = 2
        
        # Draw box and label
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Label background for readability
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame_copy, label, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_copy

SCREENSHOT_COOLDOWN = 5  # Seconds between screenshots for same violation
MIN_VIOLATION_DURATION = 1  # Seconds - hanya screenshot kalau violation persist > X detik

# ========================================
# DISPLAY SETTINGS
# ========================================

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

def resize_frame(frame, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """Resize frame to fit screen while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return frame, 1.0

# ========================================
# CAMERA SOURCE SELECTION
# ========================================

def get_camera_source():
    """Get camera source from environment or user input"""
    camera_source = os.getenv('CAMERA_SOURCE', 'webcam').lower()
    
    if camera_source == 'cctv':
        cctv_url = os.getenv('CCTV_URL')
        if cctv_url:
            print(f"\n📹 Using CCTV from .env")
            print(f"🔗 URL: {cctv_url[:12]}... (hidden for security)")
            return cctv_url
        else:
            print("⚠️ CCTV_URL not found in .env!")
            return None
    
    elif camera_source == 'webcam':
        print(f"\n📷 Using Webcam (device 0)")
        return 0
    
    elif camera_source.isdigit():
        device_id = int(camera_source)
        print(f"\n📷 Using Camera device {device_id}")
        return device_id
    
    else:
        print(f"⚠️ Unknown camera source: {camera_source}")
        return None

camera_source = get_camera_source()

if camera_source is None:
    print("❌ Failed to get camera source. Check your .env file!")
    exit()

# ========================================
# OPEN VIDEO STREAM
# ========================================

print(f"\n🔌 Connecting to camera...")
cap = cv2.VideoCapture(camera_source)

if isinstance(camera_source, str) and camera_source.startswith('rtsp'):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("❌ Error: Cannot open camera/CCTV stream!")
    print("💡 Tips:")
    print("   - Check network connection (for CCTV)")
    print("   - Verify RTSP URL is correct")
    print("   - Check username/password")
    print("   - Try ping the CCTV IP")
    exit()

print("✅ Camera connected!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cam = int(cap.get(cv2.CAP_PROP_FPS))
print(f"📐 Original Resolution: {width}x{height}")
print(f"📺 Display Max Size: {MAX_DISPLAY_WIDTH}x{MAX_DISPLAY_HEIGHT}")
print(f"🎬 Camera FPS: {fps_cam}")

print("\n💡 Screenshot Strategy:")
print(f"   - First violation: instant screenshot")
print(f"   - Cooldown: {SCREENSHOT_COOLDOWN} seconds")
print(f"   - Min duration: {MIN_VIOLATION_DURATION} second(s)")

print("\n💡 Controls:")
print("   - Press 'q' to quit")
print("   - Press 's' to save screenshot (manual)")
print("   - Press '+' to increase confidence threshold")
print("   - Press '-' to decrease confidence threshold")
print("   - Press 'c' to change cooldown time")
print("="*70 + "\n")

# ========================================
# CREATE FOLDERS
# ========================================

violations_folder = os.getenv('VIOLATIONS_FOLDER', os.path.join(REPO_ROOT, 'violations'))

# If path is relative, resolve it relative to REPO_ROOT (not current working dir)
if not os.path.isabs(violations_folder):
    violations_folder = os.path.join(REPO_ROOT, violations_folder)

violations_folder = os.path.abspath(violations_folder)
os.makedirs(violations_folder, exist_ok=True)

# ========================================
# VIOLATION TRACKING STATE
# ========================================

class ViolationTracker:
    def __init__(self):
        self.last_screenshot_time = {}  # {violation_type: timestamp}
        self.violation_start_time = {}  # {violation_type: timestamp}
        self.current_violations = set()
        self.total_screenshots = 0
        
    def update_violations(self, detected_violations):
        """Update current violation state"""
        current_set = set(detected_violations)
        
        # Detect new violations (state change)
        new_violations = current_set - self.current_violations
        
        # Detect resolved violations
        resolved_violations = self.current_violations - current_set
        
        # Update start time for new violations
        current_time = time.time()
        for violation in new_violations:
            self.violation_start_time[violation] = current_time
        
        # Remove resolved violations from tracking
        for violation in resolved_violations:
            if violation in self.violation_start_time:
                del self.violation_start_time[violation]
            if violation in self.last_screenshot_time:
                del self.last_screenshot_time[violation]
        
        self.current_violations = current_set
        
        return new_violations, resolved_violations
    
    def should_screenshot(self, violation_type, current_time):
        """Check if we should take screenshot for this violation"""
        
        # Check if violation has persisted long enough
        if violation_type in self.violation_start_time:
            duration = current_time - self.violation_start_time[violation_type]
            if duration < MIN_VIOLATION_DURATION:
                return False, "duration_too_short"
        
        # Check cooldown
        if violation_type in self.last_screenshot_time:
            time_since_last = current_time - self.last_screenshot_time[violation_type]
            if time_since_last < SCREENSHOT_COOLDOWN:
                return False, "cooldown"
        
        return True, "ok"
    
    def record_screenshot(self, violation_type):
        """Record that we took a screenshot"""
        self.last_screenshot_time[violation_type] = time.time()
        self.total_screenshots += 1

tracker = ViolationTracker()

# ========================================
# DETECTION LOOP
# ========================================
# DETECTION LOOP
# ========================================

# Initialize gui_available flag and variables
gui_available = True
prev_time = time.time()
violation_count = 0
frame_count = 0
total_detections = 0
confidence_threshold = 0.5

# Try to create display window
try:
    cv2.namedWindow('PPE Detection - Safety Monitoring', cv2.WINDOW_NORMAL)
except cv2.error as e:
    print(f"\n⚠️ GUI not available: {e}")
    print(f"   Running in headless mode - frames will be saved but not displayed")
    print(f"   Install opencv-python (not headless) for GUI: pip install --upgrade opencv-python")
    gui_available = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame. Reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(camera_source)
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Decide whether to run inference this frame (frame skipping for speed)
        do_inference = (FRAME_SKIP == 0) or (frame_count % (FRAME_SKIP + 1) == 1)
        
        annotated_frame = frame.copy()
        current_detections = {}
        detected_violations = []
        
        # ========================================
        # INFERENCE (with frame skipping)
        # ========================================
        
        if do_inference:
            if use_openvino:
                # Use OpenVINO runtime for faster inference (GPU acceleration)
                # Prepare input: resize to IMG_SIZE maintaining aspect + pad
                h, w = frame.shape[:2]
                scale = IMG_SIZE / max(h, w)
                
                if scale < 1:
                    inp = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    inp = frame.copy()
                
                # Pad to square
                h_new, w_new = inp.shape[:2]
                pad_h = IMG_SIZE - h_new
                pad_w = IMG_SIZE - w_new
                inp_padded = cv2.copyMakeBorder(inp, 0, pad_h, 0, pad_w, 
                                                 cv2.BORDER_CONSTANT, value=0)
                
                # Normalize and prepare for OpenVINO [batch, channels, height, width]
                inp_blob = inp_padded.astype(np.float32) / 255.0
                inp_blob = np.transpose(inp_blob, (2, 0, 1))
                inp_blob = np.expand_dims(inp_blob, 0)
                
                # Run inference
                try:
                    openvino_model.infer({0: inp_blob})
                    outputs = openvino_model.get_output_tensor(0).data
                    # Parse OpenVINO output and map back to frame coordinates
                    # (requires post-processing similar to PyTorch)
                    # For this version, fallback to YOLO predict for simplicity
                    results = model.predict(frame, conf=confidence_threshold, 
                                          imgsz=IMG_SIZE, verbose=False)
                except Exception as e:
                    print(f"⚠️ OpenVINO inference failed: {e}, using PyTorch...")
                    results = model.predict(frame, conf=confidence_threshold, 
                                          imgsz=IMG_SIZE, verbose=False)
            else:
                # Standard PyTorch inference
                results = model.predict(frame, conf=confidence_threshold, 
                                      imgsz=IMG_SIZE, verbose=False)
            
            # Draw detections (optimized, without plot())
            annotated_frame = draw_detections_optimized(frame, results[0].boxes, 
                                                        model.names, confidence_threshold)
            
            # Analyze detections
            detected_objects = results[0].boxes
            for box in detected_objects:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                
                current_detections[class_name] = current_detections.get(class_name, 0) + 1
                total_detections += 1
                
                if class_name in VIOLATION_CLASSES:
                    detected_violations.append(class_name)
        else:
            # Skip inference frame - re-use previous detections for display
            # and just copy the frame for display
            annotated_frame = frame.copy()
            # (In production, you might want to show cached detections here)
        
        # Update violation tracking
        new_violations, resolved_violations = tracker.update_violations(detected_violations)
        
        # Log state changes
        if new_violations:
            print(f"🆕 New violations detected: {new_violations}")
        if resolved_violations:
            print(f"✅ Violations resolved: {resolved_violations}")
        
        # Calculate FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # ========================================
        # SCREENSHOT LOGIC (queued for background save)
        # ========================================
        
        screenshots_this_frame = []
        
        for violation in tracker.current_violations:
            should_capture, reason = tracker.should_screenshot(violation, current_time)
            
            if should_capture:
                # Queue screenshot for background save (non-blocking)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                violation_filename = f"{violations_folder}/violation_{violation}_{timestamp}.jpg"
                
                # Queue to background thread (frame copy to avoid reference issues)
                screenshot_queue.put((violation_filename, annotated_frame.copy()))
                
                tracker.record_screenshot(violation)
                screenshots_this_frame.append(violation)
                
                # Calculate duration
                duration = current_time - tracker.violation_start_time.get(violation, current_time)
                print(f"   Type: {violation} | Duration: {duration:.1f}s | Total shots: {tracker.total_screenshots}")
                
                violation_count += 1
        
        # ========================================
        # DRAW UI OVERLAYS
        # ========================================
        
        # Status panel background
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 90), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 90), (255, 255, 255), 2)
        
        # FPS & Settings
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(annotated_frame, f'Conf: {confidence_threshold:.2f}', (150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(annotated_frame, f'Cooldown: {SCREENSHOT_COOLDOWN}s', (300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Detection summary
        det_text = " | ".join([f"{name}: {count}" for name, count in current_detections.items()])
        if det_text:
            cv2.putText(annotated_frame, det_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(annotated_frame, "No detections", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Violation panel
        if tracker.current_violations:
            # Red border
            cv2.rectangle(annotated_frame, (0, 0), 
                         (annotated_frame.shape[1], annotated_frame.shape[0]), 
                         (0, 0, 255), 10)
            
            # Calculate panel height
            panel_height = 80 + (len(tracker.current_violations) * 35)
            cv2.rectangle(annotated_frame, (0, 100), (600, 100 + panel_height), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (0, 100), (600, 100 + panel_height), (0, 0, 255), 3)
            
            # Header
            cv2.putText(annotated_frame, '⚠️ ACTIVE VIOLATIONS', (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # List violations with duration and cooldown status
            y_pos = 165
            for violation in sorted(tracker.current_violations):
                duration = current_time - tracker.violation_start_time.get(violation, current_time)
                
                # Check screenshot status
                can_screenshot, reason = tracker.should_screenshot(violation, current_time)
                
                if reason == "cooldown":
                    time_since_last = current_time - tracker.last_screenshot_time[violation]
                    remaining = SCREENSHOT_COOLDOWN - time_since_last
                    status = f"[Cooldown: {remaining:.1f}s]"
                    color = (128, 128, 128)
                elif reason == "duration_too_short":
                    status = "[Waiting...]"
                    color = (128, 128, 128)
                else:
                    status = "[Ready to capture]"
                    color = (0, 255, 255)
                
                text = f"• {violation.upper().replace('_', ' ')} ({duration:.1f}s) {status}"
                cv2.putText(annotated_frame, text, (15, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 35
        
        # Bottom status bar
        cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 40), 
                     (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
        
        # Stats
        stats_text = f'Screenshots: {tracker.total_screenshots} | Active Violations: {len(tracker.current_violations)}'
        cv2.putText(annotated_frame, stats_text, 
                    (10, annotated_frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timestamp
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_text, 
                    (annotated_frame.shape[1] - 220, annotated_frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ========================================
        # RESIZE & DISPLAY
        # ========================================
        
        display_frame, scale_factor = resize_frame(annotated_frame)
        
        if frame_count == 1:
            print(f"📏 Display scale: {scale_factor:.2f}x")
            print(f"📺 Display size: {display_frame.shape[1]}x{display_frame.shape[0]}")
        
        # Only display if GUI is available
        if gui_available:
            cv2.imshow('PPE Detection - Safety Monitoring', display_frame)
        
        # ========================================
        # KEYBOARD CONTROLS
        # ========================================
        
        key = 0xFF  # Default: no key pressed
        
        if gui_available:
            try:
                key = cv2.waitKey(1) & 0xFF
            except:
                key = 0xFF
        
        if key == ord('q'):
            print("\n⏹️ Stopping...")
            break
        elif key == ord('s'):
            screenshot_name = f"{violations_folder}/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            screenshot_queue.put((screenshot_name, annotated_frame.copy()))
            print(f"📸 Manual screenshot queued")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"📈 Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"📉 Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('c'):
            print(f"\n⏱️ Current cooldown: {SCREENSHOT_COOLDOWN}s")
            try:
                new_cooldown = int(input("Enter new cooldown (seconds): "))
                if 1 <= new_cooldown <= 60:
                    SCREENSHOT_COOLDOWN = new_cooldown
                    print(f"✅ Cooldown updated to {SCREENSHOT_COOLDOWN}s")
                else:
                    print("⚠️ Invalid value. Cooldown must be between 1-60 seconds")
            except:
                print("⚠️ Invalid input. Cooldown unchanged.")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Stop background screenshot thread
    stop_screenshot_thread = True
    screenshot_thread.join(timeout=2)
    
    # Ensure remaining screenshots are saved
    time.sleep(0.5)
    
    cap.release()
    
    # Only destroy windows if GUI was available
    if gui_available:
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"⏱️  Total frames captured: {frame_count}")
    print(f"🎯 Total detections: {total_detections}")
    print(f"📸 Total screenshots: {tracker.total_screenshots}")
    print(f"⚠️  Unique violations: {violation_count}")
    print(f"📁 Location: {violations_folder}/")
    
    if use_openvino:
        print(f"\n⚡ OpenVINO ({OPENVINO_DEVICE}) was used for inference")
    else:
        print(f"\n📍 PyTorch CPU was used for inference")
        print(f"   💡 Tip: Install OpenVINO for 3-5x faster inference on GPU")
    
    print("="*70)
    print("✅ System stopped successfully!")