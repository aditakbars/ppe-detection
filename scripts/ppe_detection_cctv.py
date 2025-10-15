from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict

# Load environment variables
load_dotenv()

print("="*70)
print("🏭 PPE DETECTION - Multi-Source Real-Time Monitoring")
print("="*70)

# ========================================
# CONFIGURATION
# ========================================

# Load trained model
model_path = "../models/ppe_model_v1/ppe_best_model_3epoch.pt"
print(f"\n📥 Loading model: {model_path}")
model = YOLO(model_path)
print("✅ Model loaded!")

# All classes
ALL_CLASSES = list(model.names.values())
print(f"\n🎯 Monitoring all classes: {ALL_CLASSES}")

# Violation classes
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']

# ========================================
# SCREENSHOT SETTINGS
# ========================================

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

violations_folder = "../PPE-DETECTION/violations"
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

prev_time = time.time()
violation_count = 0
frame_count = 0
total_detections = 0
confidence_threshold = 0.5

cv2.namedWindow('PPE Detection - Safety Monitoring', cv2.WINDOW_NORMAL)

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
        
        # Run detection
        results = model.predict(
            frame, 
            conf=confidence_threshold,
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        
        # Analyze detections
        detected_objects = results[0].boxes
        current_detections = {}
        detected_violations = []
        
        for box in detected_objects:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            current_detections[class_name] = current_detections.get(class_name, 0) + 1
            total_detections += 1
            
            if class_name in VIOLATION_CLASSES:
                detected_violations.append(class_name)
        
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
        # SCREENSHOT LOGIC
        # ========================================
        
        screenshots_this_frame = []
        
        for violation in tracker.current_violations:
            should_capture, reason = tracker.should_screenshot(violation, current_time)
            
            if should_capture:
                # Take screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                violation_filename = f"{violations_folder}/violation_{violation}_{timestamp}.jpg"
                cv2.imwrite(violation_filename, annotated_frame)
                
                tracker.record_screenshot(violation)
                screenshots_this_frame.append(violation)
                
                # Calculate duration
                duration = current_time - tracker.violation_start_time.get(violation, current_time)
                
                print(f"📸 Screenshot saved: {violation_filename}")
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
        
        cv2.imshow('PPE Detection - Safety Monitoring', display_frame)
        
        # ========================================
        # KEYBOARD CONTROLS
        # ========================================
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️ Stopping...")
            break
        elif key == ord('s'):
            screenshot_name = f"{violations_folder}/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, annotated_frame)
            print(f"📸 Manual screenshot saved: {screenshot_name}")
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
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"⏱️  Total runtime: {frame_count / fps if fps > 0 else 0:.1f} seconds")
    print(f"🎬 Total frames: {frame_count}")
    print(f"🎯 Total detections: {total_detections}")
    print(f"📸 Total screenshots: {tracker.total_screenshots}")
    print(f"⚠️  Unique violations: {violation_count}")
    print(f"📁 Location: {violations_folder}/")
    print("="*70)
    print("✅ System stopped successfully!")