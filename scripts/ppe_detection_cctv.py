from ultralytics import YOLO
import cv2
import time
import os
from dotenv import load_dotenv
from datetime import datetime

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

# All classes (tidak difilter)
ALL_CLASSES = list(model.names.values())
print(f"\n🎯 Monitoring all classes: {ALL_CLASSES}")

# Violation classes
VIOLATION_CLASSES = ['no_helmet', 'no_glove', 'no_goggles', 'no_mask', 'no_shoes']

# ========================================
# DISPLAY SETTINGS
# ========================================

# Max display size (fit to laptop screen)
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

def resize_frame(frame, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """Resize frame to fit screen while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
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
        # Build RTSP URL from env variables
        cctv_url = os.getenv('CCTV_URL')
        if cctv_url:
            print(f"\n📹 Using CCTV from .env")
            print(f"🔗 URL: {cctv_url[:20]}... (hidden for security)")
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

# Get camera source
camera_source = get_camera_source()

if camera_source is None:
    print("❌ Failed to get camera source. Check your .env file!")
    exit()

# ========================================
# OPEN VIDEO STREAM
# ========================================

print(f"\n🔌 Connecting to camera...")
cap = cv2.VideoCapture(camera_source)

# CCTV optimization settings
if isinstance(camera_source, str) and camera_source.startswith('rtsp'):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    cap.set(cv2.CAP_PROP_FPS, 15)  # Request 15 FPS

# Check if opened successfully
if not cap.isOpened():
    print("❌ Error: Cannot open camera/CCTV stream!")
    print("💡 Tips:")
    print("   - Check network connection (for CCTV)")
    print("   - Verify RTSP URL is correct")
    print("   - Check username/password")
    print("   - Try ping the CCTV IP")
    exit()

print("✅ Camera connected!")

# Get camera info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cam = int(cap.get(cv2.CAP_PROP_FPS))
print(f"📐 Original Resolution: {width}x{height}")
print(f"📺 Display Max Size: {MAX_DISPLAY_WIDTH}x{MAX_DISPLAY_HEIGHT}")
print(f"🎬 Camera FPS: {fps_cam}")

print("\n💡 Controls:")
print("   - Press 'q' to quit")
print("   - Press 's' to save screenshot")
print("   - Press '+' to increase confidence threshold")
print("   - Press '-' to decrease confidence threshold")
print("="*70 + "\n")

# ========================================
# CREATE FOLDERS
# ========================================

violations_folder = "../PPE-DETECTION/violations"
os.makedirs(violations_folder, exist_ok=True)

# ========================================
# DETECTION LOOP
# ========================================

prev_time = time.time()
violation_count = 0
frame_count = 0
total_detections = 0
confidence_threshold = 0.5

# Create resizable window
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
        
        # Run detection (TANPA FILTER - detect semua class)
        results = model.predict(
            frame, 
            conf=confidence_threshold,
            verbose=False
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Analyze detections
        detected_objects = results[0].boxes
        has_violation = False
        violation_details = []
        current_detections = {}
        
        for box in detected_objects:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            # Count detections
            current_detections[class_name] = current_detections.get(class_name, 0) + 1
            total_detections += 1
            
            # Check violations (semua class yang ada "no_")
            if class_name in VIOLATION_CLASSES:
                has_violation = True
                violation_details.append(f"{class_name.upper().replace('_', ' ')} ({confidence:.0%})")
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # ========================================
        # DRAW UI OVERLAYS
        # ========================================
        
        # Status panel background (top)
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 90), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 90), (255, 255, 255), 2)
        
        # FPS
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Confidence threshold
        cv2.putText(annotated_frame, f'Conf: {confidence_threshold:.2f}', (150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Detection count
        det_text = " | ".join([f"{name}: {count}" for name, count in current_detections.items()])
        if det_text:
            cv2.putText(annotated_frame, det_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(annotated_frame, "No detections", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Violation warning
        if has_violation:
            violation_count += 1
            
            # Red border
            border_thickness = 10
            cv2.rectangle(annotated_frame, (0, 0), 
                         (annotated_frame.shape[1], annotated_frame.shape[0]), 
                         (0, 0, 255), border_thickness)
            
            # Warning panel background
            warning_height = 50 + (len(violation_details) * 40)
            cv2.rectangle(annotated_frame, (0, 100), (500, 100 + warning_height), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (0, 100), (500, 100 + warning_height), (0, 0, 255), 3)
            
            # Warning header
            cv2.putText(annotated_frame, '⚠️ SAFETY VIOLATION DETECTED!', (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Violation details
            y_pos = 165
            for vtext in violation_details:
                cv2.putText(annotated_frame, f'• {vtext}', (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_pos += 35
            
            # Auto-save violation screenshot (original size)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            violation_filename = f"{violations_folder}/violation_{timestamp}.jpg"
            cv2.imwrite(violation_filename, annotated_frame)
            print(f"📸 Violation saved: {violation_filename}")
        
        # Bottom status bar background
        cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 40), 
                     (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
        
        # Violation counter
        cv2.putText(annotated_frame, f'Total Violations: {violation_count}', 
                    (10, annotated_frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if violation_count > 0 else (255, 255, 255), 2)
        
        # Timestamp
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_text, 
                    (annotated_frame.shape[1] - 220, annotated_frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ========================================
        # RESIZE FRAME TO FIT SCREEN
        # ========================================
        
        display_frame, scale_factor = resize_frame(annotated_frame)
        
        # Show original size info (only once)
        if frame_count == 1:
            print(f"📏 Display scale: {scale_factor:.2f}x")
            print(f"📺 Display size: {display_frame.shape[1]}x{display_frame.shape[0]}")
        
        # ========================================
        # DISPLAY FRAME
        # ========================================
        
        cv2.imshow('PPE Detection - Safety Monitoring', display_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️ Stopping...")
            break
        elif key == ord('s'):
            # Manual screenshot (save original size)
            screenshot_name = f"{violations_folder}/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, annotated_frame)
            print(f"📸 Screenshot saved: {screenshot_name}")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"📈 Confidence threshold increased: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"📉 Confidence threshold decreased: {confidence_threshold:.2f}")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"⏱️  Total runtime: {frame_count / fps if fps > 0 else 0:.1f} seconds")
    print(f"🎬 Total frames processed: {frame_count}")
    print(f"🎯 Total detections: {total_detections}")
    print(f"⚠️  Total violations: {violation_count}")
    print(f"📁 Violations saved in: {violations_folder}/")
    print("="*70)
    print("✅ System stopped successfully!")