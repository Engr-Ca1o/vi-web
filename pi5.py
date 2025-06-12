import cv2
import numpy
import time
import json
import threading
import socket
import os
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import Transform
from paddleocr import PaddleOCR
import logging
import datetime

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Global variables
red_light_model = YOLO('best_traffic_nano_yolo.pt')  # Red light detection model
vehicle_model = YOLO('yolov8n.pt')  # Vehicle type detection model
license_plate_model = YOLO('lpd.pt')  # License plate detection model
red_light_cam = None
license_plate_cam = None
red_detected = False
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR
red_light_active = False  # Track red light camera activation
current_mode = None

# Ethernet configuration
SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096
SERVER_IP = "192.168.2.1"  # Pi 4B's static IP
PORT = 5001

def initialize_cameras():
    """Initialize and configure the cameras with a single stream."""
    global red_light_cam, license_plate_cam
    red_light_cam = Picamera2(0)
    license_plate_cam = Picamera2(1)
    
    # Configure red light camera (low resolution for detection)
    config0 = red_light_cam.create_preview_configuration(main={"size": (640, 480)})
    red_light_cam.configure(config0)
    
    # Configure license plate camera (high resolution for capturing and detection)
    config1 = license_plate_cam.create_still_configuration(
        main={"size": (1920, 1080)},  # High resolution for capturing and detection
        transform=Transform(hflip=1, vflip=1),
    )
    license_plate_cam.configure(config1)
    
    # Start cameras
    red_light_cam.start()
    license_plate_cam.start()
    time.sleep(2)  # Allow cameras to stabilize

    # Set autofocus controls after a delay
    license_plate_cam.set_controls({
        "AfMode": 2,
        "AfPause": 2,
        "AfSpeed": 1})


def apply_night_settings():
    """Apply night settings to the license plate camera."""
    global license_plate_cam
    print("Applying night settings to the license plate camera...")
    license_plate_cam.set_controls({
        "AeEnable": False,
        "ExposureTime": 50_000,
        "AnalogueGain": 10.0,
        "AwbEnable": False,
        "FrameDurationLimits": (20_000, 20_000),
        "NoiseReductionMode": 2,
        "ColourGains": (1.0, 1.2),
        "Brightness": 0.3,
        "Contrast": 1.5,
        "Saturation": 1.0,
        "AfMode": 0,
        "LensPosition":6.5,
        "AfSpeed": 1
    })

def apply_day_settings():
    """Apply day settings to the license plate camera."""
    global license_plate_cam
    print("Applying day settings to the license plate camera...")
    license_plate_cam.set_controls({
        "AeEnable": True,  # Enable auto-exposure
        "ExposureTime": 1_000,  # Drop exposure for daytime
        "AnalogueGain": 1.0,  # Default gain
        "AwbEnable": True, # Enable auto white balance
        "AfMode": 2, 
        "AfPause": 2,
        "AfSpeed": 1, 
    })

def check_and_apply_time_based_settings():
    """Check the current time and apply appropriate camera settings only when the mode changes."""
    global current_mode
    current_time = datetime.datetime.now().time()
    night_start = datetime.time(18, 15)  # 6:30 PM
    day_start = datetime.time(6, 0)  # 6:00 AM

    # Determine the current mode based on the time
    if night_start <= current_time or current_time < day_start:
        new_mode = "night"
    else:
        new_mode = "day"

    # Apply settings only if the mode has changed
    if new_mode != current_mode:
        if new_mode == "night":
            apply_night_settings()
        else:
            apply_day_settings()
        current_mode = new_mode  # Update the current mode

def detect_red_light(frame):
    """Detect if the red light is active."""
    results = red_light_model(frame)
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Update this ID based on your YOLO model class mapping for red light
                return True
    return False

def detect_violating_vehicles(frame, line_start=(660, 560), line_end=(1500, 700)):
    """Detect vehicles violating the red light and classify their types."""
    global license_plate_cam  # Ensure access to the high-quality camerafrr
    vehicle_results = vehicle_model(frame)  # Use vehicle detection model
    detected_vehicles = []

    # Draw the diagonal line for visualization
    cv2.line(frame, line_start, line_end, (255, 0, 0), 2)

    # Define the class map before using it
    class_map = {2: "car", 3: "motorcycle", 5: "truck", 7: "bus"}

    for r in vehicle_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Class ID of the detected object
            confidence = float(box.conf[0])  # Confidence score

            # Filter detections by confidence threshold (e.g., 0.5)
            if confidence > 0.4 and class_id in class_map:
                # Map class IDs to vehicle types
                vehicle_type = class_map[class_id]  # Directly access the mapped vehicle type

                # Append detected vehicle information
                detected_vehicles.append((x1, y1, x2, y2, vehicle_type))

                # Check if the bounding box intersects with the diagonal line
                if line_intersects_box(line_start, line_end, (x1, y1, x2, y2)):
                    print(f"Vehicle {vehicle_type} intersects with the detection line. Capturing license plate...")
                    threading.Thread(target=process_vehicle, args=((x1, y1, x2, y2), vehicle_type)).start()
                time.sleep(0.5)

    return detected_vehicles, frame


def line_intersects_box(line_start, line_end, box):
    """Check if a line intersects with a bounding box."""
    x1, y1, x2, y2 = box
    box_lines = [
        ((x1, y1), (x2, y1)),  # Top edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Bottom edge
        ((x1, y2), (x1, y1)),  # Left edge
    ]
    for box_line_start, box_line_end in box_lines:
        if lines_intersect(line_start, line_end, box_line_start, box_line_end):
            return True
    return False


def lines_intersect(p1, p2, q1, q2):
    """Check if two lines intersect."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def extract_license_plate_text(image_path):
    """Extract text from the license plate using PaddleOCR with preprocessing."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load the image at {image_path}")
        return ""

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_path = image_path.replace(".jpg", "_gray.jpg")
    cv2.imwrite(gray_path, gray)  # Save for debugging
    print(f"Saved grayscale image: {gray_path}")

    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_path = image_path.replace(".jpg", "_blurred.jpg")
    cv2.imwrite(blurred_path, blurred)  # Save for debugging
    print(f"Saved blurred image: {blurred_path}")

    # Perform OCR on the processed image
    result = ocr.ocr(blurred, cls=True)

    # Extract and return the text from the license plate
    license_plate_text = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            license_plate_text += line[1][0] + " "
    return license_plate_text.strip()

def send_files_to_ethernet(files_to_send):
    """Send files to a server over Ethernet."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((SERVER_IP, PORT))
            for file_path in files_to_send:
                if os.path.exists(file_path):
                    filesize = os.path.getsize(file_path)
                    # Send file metadata: filename and filesize
                    metadata = f"{os.path.basename(file_path)}{SEPARATOR}{filesize}"
                    s.send(metadata.encode())
                    time.sleep(0.5)  # small delay to let the server prepare (optional)

                    # Open the file in binary mode and send its content
                    with open(file_path, "rb") as f:
                        while True:
                            bytes_read = f.read(BUFFER_SIZE)
                            if not bytes_read:
                                break
                            s.sendall(bytes_read)
                    print(f"Sent {file_path}")
                    time.sleep(0.5)  # small delay before next file
                else:
                    print(f"File not found: {file_path}")
            s.shutdown(socket.SHUT_WR)
        except Exception as e:
            print(f"Error sending files: {e}")

def save_data_to_json(license_plate, vehicle_type, timestamp, image_filename):
    """Save violator data to a JSON file and send it over Ethernet."""
    data = {
        "license_plate": license_plate,
        "vehicle_type": vehicle_type,
        "time_date": timestamp,
        "image_filename": image_filename
    }
    json_filename = f"violator_{timestamp}.json"
    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Saved violator data to JSON: {json_filename}")

    # Send the JSON file and the license plate image over Ethernet
    send_files_to_ethernet([image_filename, json_filename])

def process_vehicle(vehicle_box, vehicle_type):
    """Process a detected vehicle: capture license plate and save data."""
    global license_plate_cam, license_plate_model
    x1, y1, x2, y2 = vehicle_box

    # Add a margin around the bounding box
    margin = 50  # Adjust this value as needed
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(license_plate_cam.capture_array("main").shape[1], x2 + margin)  # Ensure it doesn't exceed frame width
    y2 = min(license_plate_cam.capture_array("main").shape[0], y2 + margin)  # Ensure it doesn't exceed frame height

    # Capture high-resolution frame from the still stream
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    frame = license_plate_cam.capture_array("main")  # Use the high-resolution stream
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    fullframe_path = f"full_frame_violator_{timestamp}.jpg"
    cv2.imwrite(fullframe_path, frame)

    # Crop the vehicle bounding box
    vehicle_crop = frame[y1:y2, x1:x2]
    vehicle_crop_path = f"vehicle_crop_{timestamp}.jpg"
    cv2.imwrite(vehicle_crop_path, vehicle_crop)
    print(f"Captured vehicle crop: {vehicle_crop_path}")

    # Detect license plate within the vehicle crop

            # Extract license plate text
    license_plate_text = extract_license_plate_text(vehicle_crop_path)
    print(f"Extracted license plate text: {license_plate_text}")

            # Save data to JSON
    save_data_to_json(license_plate_text, vehicle_type, timestamp, fullframe_path)
    return  # Process only the first detected license plate

def run():
    """Main function to monitor red light and process violators."""
    global red_detected, red_light_cam, license_plate_cam
    try:
        while True:
            check_and_apply_time_based_settings()
            # Capture frame from red light camera
            red_light_frame = red_light_cam.capture_array("main")  # Use the main stream
            red_light_frame = cv2.cvtColor(red_light_frame, cv2.COLOR_RGB2BGR)

            # Check if red light is detected
            red_detected_now = detect_red_light(red_light_frame)
            if red_detected_now:
                if not red_detected:
                    print("Red light detected! Starting vehicle capture...")
                red_detected = True

                # Capture frame from license plate camera
                vehicle_frame = license_plate_cam.capture_array("main")  # Use the main stream
                vehicle_frame = cv2.cvtColor(vehicle_frame, cv2.COLOR_RGB2BGR)
                vehicles, processed_frame = detect_violating_vehicles(vehicle_frame)

                if vehicles:
                    print(f"Detected {len(vehicles)} violating vehicles")
                    for vehicle in vehicles:
                        x1, y1, x2, y2, vehicle_type = vehicle
                        threading.Thread(target=process_vehicle, args=((x1, y1, x2, y2), vehicle_type)).start()
            else:
                if red_detected:
                    print("Red light no longer detected. Stopping vehicle capture.")
                red_detected = False

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Detection stopped by user.")
    finally:
        red_light_cam.stop()
        license_plate_cam.stop()

def preview_cameras():
    """Preview the video feed from both cameras with toggleable red light detection."""
    global red_light_cam, license_plate_cam, red_detected, red_light_active
    try:
        while True:

            check_and_apply_time_based_settings()
            # Capture frames from both cameras
            red_light_frame = red_light_cam.capture_array("main")  # Use the main stream
            license_plate_frame = license_plate_cam.capture_array("main")  # Use the main stream
            
            # Convert frames to BGR format for OpenCV
            red_light_frame = cv2.cvtColor(red_light_frame, cv2.COLOR_RGB2BGR)
            license_plate_frame = cv2.cvtColor(license_plate_frame, cv2.COLOR_RGB2BGR)
            
            # Toggle red light detection with 'r' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Press 'r' to toggle red light detection
                red_light_active = not red_light_active
                print(f"Red light detection {'activated' if red_light_active else 'deactivated'}.")

            if red_light_active:
                red_detected = True
                cv2.putText(red_light_frame, "Red Light Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(license_plate_frame, "Red Light Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                # Detect vehicles only when red light is detected
                vehicles, license_plate_frame_with_boxes = detect_violating_vehicles(license_plate_frame)
                
            else:
                red_detected = False
                license_plate_frame_with_boxes = license_plate_frame  # No bounding boxes if detection is off
            
            # Display the frames in separate windows
            cv2.imshow("Red Light Camera", red_light_frame)
            cv2.imshow("License Plate Camera", license_plate_frame_with_boxes)
            
            # Break the loop if 'q' is pressed
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Preview stopped by user.")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        red_light_cam.stop()
        license_plate_cam.stop()


if __name__ == "__main__":
    initialize_cameras()
    preview_cameras()






