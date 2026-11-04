import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. DOWNLOAD SAMPLE VIDEO IF NEEDED
# ==============================================================================
VIDEO_PATH = "speed_tracker_vid.mp4"

if not os.path.exists(VIDEO_PATH):
    print("Downloading sample traffic video...")
    url = "https://github.com/Alexxabmp/speed_tracker/releases/download/v1/speed_tracker_vid.mp4"
    urllib.request.urlretrieve(url, VIDEO_PATH)
    print("Download complete!\n")

# ==============================================================================
# 2. INTERACTIVE 4-POINT ROAD SELECTION (MANUAL STEP)
# ==============================================================================
clicked_points = []

def select_points(event, x, y, flags, param):
    """Callback function to record 4 mouse clicks on the video frame."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)} selected: ({x}, {y})")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video frame.")

window_title = "Click 4 Road Corners: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, select_points)

print("--- INSTRUCTIONS ---")
print("1. Click 4 corners of a road rectangle in order:")
print("   P1: Top-Left, P2: Top-Right, P3: Bottom-Right, P4: Bottom-Left")
print("2. After 4 dots appear, press ANY KEY to launch tracking.\n")

while True:
    display_frame = first_frame.copy()
    
    # Draw clicked points and labels
    for i, pt in enumerate(clicked_points):
        cv2.circle(display_frame, pt, 6, (0, 0, 255), -1)
        cv2.putText(display_frame, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Connect the polygon when 4 points are clicked
    if len(clicked_points) == 4:
        pts = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(display_frame, "Press ANY KEY to launch tracker", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(window_title, display_frame)
    key = cv2.waitKey(1) & 0xFF
    if len(clicked_points) == 4 and key != 255:
        break

cv2.destroyWindow(window_title)

# ==============================================================================
# 3. HOMOGRAPHY MATRIX COMPUTATION (2D -> 2D)
# ==============================================================================
src_pixels = np.float32(clicked_points)

# Known real-world physical road dimensions in meters (Width x Length)
# Example: 3.6 meters wide x 25.0 meters long
dst_meters = np.float32([
    [0.0,  0.0],   # Top-Left
    [3.6,  0.0],   # Top-Right
    [3.6, 25.0],   # Bottom-Right
    [0.0, 25.0]    # Bottom-Left
])

M = cv2.getPerspectiveTransform(src_pixels, dst_meters)

# ==============================================================================
# 4. TRACKING LOOP WITH DYNAMIC MOVING SPEED LABELS
# ==============================================================================
model = YOLO("yolov8n.pt")
track_history = {}
smooth_speeds = {}

# Reset video frame index to start from the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Detect and track vehicles (class 2=car, 3=motorcycle, 5=bus, 7=truck)
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Rear-tire anchor point (Bottom-center of bounding box)
            tire_x, tire_y = (x1 + x2) / 2.0, y2

            # Transform pixel location [x, y] to ground meters [X, Y]
            pixel_pt = np.array([[[tire_x, tire_y]]], dtype=np.float32)
            meter_pt = cv2.perspectiveTransform(pixel_pt, M)[0][0]

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((meter_pt[0], meter_pt[1], frame_idx))

            # Calculate speed over a 6-frame window
            current_speed_kmh = 0.0
            history = track_history[track_id]

            if len(history) >= 6:
                x_prev, y_prev, f_prev = history[-6]
                x_curr, y_curr, f_curr = history[-1]

                distance_m = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
                time_sec = (f_curr - f_prev) / fps

                if time_sec > 0:
                    raw_speed_kmh = (distance_m / time_sec) * 3.6
                    
                    # Exponential Moving Average (EMA) smoothing
                    if track_id in smooth_speeds:
                        current_speed_kmh = 0.75 * smooth_speeds[track_id] + 0.25 * raw_speed_kmh
                    else:
                        current_speed_kmh = raw_speed_kmh
                    smooth_speeds[track_id] = current_speed_kmh

            # --- DRAW MOVING OVERLAY ---
            # Bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Tire contact point dot
            cv2.circle(frame, (int(tire_x), int(tire_y)), 5, (0, 0, 255), -1)
            # Label anchored above the moving car box
            label = f"Car #{track_id}: {current_speed_kmh:.1f} km/h"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Speed Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
