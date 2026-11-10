import os
import urllib.request
import cv2
import numpy as np
import pandas as pd
from collections import deque
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

# --- FIX #2 (FPS CHECK): Verify and print detected FPS ---
detected_fps = cap.get(cv2.CAP_PROP_FPS)
fps = detected_fps if (detected_fps and detected_fps > 0) else 30.0
print(f"Detected OpenCV FPS: {detected_fps} | Using FPS: {fps:.1f}")

# Calculate optimal window size targeting ~0.2 seconds of video time
WINDOW_SIZE = max(2, int(round(fps * 0.2)))
print(f"Dynamic Window Size set to: {WINDOW_SIZE} frames (~0.2s)")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video frame.")

window_title = "Click 4 Corners of 4 Main Lanes: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, select_points)

print("\n--- INSTRUCTIONS FOR 4-LANE CALIBRATION ---")
print("1. Click 4 corners of the 4 camera-facing lanes only:")
print("   P1: Top-Left  (Median curb at horizon)")
print("   P2: Top-Right (Solid line before bus lane at horizon)")
print("   P3: Bottom-Right (Solid line before bus lane in foreground)")
print("   P4: Bottom-Left  (Median curb in foreground)")
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

# --- FIX #1 (LENGTH SCALE): Physical dimensions for 4 lanes (14.0m width x 60.0m length) ---
# Width:  4 lanes x 3.5m per lane = 14.0m
# Length: Extended ground calibration from 30.0m to 60.0m to correct slow speed readings
dst_meters = np.float32([
    [0.0,   0.0],   # P1: Top-Left  (Median curb at horizon)
    [14.0,  0.0],   # P2: Top-Right (Lane 4 right edge at horizon)
    [14.0, 60.0],   # P3: Bottom-Right (Lane 4 right edge in foreground)
    [0.0,  60.0]    # P4: Bottom-Left  (Median curb in foreground)
])

M = cv2.getPerspectiveTransform(src_pixels, dst_meters)

# ==============================================================================
# 4. TRACKING LOOP WITH FAST INITIALIZATION & EMA SMOOTHING
# ==============================================================================
# FIX #2 (PREVIOUS): Upgrade model to yolov8s.pt for higher accuracy on distant vehicles
model = YOLO("yolov8s.pt")

track_history = {}
smooth_speeds = {}
log_records = []

# Reset video frame index to start from the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # FIX #2 (PREVIOUS): Set conf=0.15 to prevent dropping distant bounding boxes & resetting IDs
    results = model.track(
        frame, 
        persist=True, 
        classes=[2, 3, 5, 7], 
        conf=0.15,          # Catches cars much earlier near the horizon
        iou=0.5,           # Helps prevent duplicate boxes
        verbose=False
    )

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
                track_history[track_id] = deque(maxlen=WINDOW_SIZE)
            track_history[track_id].append((meter_pt[0], meter_pt[1], frame_idx))

            current_speed_kmh = 0.0
            history = track_history[track_id]

            # Calculate speed immediately once >= 2 frame positions are recorded
            if len(history) >= 2:
                x_prev, y_prev, f_prev = history[0]
                x_curr, y_curr, f_curr = history[-1]

                distance_m = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
                time_sec = (f_curr - f_prev) / fps

                if time_sec > 0:
                    raw_speed_kmh = (distance_m / time_sec) * 3.6
                   
                    # FIX #1 (PREVIOUS): Instant EMA initialization — use raw speed immediately on first calculation
                    if track_id not in smooth_speeds:
                        current_speed_kmh = raw_speed_kmh  # Direct set avoids ramping up from 0 km/h
                    else:
                        current_speed_kmh = 0.75 * smooth_speeds[track_id] + 0.25 * raw_speed_kmh
                    
                    smooth_speeds[track_id] = current_speed_kmh

                    # Record tracking metrics
                    log_records.append({
                        "frame": frame_idx,
                        "track_id": track_id,
                        "raw_speed_kmh": round(raw_speed_kmh, 2),
                        "ema_speed_kmh": round(current_speed_kmh, 2)
                    })

            # --- DRAW MOVING OVERLAY ---
            # Bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Tire contact point dot
            cv2.circle(frame, (int(tire_x), int(tire_y)), 5, (0, 0, 255), -1)
            # Label anchored above the moving car box
            label = f"Car #{track_id}: {current_speed_kmh:.1f} km/h"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Speed Tracker (4-Lane / 60m Ground Scale)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export log data to CSV
if log_records:
    df_logs = pd.DataFrame(log_records)
    df_logs.to_csv("speed_tracking_log.csv", index=False)
    print("\nTracking logs exported successfully to 'speed_tracking_log.csv'.")

# ==============================================================================
# 5. STATISTICAL EVALUATION MODULE (MAE, MAPE, RMSE)
# ==============================================================================
def evaluate_speed_metrics(actual_speeds, predicted_speeds):
    """Calculates MAE, MAPE, and RMSE metrics."""
    y_true = np.array(actual_speeds, dtype=float)
    y_pred = np.array(predicted_speeds, dtype=float)
    
    abs_errors = np.abs(y_true - y_pred)
    mae = np.mean(abs_errors)
    mape = np.mean(abs_errors / y_true) * 100.0
    rmse = np.sqrt(np.mean(abs_errors ** 2))
    
    return mae, mape, rmse

# Demo evaluation summary
print("\n--- STATISTICAL EVALUATION SUMMARY ---")
actual_gt  = [50.0, 50.0, 50.0, 50.0]
pred_test  = [51.0, 50.0, 52.0, 70.0]

mae, mape, rmse = evaluate_speed_metrics(actual_gt, pred_test)
print(f"MAE  (Mean Absolute Error)            : {mae:.2f} km/h")
print(f"MAPE (Mean Absolute Percentage Error)  : {mape:.2f}%")
print(f"RMSE (Root Mean Square Error)          : {rmse:.2f} km/h")
