import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # replace with your model path
person_class_id = 0          # COCO person class id

polygon_points = []
polygon_file = "polygon.json"
window_name = "Draw ROI - 'c' clear, 's' save, ESC done"

# Global scale factors from display frame to original frame
scale_x, scale_y = 1.0, 1.0

def mouse_callback(event, x, y, flags, param):
    global polygon_points, scale_x, scale_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert click coords from resized display frame to original frame
        orig_x = int(x / scale_x)
        orig_y = int(y / scale_y)
        polygon_points.append((orig_x, orig_y))
        print(f"Point added (original coords): {(orig_x, orig_y)}")

def draw_polygon(frame):
    temp = frame.copy()
    if len(polygon_points) > 1:
        # To draw polygon, scale original points to display coords
        scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon_points]
        cv2.polylines(temp, [np.array(scaled_points, np.int32)], isClosed=True, color=(0,255,0), thickness=2)
    for pt in polygon_points:
        scaled_pt = (int(pt[0] * scale_x), int(pt[1] * scale_y))
        cv2.circle(temp, scaled_pt, 5, (0,0,255), -1)
    return temp

def save_polygon(file_path):
    with open(file_path, 'w') as f:
        json.dump(polygon_points, f)
    print(f"Polygon saved to {file_path}")

def load_polygon(file_path):
    global polygon_points
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            polygon_points = json.load(f)
        print(f"Polygon loaded from {file_path}")
    else:
        print(f"No polygon file found at {file_path}")

def create_mask(shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def polygon_draw_loop(frame):
    global polygon_points, scale_x, scale_y
    load_polygon(polygon_file)

    h, w = frame.shape[:2]
    DISPLAY_WIDTH = 800
    scale_x = DISPLAY_WIDTH / w
    scale_y = scale_x
    DISPLAY_HEIGHT = int(h * scale_y)

    resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_frame = draw_polygon(resized_frame)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to finish
            if len(polygon_points) >= 3:
                break
            else:
                print("Need at least 3 points to finish")
        elif key == ord('c'):  # Clear polygon points
            polygon_points = []
            print("Polygon cleared")
        elif key == ord('s'):  # Save polygon points
            if len(polygon_points) >= 3:
                save_polygon(polygon_file)
            else:
                print("Need at least 3 points to save polygon")

    cv2.destroyWindow(window_name)
    return polygon_points

def main(video_source=0):
    global polygon_points

    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from source")
        return

    # Draw polygon ROI and load/save support on resized display frame
    polygon_points = polygon_draw_loop(frame)

    if len(polygon_points) < 3:
        print("No valid polygon points, exiting...")
        return

    mask = create_mask(frame.shape, polygon_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        results = model(masked_frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == person_class_id:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if cv2.pointPolygonTest(np.array(polygon_points, dtype=np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw polygon on displayed frame (scale points)
        scaled_poly = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon_points]
        display_frame = cv2.resize(frame, (int(frame.shape[1]*scale_x), int(frame.shape[0]*scale_y)))
        cv2.polylines(display_frame, [np.array(scaled_poly, np.int32)], isClosed=True, color=(0,255,0), thickness=2)

        cv2.imshow("YOLOv8 Human Detection in ROI", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main")
