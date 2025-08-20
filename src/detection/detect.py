import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Create output directories
# os.makedirs("faces", exist_ok=True)

# Load YOLOv8 model (for person detection)
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

# Set input video and output path
input_video_path = "/content/office_video.mp4"
output_video_path = "output_with_boxes.mp4"

# Tracker settings
tracker_config = "bytetrack.yaml"

# Initialize video capture for frame size
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Set up video writer to save final output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Track people in the video using YOLOv8
results = model.track(
    source=input_video_path,
    stream=True,
    persist=True,
    save=False,
    tracker=tracker_config,
    classes=[0],  # Only detect 'person'
)

frame_count = 0
last_detections = {}
max_frames_missing = 60  # Retain box for 2 seconds at 30 FPS

for result in results:
    frame = result.orig_img.copy()
    frame_count += 1

    # Update detections if available in this frame
    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            last_detections[track_id] = {
                "box": box,
                "conf": conf,
                "frame_seen": frame_count
            }

            # Face extraction every 30th frame
            if frame_count % 30 == 0:
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y1 + int((y2 - y1) * 0.4), x1:x2]  # Top 40%

                # if face_img.size != 0:
                #     person_folder = f"faces/Person_{track_id}"
                #     os.makedirs(person_folder, exist_ok=True)
                #     face_path = os.path.join(person_folder, f"frame_{frame_count}.jpg")
                #     cv2.imwrite(face_path, face_img)

    # Draw all (current + recent) boxes
    for track_id in list(last_detections.keys()):
        det = last_detections[track_id]
        if frame_count - det["frame_seen"] > max_frames_missing:
            del last_detections[track_id]
            continue

        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["conf"]
        color = (0, 0, 255) if conf < 0.5 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the processed frame to output video
    out_video.write(frame)

# Cleanup
out_video.release()
# print(" Face extraction complete. Check the 'faces/' folder.")
print(" Output video saved as:", output_video_path)
