import cv2
import os
import json
import numpy as np
import time
import pickle
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import faiss

# =================== CONFIG ====================
VIDEO_PATH = "test_video.mp4"
YOLO_MODEL_PATH = "yolov12l-face.pt"
FAISS_INDEX_PATH = "faiss.index"
CACHE_PATH = "face_embeddings_faiss.pkl"
OUTPUT_VIDEO_PATH = "output_tracked_video.mp4"
LOG_FILE = "attendance_log.jsonl"
CAMERA_ID = "entrance_camera"

SIMILARITY_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
K = 1

# Logging & tracking control
LOG_INTERVAL = 10             # seconds between logs for same person

# AOI CONFIG
USE_POLYGON = True
AOI_RECT = (300, 300, 900, 900)
AOI_POLYGON = np.array([
    [8, 424],
    [642, 372],
    [744, 1566],
    [0, 1564],
    [10, 432]
], dtype=np.int32)

# =================== Helper Functions ====================
def is_inside_rectangle(box, rect):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def is_inside_polygon(box, polygon):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cv2.pointPolygonTest(polygon.astype(np.int32), (float(cx), float(cy)), False) >= 0

def log_event(event):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

# =================== Load Models ====================
print("🔍 Loading models...")
model = YOLO(YOLO_MODEL_PATH)
face_analyzer = FaceAnalysis(name='buffalo_l', root='~/.insightface')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

with open(CACHE_PATH, "rb") as f:
    database = pickle.load(f)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# =================== Video Setup ====================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Tracking memory
last_log_time = {}  # track_id -> last log timestamp

# =================== Processing Loop ====================
results = model.track(
    source=VIDEO_PATH,
    stream=True,
    persist=True,
    save=False,
    tracker="bytetrack.yaml",
    conf=0.5
)

for result in results:
    frame = result.orig_img
    current_time = time.time()

    # Draw AOI
    if USE_POLYGON:
        cv2.polylines(frame, [AOI_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)
    else:
        cv2.rectangle(frame, (AOI_RECT[0], AOI_RECT[1]), (AOI_RECT[2], AOI_RECT[3]), (255, 255, 0), 2)

    online_targets = result.boxes
    if online_targets is None:
        out_writer.write(frame)
        cv2.imshow("Live Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    faces = face_analyzer.get(frame)

    for i in range(len(online_targets)):
        x1, y1, x2, y2 = map(int, online_targets.xyxy[i].tolist())
        track_id = int(online_targets.id[i]) if online_targets.id is not None else -1

        # Check AOI
        if USE_POLYGON:
            if not is_inside_polygon((x1, y1, x2, y2), AOI_POLYGON):
                continue
        else:
            if not is_inside_rectangle((x1, y1, x2, y2), AOI_RECT):
                continue

        # Match detection to best face
        best_face = None
        max_iou = 0
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            inter_x1 = max(x1, fx1)
            inter_y1 = max(y1, fy1)
            inter_x2 = min(x2, fx2)
            inter_y2 = min(y2, fy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = (x2 - x1)*(y2 - y1) + (fx2 - fx1)*(fy2 - fy1) - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if iou > max_iou and iou > IOU_THRESHOLD:
                best_face = face
                max_iou = iou

        if best_face is None or best_face.embedding is None:
            continue

        # Normalize embedding
        emb = best_face.embedding.astype(np.float32).reshape(1, -1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        D, I = faiss_index.search(emb, K)
        similarity = D[0][0]

        if similarity > SIMILARITY_THRESHOLD:
            label = database["labels"][I[0][0]]
        else:
            label = "Unknown"

        # Draw on frame
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {label} ({similarity:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Attendance logging
        if label != "Unknown":
            if (track_id not in last_log_time) or (current_time - last_log_time[track_id] > LOG_INTERVAL):
                event = {
                    "camera_id": CAMERA_ID,
                    "timestamp": datetime.now().isoformat(),
                    "detections": [{
                        "track_id": track_id,
                        "label": label,
                        "confidence": round(float(similarity), 2),
                        "bbox": [x1, y1, x2, y2]
                    }]
                }
                log_event(event)
                last_log_time[track_id] = current_time

    # Save & show live
    out_writer.write(frame)
    cv2.imshow("Live Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()

print(f"🎥 Output video saved to {OUTPUT_VIDEO_PATH}")
print(f"📝 Attendance log saved to {LOG_FILE}")
