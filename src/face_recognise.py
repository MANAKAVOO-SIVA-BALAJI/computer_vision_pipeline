import numpy as np
import cv2
import os
from datetime import datetime

def recognize_faces_in_frame(frame, model, face_analyzer, faiss_index, database,
                             aoi_polygon=None, iou_threshold=0.3, similarity_threshold=0.3, k=1) -> list:
    """
    Detects and recognizes faces in a single video frame.

    Returns:
        List of dicts: [{track_id, label, confidence, bbox}, ...]
    """
    # print("Recognizing faces...")
    

    recognized_data = []

    results = model.track(source=frame, stream=False, persist=False, conf=0.5, tracker="bytetrack.yaml", verbose=False)
    
    if not results:
        return recognized_data

    faces = face_analyzer.get(frame)

    for result in results:
        if result.boxes is None:
            continue

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
            track_id = int(result.boxes.id[i]) if result.boxes.id is not None else -1

            if aoi_polygon is not None:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if cv2.pointPolygonTest(aoi_polygon.astype(np.int32), (float(cx), float(cy)), False) < 0:
                    continue

            # Match detection to the best face in frame
            best_face = None
            max_iou = 0
            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                inter_x1 = max(x1, fx1)
                inter_y1 = max(y1, fy1)
                inter_x2 = min(x2, fx2)
                inter_y2 = min(y2, fy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                union_area = (x2 - x1) * (y2 - y1) + (fx2 - fx1) * (fy2 - fy1) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou > max_iou and iou > iou_threshold:
                    best_face = face
                    max_iou = iou

            if best_face is None or best_face.embedding is None:
                continue

            emb = best_face.embedding.astype(np.float32).reshape(1, -1)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)

            D, I = faiss_index.search(emb, k)
            similarity = D[0][0]

            if similarity > similarity_threshold:
                label = database["labels"][I[0][0]]
            else:
                label = "Unknown"

            recognized_data.append({
                "track_id": track_id,
                "label": label,
                "confidence": round(float(similarity), 2),
                "bbox": [x1, y1, x2, y2]
            })
    # print("Recognition Result:", recognized_data)
    

    return recognized_data
