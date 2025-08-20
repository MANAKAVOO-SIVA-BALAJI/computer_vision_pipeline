
import logging
from src.motion_detector import MotionDetector
from src.face_recognition_pipeline import FaceRecognitionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Dummy motion detector (not used in test mode)
    motion_detector = MotionDetector("192.168.0.108", 8000, "admin", "Kadit1234")

    # video_source = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"
    video_source = "src/test_video.mp4"
    # video_source = 0

    pipeline = FaceRecognitionPipeline(
        camera_rtsp_url=video_source,
        motion_detector=motion_detector,
        frame_skip=10 
    )

    pipeline.run_test_pipeline(source=video_source)











# import cv2
# import time
# import pickle
# import faiss
# import numpy as np
# from ultralytics import YOLO
# from insightface.app import FaceAnalysis
# from face_recognise import recognize_faces_in_frame
# import logging
# import os
# from datetime import datetime
# import av
# from collections import deque

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov12l-face.pt")
# FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
# CACHE_PATH = os.path.join(BASE_DIR, "face_embeddings_faiss.pkl")

# IOU_THRESHOLD = 0.3
# SIMILARITY_THRESHOLD = 0.3
# K = 1

# save_dir = "saved_frames"
# os.makedirs(save_dir, exist_ok=True)

# AOI_POLYGON = np.array([
#     [8, 424],
#     [642, 372],
#     [744, 1566],
#     [0, 1564],
#     [10, 432]
# ], dtype=np.int32)

# frame_count = 0
# frame_interval = 30
# display_width = 800
# display_height = 600
# buffer_size = 10  # Number of frames to keep in buffer

# frame_buffer = deque(maxlen=buffer_size)

# rtsp_url = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"

# logger.info("Loading YOLO model...")
# model = YOLO(YOLO_MODEL_PATH)

# logger.info("Loading InsightFace...")
# face_analyzer = FaceAnalysis(name='buffalo_l', root='~/.insightface')
# face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# logger.info("Loading FAISS database...")
# with open(CACHE_PATH, "rb") as f:
#     database = pickle.load(f)
# faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# logger.info("Starting live video feed... Press 'Q' to quit.")

# # Open RTSP stream using PyAV
# try:
#     container = av.open(
#         rtsp_url,
#         options={
#             "rtsp_transport": "tcp",
#             "fflags": "nobuffer",
#             "flags": "low_delay",
#             "framedrop": "1",
#             "avioflags": "direct",
#             "err_detect": "ignore_err"  # Skip corrupted frames
#         }
#     )
# except Exception as e:
#     logger.error(f"Failed to open video stream: {e}")
#     exit()

# # Real-time frame decoding loop
# for packet in container.demux(video=0):
#     for img in packet.decode():
#         try:
#             frame = img.to_ndarray(format="bgr24")
#         except Exception as e:
#             logger.warning(f"Skipped corrupted frame: {e}")
#             continue

#         # Add the frame to the buffer
#         frame_buffer.append(frame)

#         # Only process every 'frame_interval' frames
#         if frame_count % frame_interval == 0 and frame_buffer:
#             # Use the most recent frame in buffer
#             current_frame = frame_buffer[-1]

#             recognized_faces = recognize_faces_in_frame(
#                 current_frame,
#                 model,
#                 face_analyzer,
#                 faiss_index,
#                 database,
#                 aoi_polygon=AOI_POLYGON,
#                 iou_threshold=IOU_THRESHOLD,
#                 similarity_threshold=SIMILARITY_THRESHOLD,
#                 k=K
#             )

#             if recognized_faces:
#                 for face in recognized_faces:
#                     x1, y1, x2, y2 = face['bbox']
#                     label = f"{face['label']} ({face['confidence']:.2f})"
#                     color = (0, 255, 0) if face['label'] != "Unknown" else (0, 0, 255)
#                     cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(current_frame, label, (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#                 save_path = os.path.join(save_dir, f"frame_{timestamp}.jpg")
#                 cv2.imwrite(save_path, current_frame)

#         # Draw AOI polygon on the frame
#         cv2.polylines(frame, [AOI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

#         # Resize and display
#         resized_frame = cv2.resize(frame, (display_width, display_height))
#         cv2.imshow("Face Recognition Live", resized_frame)

#         frame_count += 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cv2.destroyAllWindows()
# container.close()

