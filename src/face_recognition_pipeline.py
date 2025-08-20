import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import faiss
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import onnxruntime as ort

# # Suppress info logs
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 2  # only WARNING and above

# # Initialize FaceAnalysis with session options
# app = FaceAnalysis(session_options=sess_options)
# app.prepare(ctx_id=-1)  # 0 = GPU, -1 = CPU


from src.motion_detector import MotionDetector
from src.face_recognise import recognize_faces_in_frame
from src.utils import MetadataHandler, store_attendance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov12l-face.pt")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
CACHE_PATH = os.path.join(BASE_DIR, "face_embeddings_faiss.pkl")

IOU_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.3
K = 1

AOI_POLYGON = np.array([
    [8, 424],
    [642, 372],
    [744, 1566],
    [0, 1564],
    [10, 432]
], dtype=np.int32)

model = YOLO(YOLO_MODEL_PATH)
face_analyzer = FaceAnalysis(name="buffalo_l", root="~/.insightface",session_options=sess_options)
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640)) # 0 = GPU, -1 = CPU 

with open(CACHE_PATH, "rb") as f:
    database = pickle.load(f)

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

class FaceRecognitionPipeline:
    def __init__(self, camera_rtsp_url, motion_detector,
                 window_seconds=10, frame_skip=6, process_interval=None, buffer_size=10):
        self.rtsp_url = camera_rtsp_url
        self.motion_detector = motion_detector
        self.window_seconds = window_seconds
        self.frame_skip = frame_skip
        self.process_interval = process_interval
        self._running = False
        self.backend_url="http://127.0.0.1:8001/store_attendance"
        self.metadata_handler = MetadataHandler(self.backend_url)
        self.capture = None
        self.frame_buffer = deque(maxlen=buffer_size)

        # threaded reader
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.reader_thread = None
        self.reader_running = False

    def opencv_stream(self):
        """Open RTSP stream using OpenCV + FFmpeg."""
        if self.capture is None or not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.capture.isOpened():
                raise Exception("[ERROR] Cannot open RTSP stream with OpenCV/FFmpeg")

    def close_stream(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def start_reader(self):
        """Start background thread to grab frames quickly."""
        def _reader():
            self.opencv_stream()
            while self.reader_running:
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("[WARN] Failed to read frame, reconnecting...")
                    time.sleep(0.5)
                    self.opencv_stream()
                    continue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                    self.frame_queue.put(frame)

        self.reader_running = True
        self.reader_thread = threading.Thread(target=_reader, daemon=True)
        self.reader_thread.start()

    def stop_reader(self):
        self.reader_running = False
        if self.reader_thread:
            self.reader_thread.join()
        self.close_stream()

    def recognize_face(self, frame):
        return recognize_faces_in_frame(
            frame,
            model,
            face_analyzer,
            faiss_index,
            database,
            aoi_polygon=AOI_POLYGON,
            iou_threshold=IOU_THRESHOLD,
            similarity_threshold=SIMILARITY_THRESHOLD,
            k=K
        )

    def run_pipeline(self):
        """Main pipeline: runs only when motion is detected."""
        self._running = True
        main_folder = "detected_frames"
        sub_folder = datetime.now().strftime("%Y_%m_%d")
        save_dir = os.path.join(main_folder, sub_folder)
        os.makedirs(save_dir, exist_ok=True)

        # start background reader
        self.start_reader()

        # FPS tracking
        capture_frames = 0
        processed_frames = 0
        fps_timer = time.time()

        no_motion_logged = False   
        no_motion_start_time = None   # <-- NEW

        try:
            while self._running:
                if self.motion_detector.check_motion():
                    # if motion just resumed, log duration of no-motion
                    if no_motion_logged and no_motion_start_time is not None:
                        duration = time.time() - no_motion_start_time
                        logger.info(f"[INFO] Motion detected after {duration:.1f}s of no motion")
                    else:
                        logger.info("[INFO] Motion detected, starting recognition window")

                    no_motion_logged = False
                    no_motion_start_time = None

                    start_time = time.time()
                    frame_count = 0
                    last_process_time = 0

                    while self._running and (time.time() - start_time < self.window_seconds):
                        if self.frame_queue.empty():
                            # time.sleep(0.01)
                            continue

                        frame = self.frame_queue.get()
                        self.frame_buffer.append(frame)
                        frame_count += 1
                        capture_frames += 1

                        # frame selection logic
                        if self.process_interval is not None:
                            current_time = time.time()
                            if current_time - last_process_time < self.process_interval:
                                continue
                            last_process_time = current_time
                        else:
                            if frame_count % self.frame_skip != 0:
                                continue

                        current_frame = self.frame_buffer[-1]
                        processed_frames += 1
                        
                        frame_processing_start = time.time()

                        result = self.recognize_face(current_frame)

                        time_taken = time.time() - frame_processing_start

                        print(f"[INFO] Frame processed in {time_taken:.2f}s")

                        for face in result:
                            logger.info(f" - Track ID: {face['track_id']}, Label: {face['label']}, Confidence: {face['confidence']}")

                        labels = []

                        for face in result:
                            x1, y1, x2, y2 = face['bbox']
                            label = f"{face['label']} ({face['confidence']:.2f})"
                            color = (0, 255, 0) if face['label'] != "Unknown" else (0, 0, 255)
                            cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(current_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.polylines(current_frame, [AOI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

                            if face['label'] != "Unknown":
                                labels.append(face['label'])

                        file_name = "_".join(labels)
                        if not file_name:
                            file_name = "Unknown"

                        if result:
                            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                            save_path = os.path.join(save_dir, f"{file_name}_{timestamp}.jpg")
                            cv2.imwrite(save_path, current_frame)
                            for face in result:
                                print(f"Label: {face['label']}, Confidence: {face['confidence']}")                           
                        
                        if labels:
                            self.metadata_handler.handle_detection(labels)

                        # # log FPS every 5 seconds
                        # if time.time() - fps_timer >= 5:
                        #     cap_fps = capture_frames / (time.time() - fps_timer)
                        #     proc_fps = processed_frames / (time.time() - fps_timer)
                        #     logger.info(f"[FPS] Capture FPS: {cap_fps:.2f}, Processed FPS: {proc_fps:.2f}")
                        #     capture_frames = 0
                        #     processed_frames = 0
                        #     fps_timer = time.time()

                else:
                    if not no_motion_logged:
                        logger.info("[INFO] No motion detected, waiting...")
                        no_motion_logged = True
                        no_motion_start_time = time.time()   

        except Exception as e:
            logger.error(f"[ERROR] While running pipeline: {e}")

        finally:
            self.stop_reader()
            logger.info("[INFO] Pipeline stopped.")

    def run_test_pipeline(self, source=0):
        """
        Run pipeline in test mode with cv2.imshow().
        Works with a video file path, RTSP URL, or webcam (default=0).
        """
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        main_folder = "detected_frames"
        sub_folder = datetime.now().strftime("%Y_%m_%d")
        save_dir = os.path.join(main_folder, sub_folder)
        os.makedirs(save_dir, exist_ok=True)
        display_width = 800
        display_height = 600

        if not cap.isOpened():
            logger.error(f"[ERROR] Cannot open source: {source}")
            return

        logger.info("[INFO] Starting test pipeline...")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("[WARN] Failed to grab frame, exiting...")
                    break
                
                frame_count += 1

                # Skip frames if requested
                if self.process_interval is None and frame_count % self.frame_skip != 0:
                    continue

                # Run recognition 
                results = self.recognize_face(frame)

                labels = []

                # Draw results
                for face in results:
                    x1, y1, x2, y2 = face['bbox']
                    label = f"{face['label']} ({face['confidence']:.2f})"
                    color = (0, 255, 0) if face['label'] != "Unknown" else (0, 0, 255)

                    if face['label'] != "Unknown":
                        labels.append(face['label'])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                # Show output frame
                resized_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow("Face Recognition Live", resized_frame)

                if results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = os.path.join(save_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(save_path, frame)

                if labels:
                    self.metadata_handler.handle_detection(labels)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("[INFO] Stopping test pipeline...")
                    break

        except Exception as e:
            logger.error(f"[ERROR] Exception in run_test_pipeline: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("[INFO] Test pipeline ended.")

    def stop(self):
        self._running = False







