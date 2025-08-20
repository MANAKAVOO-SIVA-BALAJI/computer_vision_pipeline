import cv2
import threading
import queue
import time
from ultralytics import YOLO

model = YOLO("models/yolov12n-face.pt")

url = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"
print("Opening video stream...")
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
print("Video stream opened.")
frame_queue = queue.Queue(maxsize=5)

stop_event = threading.Event()
frame_rate = 20
frame_i = 0

def read_frames():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  


def process_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            display_frame = cv2.resize(frame, (960, 540))

            inference_frame = cv2.resize(frame, (480, 270))
            results = model(inference_frame, verbose=False)[0]

            x_scale = 960 / 480
            y_scale = 540 / 270

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)
                conf = float(box.conf[0])
                label = f"Face {conf:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv12n Face Detection", display_frame)
            # print("Processing frame...", frame_rate, frame_i)
            # frame_i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            time.sleep(0.01)


reader_thread = threading.Thread(target=read_frames)
processor_thread = threading.Thread(target=process_frames)

reader_thread.start()
processor_thread.start()

reader_thread.join()
processor_thread.join()

cap.release()
cv2.destroyAllWindows()
