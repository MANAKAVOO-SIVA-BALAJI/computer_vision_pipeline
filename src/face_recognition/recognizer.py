from ultralytics import YOLO
import cv2

def load_yolo_model():
    model = YOLO("models/yolov12n-face.pt")
    return model

def resize_frame(frame, target_size=(640, 480)):
    return cv2.resize(frame, target_size)

def draw_boxes(frame, boxes):
    print("Detected boxes:", len(boxes), type(boxes))
    print("Results:", boxes)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

    return frame

def recognize_faces(frame):
    yolo_model = load_yolo_model()
    # frame = resize_frame(frame)
    results = yolo_model(frame,verbose=False)[0]
    return results.boxes 


