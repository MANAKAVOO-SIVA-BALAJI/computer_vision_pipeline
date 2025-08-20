
import cv2
from face_recognition import frame_extractor, recognizer

rtsp_url = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"

while True:

    for frame in frame_extractor.frame_extract(0):
        if frame is None:
            continue
        result_boxes = recognizer.recognize_faces(frame)
        recognized_frame = recognizer.draw_boxes(frame, result_boxes)
        if recognized_frame is not None:
            cv2.imshow("Face Recognition", recognized_frame)
        else:
            cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    break

cv2.destroyAllWindows()


