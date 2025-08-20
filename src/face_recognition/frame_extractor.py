import cv2 

def frame_extract(url):
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    frame_length_gap = 20
    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_i % frame_length_gap == 0:
            if not ret:
                yield None
            else:
                frame = cv2.resize(frame, (640, 480))
                yield frame
        frame_i += 1
    cap.release()
    print("Frame extraction completed.")


