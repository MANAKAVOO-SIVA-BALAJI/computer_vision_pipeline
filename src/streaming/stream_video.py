from flask import Flask, Response
import cv2

# Change this to your video/RTSP URL
# VIDEO_URL = "rtsp://admin:admin@192.168.0.107:1935"
VIDEO_URL = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"

# VIDEO_URL = "videos/entrace_video.mp4"

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {VIDEO_URL}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: Resize to reduce bandwidth
        # frame = cv2.resize(frame, (640, 480))

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Streaming {VIDEO_URL} on http://0.0.0.0:5000/video_feed")
    app.run(host="0.0.0.0", port=5000)
