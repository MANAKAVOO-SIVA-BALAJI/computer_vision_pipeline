import threading
from fastapi import FastAPI
from src.motion_detector import MotionDetector
from src.face_recognition_pipeline import FaceRecognitionPipeline
import logging 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

pipeline = None
pipeline_thread = None
is_running = False
lock = threading.Lock() 

detector = MotionDetector("192.168.0.108", 8000, "admin", "Kadit1234")

def start_pipeline():
    global pipeline, pipeline_thread, is_running

    with lock:
        if is_running:
            logger.warning("Pipeline is already running. but request to start")
            return {"status": "already_running"}

        pipeline = FaceRecognitionPipeline(
            camera_rtsp_url="rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main",
            motion_detector=detector,
            window_seconds=10,
            frame_skip=3
        )

        pipeline_thread = threading.Thread(target=pipeline.run_pipeline, daemon=True)
        pipeline_thread.start()
        is_running = True
    return {"status": "pipeline_started"}

def stop_pipeline():
    global pipeline, pipeline_thread, is_running

    with lock:
        if not is_running or pipeline is None:
            logger.warning("Pipeline is not running. but request to stop")
            return {"status": "not_running"}

        pipeline.stop()
        pipeline_thread.join(timeout=5)
        is_running = False
        pipeline = None
        pipeline_thread = None

    return {"status": "pipeline_stopped"}


@app.post("/start")
def start():
    return start_pipeline()


@app.post("/stop")
def stop():
    return stop_pipeline()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)




