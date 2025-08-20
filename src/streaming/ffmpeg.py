import av
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rtsp_url = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"


try:
    container = av.open(
        rtsp_url,
        options={
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "framedrop": "1",
            "avioflags": "direct"
        }
    )
except Exception as e:
    logger.error(f"Failed to open video stream: {e}")
    exit()

while True:
    try:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")  #
            resized_img = cv2.resize(img, (800, 600))
            cv2.imshow("RTSP Stream", resized_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        logger.warning(f"Frame not received, reconnecting... {e}")
        break

cv2.destroyAllWindows()

