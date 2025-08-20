from onvif import ONVIFCamera
import cv2
import threading
import time

# ====== CONFIGURATION ======
CAMERA_IP = "192.168.0.108"   # Change to your Reolink's IP
ONVIF_PORT = 8000             # ONVIF port (check in Reolink app settings)
USERNAME = "admin"            # Your camera username
PASSWORD = "Kadit1234"    
# rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/h264Preview_01_main"

print(f"[CONFIG] RTSP URL: {RTSP_URL}")
# ====== GLOBAL STATE ======
motion_detected = False
streaming_active = False

# ====== ONVIF EVENT LISTENER ======
def listen_for_motion():
    global motion_detected
    try:
        mycam = ONVIFCamera(CAMERA_IP, ONVIF_PORT, USERNAME, PASSWORD)
        event_service = mycam.create_events_service()
        pullpoint = event_service.CreatePullPointSubscription()
        
        print("[ONVIF] Listening for motion events...")
        while True:
            messages = pullpoint.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
            for msg in messages.NotificationMessage:
                if "Motion" in str(msg):
                    state = "true" in str(msg).lower()
                    motion_detected = state
                    print(f"[ONVIF] Motion detected: {motion_detected}")
            # time.sleep(0.5)
    except Exception as e:
        print("[ERROR] ONVIF listener error:", e)

# ====== STREAM HANDLER ======
def stream_rtsp():
    global streaming_active, motion_detected  # Add motion_detected here
    while True:
        if motion_detected and not streaming_active:
            print("[STREAM] Motion started — opening RTSP...")
            cap = cv2.VideoCapture(RTSP_URL)
            streaming_active = True

            while motion_detected:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Reolink Motion Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    motion_detected = False
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            streaming_active = False
            print("[STREAM] Motion ended — stream closed.")

        # time.sleep(0.2)

# ====== MAIN ======
if __name__ == "__main__":
    # Start ONVIF listener in a separate thread
    threading.Thread(target=listen_for_motion, daemon=True).start()
    # Start RTSP stream controller
    print("[STREAM] Starting RTSP stream controller...")
    stream_rtsp()

