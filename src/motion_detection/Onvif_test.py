from onvif import ONVIFCamera
from zeep.exceptions import Fault
import time
import xml.etree.ElementTree as ET
import datetime
from zeep.helpers import serialize_object

CAMERA_IP = "192.168.0.108"
CAMERA_PORT = 8000
USERNAME = "admin"
PASSWORD = "Kadit1234"

def connect_and_subscribe():
    cam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)
    events_service = cam.create_events_service()
    events_service.CreatePullPointSubscription({'InitialTerminationTime': 'PT60S'})
    pull_service = cam.create_pullpoint_service()
    return cam, pull_service

def main():
    fail_count = 0
    cam, pull_service = connect_and_subscribe()
    # print(f"[INFO] Connected to {CAMERA_IP}:{CAMERA_PORT}")
    print("[INFO] Listening for motion events")

    while True:
        try:
            messages = pull_service.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
            fail_count = 0  # reset on success

            for msg in getattr(messages, "NotificationMessage", []):
                if not hasattr(msg.Message, "_value_1") or msg.Message._value_1 is None:
                    continue
                # xml_obj = serialize_object(msg.Message)
                # xml_str = ET.tostring(xml_obj, encoding="unicode")
                xml_str = ET.tostring(msg.Message._value_1, encoding="unicode")
                if "MotionAlarm" not in xml_str and "Motion" not in xml_str:
                    continue

                try:
                    root = ET.fromstring(xml_str)
                    for simple_item in root.findall(".//{http://www.onvif.org/ver10/schema}SimpleItem"):
                        name = simple_item.attrib.get("Name", "").lower()
                        value = simple_item.attrib.get("Value", "")
                        print(f"[INFO] {name}: {value}")
                        if name in ("ismotion", "motionalarm"):
                            print(f"[MOTION] {value}")
                            if value.lower() == "true":
                                print(f"[INFO] Motion detected at {datetime.datetime.now()}")
                except Exception as e:
                    print(f"[WARN] Parse error: {e}")

        except Exception as e:
            fail_count += 1
            print(f"[WARN] Pull failed ({fail_count}): {e}")
            # time.sleep(1)

            # Reconnect if too many consecutive failures
            if fail_count >= 5:
                print("[INFO] Reconnecting and re-subscribing...")
                try:
                    cam, pull_service = connect_and_subscribe()
                    fail_count = 0
                except Exception as e2:
                    print(f"[ERROR] Reconnect failed: {e2}")
                    # time.sleep(3)

if __name__ == "__main__":
    main()
