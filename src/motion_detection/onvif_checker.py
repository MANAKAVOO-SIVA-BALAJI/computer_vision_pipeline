from onvif import ONVIFCamera
from zeep.exceptions import Fault
from zeep.helpers import serialize_object
import time
import xml.etree.ElementTree as ET
import datetime
import random

CAMERA_IP = "192.168.0.108"
CAMERA_PORT = 8000
USERNAME = "admin"
PASSWORD = "Kadit1234"

SUBSCRIPTION_BUFFER_SECONDS = 10

def parse_onvif_time(time_str):
    """Convert ONVIF time string to datetime"""
    if time_str.endswith("Z"):
        time_str = time_str[:-1] + "+00:00"  
    return datetime.datetime.fromisoformat(time_str)

def connect_and_subscribe():
    cam = ONVIFCamera(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)
    events_service = cam.create_events_service()
    sub_response = events_service.CreatePullPointSubscription({'InitialTerminationTime': 'PT1200S'})
    pull_service = cam.create_pullpoint_service()
    
    termination_time = sub_response.TerminationTime
    if isinstance(termination_time, str):
        termination_time = datetime.datetime.fromisoformat(termination_time.replace("Z", "+00:00"))
        
    # print("Current time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Subscribed until {termination_time.astimezone()}")


    return cam, pull_service, termination_time

def normalize_motion_value(value):
    """Normalize motion value to boolean"""
    return str(value).lower() in ("true", "1")

def main():
    fail_count = 0
    cam, pull_service, termination_time = connect_and_subscribe()
    print("[INFO] Listening for motion events")
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    while True:
        try:
            
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if now_utc + datetime.timedelta(seconds=SUBSCRIPTION_BUFFER_SECONDS) >= termination_time:
                print("[INFO] Subscription expiring. Reconnecting...", termination_time.astimezone())
                # print("Current time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                cam, pull_service, termination_time = connect_and_subscribe()
                fail_count = 0
            print("send ")
            messages = pull_service.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
            fail_count = 0 

            notification_list = getattr(messages, "NotificationMessage", [])
            if not notification_list:
                continue  

            for msg in notification_list:
                # msg_obj = serialize_object(msg.Message)
                # xml_str = ET.tostring(msg_obj, encoding="unicode")
                xml_str = ET.tostring(msg.Message._value_1, encoding="unicode")
                root = ET.fromstring(xml_str)

                if "MotionAlarm" not in xml_str and "Motion" not in xml_str:
                    continue

                print("Received notification:")
                try:
                    root = ET.fromstring(xml_str)
                    for simple_item in root.findall(".//{http://www.onvif.org/ver10/schema}SimpleItem"):
                        name = simple_item.attrib.get("Name", "").lower()
                        value = simple_item.attrib.get("Value", "")
                        # print(f"[INFO] {name}: {value}")
                        if name in ("ismotion", "motionalarm"):
                            is_motion = normalize_motion_value(value)
                            print(f"[MOTION] {value}")
                            if is_motion:
                                print(f"[INFO] Motion detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    print(f"[WARN] Parse error: {e}")

        except Fault as f:
            print(f"[INFO] No new events or camera fault: {f}")
            fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"[WARN] Pull failed ({fail_count}): {e}")

        if fail_count >= 1:
            print("[INFO] Reconnecting due to repeated pull failures...")
            try:
                cam, pull_service, termination_time = connect_and_subscribe()
                fail_count = 0
            except Exception as e2:
                print(f"[ERROR] Reconnect failed: {e2}")
                # time.sleep(5)

if __name__ == "__main__":
    main()

