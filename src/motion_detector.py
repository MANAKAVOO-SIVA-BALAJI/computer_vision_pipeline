from onvif import ONVIFCamera
from zeep.exceptions import Fault
import xml.etree.ElementTree as ET
import datetime
import time

SUBSCRIPTION_BUFFER_SECONDS = 10

class MotionDetector:
    def __init__(self, ip, port, username, password):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password

        self.cam = None
        self.pull_service = None
        self.termination_time = None
        self.fail_count = 0

        # Initialize subscription at creation
        self.connect_and_subscribe()

    def connect_and_subscribe(self):
        """Create a new ONVIF subscription."""
        print("[INFO] Creating new subscription...")
        self.cam = ONVIFCamera(self.ip, self.port, self.username, self.password)
        events_service = self.cam.create_events_service()
        sub_response = events_service.CreatePullPointSubscription({'InitialTerminationTime': 'PT1200S'})
        self.pull_service = self.cam.create_pullpoint_service()

        self.termination_time = sub_response.TerminationTime
        if isinstance(self.termination_time, str):
            self.termination_time = datetime.datetime.fromisoformat(self.termination_time.replace("Z", "+00:00"))
        
        print(f"[INFO] Created Time: {datetime.datetime.now().astimezone()}")
        print(f"[INFO] Subscribed until {self.termination_time.astimezone()}")

    @staticmethod
    def normalize_motion_value(value):
        """Convert ONVIF motion value to boolean."""
        return str(value).lower() in ("true", "1")

    def pull_messages(self):
        """Pull messages from camera and detect motion."""
        try:
            messages = self.pull_service.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
            self.fail_count = 0  # reset fail count on success

            notification_list = getattr(messages, "NotificationMessage", [])
            for msg in notification_list:
                xml_str = ET.tostring(msg.Message._value_1, encoding="unicode")
                if "MotionAlarm" not in xml_str and "Motion" not in xml_str:
                    continue
                root = ET.fromstring(xml_str)
                for simple_item in root.findall(".//{http://www.onvif.org/ver10/schema}SimpleItem"):
                    name = simple_item.attrib.get("Name", "").lower()
                    value = simple_item.attrib.get("Value", "")
                    if name in ("ismotion", "motionalarm"):
                        if self.normalize_motion_value(value):
                            print(f"[INFO] Motion detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            return True
            return False
        except Fault as f:
            print(f"[WARN] ONVIF Fault: {f}")
            self.fail_count += 1
            return None
        except Exception as e:
            print(f"[WARN] Pull failed ({self.fail_count}): {e}")
            print(f"[INFO] Failed Time: {datetime.datetime.now().astimezone()}")

            self.fail_count += 1
            return None

    def check_motion(self):
        """Main method to check motion; handles subscription expiry and reconnection."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # return True
    
        # If no subscription or expired → reconnect
        if self.pull_service is None or self.termination_time is None or \
           (now_utc + datetime.timedelta(seconds=SUBSCRIPTION_BUFFER_SECONDS)) >= self.termination_time:
            print("[INFO] Subscription missing or expiring. Reconnecting...")
            self.connect_and_subscribe()

        motion = self.pull_messages()
        # If pull failed or returned None → reconnect and try again
        if motion is None:
            print("[INFO] Pull failed, reconnecting...")
            self.connect_and_subscribe()
            motion = self.pull_messages()
            if motion is None:
                print("[WARN] Motion check failed after reconnect")
                return False

        return motion

