import requests
import time
import datetime

CAMERA_IP = "192.168.0.108"
USERNAME = "admin"
PASSWORD = "Kadit1234"

requests.packages.urllib3.disable_warnings()

def reolink_login():
    """Login to Reolink camera and return auth token."""
    url = f"http://{CAMERA_IP}/cgi-bin/api.cgi?cmd=Login&token=null"
    payload = [{
        "cmd": "Login",
        "param": {
            "User": {
                "userName": USERNAME,
                "password": PASSWORD
            }
        }
    }]
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        token = data[0]["value"]["Token"]["name"]
        print(f"[INFO] Logged in, token: {token}")
        return token
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return None

def get_motion_state(token):
    """Get motion detection state using existing token."""
    url = f"http://{CAMERA_IP}/cgi-bin/api.cgi?cmd=GetMdState&token={token}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data[0]["value"]["state"]  # 1 = motion, 0 = no motion
    except Exception as e:
        print(f"[ERROR] Failed to get motion state: {e}")
        return None

if __name__ == "__main__":
    token = reolink_login()
    if not token:
        exit()
    i = 0
    motion_detected = False
    print("[INFO] Starting motion detection polling...")
    while True:
        state = get_motion_state(token)
        if state == 1:
            if not motion_detected:
                print("Motion detected!")
                print(datetime.datetime.now().isoformat())
            motion_detected = True
            i = 0  
            
        elif state == 0:
            if motion_detected:
                motion_detected = False
                print(f"Motion ended.")
            i += 1
        else:
            print("[WARN] Could not read motion state.")
        # time.sleep(1)  
