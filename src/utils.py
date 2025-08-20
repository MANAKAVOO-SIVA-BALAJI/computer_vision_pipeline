import csv
import os
import time
import logging
import requests
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
backend_url="http://127.0.0.1:8001/store_attendance"

class MetadataHandler:
    def __init__(self, backend_url="http://127.0.0.1:8001/store_attendance", active_window=300, csv_file="attendance.csv"):
        self.backend_url = backend_url
        self.active_window = active_window
        self.tracked = {}  # {label: last_seen_timestamp}
        self.csv_file = csv_file
        self.lock = Lock()  # ensure thread safety

    def _send_metadata(self, label: str, now: float):
        payload = {
            "timestamp": int(now),
            "label": label,
        }

        try:
            response = requests.post(self.backend_url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f"[SEND] Metadata sent: {payload}")
            else:
                logger.warning(f"[WARN] Failed to send metadata: {response.status_code} {response.text}")
        except requests.RequestException as e:
            logger.error(f"[ERROR] Exception sending metadata: {e}")

    def handle_detection(self, labels):
        """Handles detections (string or list of labels)."""
        if not labels:
            return

        if isinstance(labels, str):
            labels = [labels]

        now = time.time()

        # Cleanup expired labels
        expired = [lbl for lbl, ts in self.tracked.items() if now - ts > self.active_window]
        for lbl in expired:
            logger.debug(f"[CLEANUP] Removing expired '{lbl}'")
            del self.tracked[lbl]

        new_entries = []
        for label in labels:
            if label in self.tracked:
                # Update last seen timestamp
                self.tracked[label] = now
                logger.debug(f"[UPDATE] '{label}' seen again, not sending")
            else:
                # First time seeing this label → send + log
                self._send_metadata(label, now)
                self.tracked[label] = now

                new_entries.append({
                    "label": label,
                    "timestamp": datetime.utcfromtimestamp(now).isoformat()
                })

        if new_entries:
            self.store_attendance(new_entries)

    def store_attendance(self, data):
        """Appends attendance data to CSV (ignores 'Unknown')."""
        filtered_data = [row for row in data if row.get("label") != "Unknown"]
        if not filtered_data:
            return

        with self.lock:
            file_exists = os.path.isfile(self.csv_file)
            with open(self.csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["label", "timestamp"])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(filtered_data)

        logger.info(f"{len(filtered_data)} row(s) appended to {self.csv_file}")

def store_attendance(data, csv_file="attendance.csv"):
    """
    Append a list of dictionaries to a CSV file.
    - Ignores 'bbox' field.
    - Creates file if it doesn't exist.
    - Adds headers if missing.
    - Supports multiple rows.
    """
    print("storing data")
    fieldnames = ['label', 'confidence']

    unknown_removed_data = []

    for i  in data:
        if i.get("label") != "Unknown":
            unknown_removed_data.append(i)
            
    # Filter out unwanted fields (like bbox)
    filtered_data = []
    for row in unknown_removed_data:
        filtered_row = {k: row[k] for k in fieldnames if k in row}
        filtered_data.append(filtered_row)

    file_exists = os.path.isfile(csv_file)

    # Determine if we need to write header
    write_header = True
    if file_exists:
        with open(csv_file, 'r', newline='') as f:
            first_line = f.readline()
            if all(fn in first_line for fn in fieldnames):
                write_header = False

    # Append data to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(filtered_data)

    print(f"{len(filtered_data)} row(s) appended to {csv_file}")



# import csv
# import os
# import requests, logging
# logger = logging.getLogger(__name__)

# BACKEND_URL = "http://128.0.0.1/api/faces"


# import time
# import requests
# import logging

# logger = logging.getLogger(__name__)





# data=[{'track_id': 1, 'label': 'Bhuvana', 'confidence': 0.38, 'bbox': [501, 822, 588, 964]}]

# store_attendance(data)
