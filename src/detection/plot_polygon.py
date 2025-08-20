import cv2
import json
import numpy as np

# Your polygon JSON data (as a Python dict)
polygon_json = {
    "entrance_resized.jpg": {
        "fileref": "",
        "size": 97578,
        "filename": "entrance_resized.jpg",
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {
            "0": {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        0,
                        52.58317531893222,
                        120.0960398129288,
                        260.0617344956046,
                        310.2847190582118,
                        246.06516502733703,
                        227.9519574801672,
                        0
                    ],
                    "all_points_y": [
                        182.77873070325901,
                        410.84048027444254,
                        480,
                        470.94339622641513,
                        420.7204116638079,
                        191.8353344768439,
                        135.8490566037736,
                        182.77873070325901
                    ]
                },
                "region_attributes": {
                    "label": "In"
                }
            },
            "1": {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        316.04801236867496,
                        326.7512713738207,
                        339.92451322630785,
                        466.7169660564966,
                        507.8833468455189,
                        532.5831753189323,
                        450.2504137408876,
                        399.2041015625,
                        316.04801236867496
                    ],
                    "all_points_y": [
                        196.7753001715266,
                        377.9073756432247,
                        460.2401372212693,
                        446.24356775300174,
                        315.3344768439108,
                        183.60205831903946,
                        167.958833619211,
                        180.3087478559177,
                        196.7753001715266
                    ]
                },
                "region_attributes": {
                    "label": "Out"
                }
            }
        }
    }
}

file_name = "polygon_points.json"
with open(file_name, 'r') as f:
    polygon_json = json.load(f)

label_colors = {
    "In": (0, 255, 0),    # Green
    "Out": (0, 0, 255),   # Red
}

# Extract polygons from the JSON
def get_polygons(data):
    polygons = []
    regions = data["frame0001.jpg"]["regions"]
    for key in regions:
        region = regions[key]
        xs = region["shape_attributes"]["all_points_x"]
        ys = region["shape_attributes"]["all_points_y"]
        pts = np.array(list(zip(xs, ys)), np.int32)
        pts = pts.reshape((-1, 1, 2))
        polygons.append((pts, region["region_attributes"]["label"]))
    return polygons

polygons = get_polygons(polygon_json)

url = "rtsp://admin:Kadit1234@192.168.0.108:554/h264Preview_01_main"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    for pts, label in polygons:
        color = label_colors.get(label, (255, 255, 255))  # fallback white if unknown label

        # Draw polygon outline
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        overlay = frame.copy()
        # cv2.fillPoly(overlay, [pts], color)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Put label text near first polygon point
        x, y = pts[0][0]
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Live Feed with Polygons", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
