import cv2
import time

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

start_time = time.time()
duration = 20

while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

