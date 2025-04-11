# Imports
import cv2
import time
from ultralytics import YOLO
import serial

# Configs
title = "@evandanendraa - sidek v4 (press q to quit)"
model = YOLO("best_ncnn_model", task="detect")

GRID_DETECTION_THRESHOLD = 6
CONFIDENCE_THRESHOLD = 0.1
OPTIMIZE = True

TARGETCAMERA = 0
TARGETNUMBER = "62xxxxxxxxxxx"

fps = 0
frame_count = 0
start_time = time.time()
show_class_names = True

# Codes
def send_sms(phone_number, message):
    ser = serial.Serial('/dev/serial0', 9600, timeout=1)
    time.sleep(1)

    ser.write(b'AT+CMGF=1\r')
    time.sleep(1)
    response = ser.read(ser.inWaiting()).decode()
    print("Response:", response)

    ser.write(f'AT+CMGS="{phone_number}"\r'.encode())
    time.sleep(1)
    response = ser.read(ser.inWaiting()).decode()
    print("Response:", response)

    ser.write(f'{message}\x1A'.encode())  # \x1A is Ctrl+Z character
    time.sleep(3)  # Wait for the message to be sent
    response = ser.read(ser.inWaiting()).decode()
    print("Response:", response)

    # Close the serial connection
    ser.close()

cap = cv2.VideoCapture(TARGETCAMERA)
notification_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, show=False, conf=CONFIDENCE_THRESHOLD, stream=OPTIMIZE)

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    detected_grids = set()
    height, width, _ = frame.shape

    for result in results:
        boxes = result.boxes
        classes = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{classes[cls]}: {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if show_class_names:
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            grid_x = int((x1 + x2) / 2) // (width // 3)
            grid_y = int((y1 + y2) / 2) // (height // 3)
            detected_grids.add((grid_x, grid_y))

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    grid_color = (255, 255, 255)
    grid_thickness = 1

    for i in range(1, 3):
        x = int(i * width / 3)
        cv2.line(frame, (x, 0), (x, height), grid_color, grid_thickness)

    for i in range(1, 3):
        y = int(i * height / 3)
        cv2.line(frame, (0, y), (width, y), grid_color, grid_thickness)

    if len(detected_grids) >= GRID_DETECTION_THRESHOLD:
        if not notification_sent:
            text = "GRID LIMIT"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x_center = (width - text_size[0]) // 2
            y_center = height // 2
            cv2.putText(frame, text, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            message = f"[!] LIMIT {len(detected_grids)}/9 | TARGET = {GRID_DETECTION_THRESHOLD}"
            send_sms(TARGETNUMBER, message)
            notification_sent = True
    else:
        notification_sent = False

    cv2.putText(frame, f'Filled Grids: {len(detected_grids)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()