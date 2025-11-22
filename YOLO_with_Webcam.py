import argparse
import os
import time
import math
import os
import time
import math
from ultralytics import YOLO
import cv2
import cvzone

# --- Defaults (edit these variables if you want different weights/camera) ---
WEIGHTS_PATH = "Yolo-Weights/yolov10l.pt"
CAMERA_INDEX = 0


def main():
    weights_path = WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

    model = YOLO(weights_path)

    classNames = [
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
        "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
        "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
        "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    ]

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    prev_frame_time = time.time()

    try:
        while True:
            new_frame_time = time.time()
            success, img = cap.read()
            if not success or img is None:
                print('Failed to read from webcam or got empty frame')
                break

            results = model(img)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(img, (x1, y1, w, h))

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

            fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time != prev_frame_time else 0
            prev_frame_time = new_frame_time
            cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('YOLOv10 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('Released webcam and closed windows')


if __name__ == '__main__':
    main()
