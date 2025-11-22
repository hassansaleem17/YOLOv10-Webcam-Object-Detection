# YOLOv10-Webcam-Object-Detection

## Description
This project implements **real-time object detection** using **YOLOv10** on a webcam feed. The application detects multiple object classes such as people, vehicles, animals, and everyday objects, displays bounding boxes with confidence scores, and shows the FPS (frames per second) in real-time.

The project uses:
- **Ultralytics YOLO** for object detection
- **OpenCV** for video capture and display
- **CvZone** for visual enhancements like corner rectangles and text overlay

---

## Features
- Detects over 80 object classes
- Displays bounding boxes with confidence scores
- Shows real-time FPS
- Supports live webcam feed
- Easy-to-modify camera index and YOLO weights

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/YOLOv10-Webcam-Object-Detection.git
cd YOLOv10-Webcam-Object-Detection
````

2. Install dependencies:

```bash
pip install ultralytics opencv-python cvzone
```

3. Add YOLOv10 weights to `Yolo-Weights/yolov10l.pt`.
   You can download weights from the [Ultralytics YOLO website](https://ultralytics.com/).

---

## Usage

Run the main Python script:

```bash
python YOLO_with_Webcam.py
```

* Press `q` to quit the application.
* Make sure your webcam is connected and working.

---

## File Structure

```
YOLOv10-Webcam-Object-Detection/
│
├── Yolo-Weights/
│   └── yolov10l.pt                   # YOLOv10 weights
├── YOLO_with_Webcam.py               # Main Python script
└── README.md                         # Project documentation
```

---

