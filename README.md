# CodeAlpha_ObjectDetection

Real-time **object detection and tracking** using Python, OpenCV and Ultralytics YOLO.
Built as **Task 4** of the **CodeAlpha Artificial Intelligence Internship**.

## Description

This application reads video from a webcam (default) or a video file, detects objects in
every frame using a pretrained YOLO model, and tracks them with ByteTrack so that each
object keeps a unique ID while it stays in view. The processed video is shown live with
bounding boxes, class names, confidence scores, tracking IDs, FPS and an object counter.

## Internship Information

- Internship: CodeAlpha - Artificial Intelligence
- Task: Task 4 - Object Detection and Tracking

## Features

- Live webcam input (default) or video file input
- Pretrained YOLO11 object detection (80 COCO classes)
- Bounding boxes, class names and confidence percentage
- ByteTrack tracking with a unique ID per object
- Live FPS display and current object count
- Pause/resume (`P`), screenshot (`S`), quit (`Q`)
- Optional saving of the processed video to `outputs/`
- Error handling for missing webcam, missing video file and model loading problems

## Technologies Used

- Python 3.10+
- OpenCV
- Ultralytics YOLO (YOLO11 nano, `yolo11n.pt`)
- ByteTrack (built into Ultralytics)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_ObjectDetection.git
cd CodeAlpha_ObjectDetection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The YOLO model weights download automatically on first run (internet needed once).

## How to Run

```bash
# Webcam (default)
python object_detection.py

# Video file
python object_detection.py --source path\to\video.mp4

# Save the processed video to outputs/
python object_detection.py --save

# Other options
python object_detection.py --model yolo11s.pt --conf 0.5 --tracker botsort.yaml
```

Controls: `Q` / `ESC` quit, `P` pause/resume, `S` save screenshot.

## How It Works

1. OpenCV reads one frame at a time from the webcam or video file.
2. `model.track()` runs YOLO on the frame to find objects (class, confidence, box).
3. The tracker (ByteTrack) matches the boxes with the boxes from previous frames
   and assigns each object an ID that stays the same across frames.
4. The program draws the boxes, labels, IDs, FPS and object count on the frame.
5. The frame is shown in a window (and optionally written to a video file).

## Example Output

![Demo](outputs/demo_screenshot.png)

## Project Structure

```
CodeAlpha_ObjectDetection/
├── object_detection.py   # Main application
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── .gitignore            # Files excluded from Git
└── outputs/              # Screenshots and saved videos
```

## Future Improvements

- Count objects per class
- Draw movement trails for tracked objects
- Count objects crossing a line
- Add a simple GUI with buttons
- Test larger YOLO models and GPU acceleration
