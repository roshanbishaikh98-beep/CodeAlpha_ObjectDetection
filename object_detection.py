"""
CodeAlpha AI Internship - Task 4: Object Detection and Tracking

Uses a pretrained YOLO model (Ultralytics) with ByteTrack to detect and
track objects in real time from a webcam or a video file.

Controls (click the video window first):
    Q or ESC -> quit
    P        -> pause / resume
    S        -> save a screenshot to the outputs/ folder
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

WINDOW_NAME = "CodeAlpha - Object Detection and Tracking"
OUTPUT_DIR = Path("outputs")

# Box colours (BGR). Each tracking ID gets one colour so IDs are easy to follow.
COLORS = [
    (56, 56, 255), (31, 150, 255), (0, 190, 240), (50, 200, 50),
    (200, 160, 0), (230, 100, 50), (200, 50, 160), (130, 90, 240),
]


# ----------------------------------------------------------------------------
# 1. Command-line options
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection and tracking with YOLO."
    )
    parser.add_argument(
        "--source", default="0",
        help="0 = default webcam (try 1 or 2 for other cameras), "
             "or the path to a video file. Default: 0",
    )
    parser.add_argument(
        "--model", default="yolo11n.pt",
        help="Pretrained YOLO model. Downloaded automatically the first time. "
             "Default: yolo11n.pt (smallest and fastest)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Minimum confidence (0 to 1) to show a detection. Default: 0.4",
    )
    parser.add_argument(
        "--tracker", default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm. Default: bytetrack.yaml",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the processed video into the outputs/ folder.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# 2. Load the model (with error handling)
# ----------------------------------------------------------------------------
def load_model(model_name):
    try:
        print(f"Loading model '{model_name}' ...")
        return YOLO(model_name)
    except Exception as error:
        print(f"\nERROR: could not load the model '{model_name}'.")
        print("Check the model name and your internet connection "
              "(the first run downloads the model).")
        print(f"Details: {error}")
        sys.exit(1)


# ----------------------------------------------------------------------------
# 3. Open the video source (webcam or file)
# ----------------------------------------------------------------------------
def open_source(source):
    """Returns (capture, is_webcam). capture is None if opening failed."""
    if source.isdigit():
        index = int(source)
        cap = None
        # On Windows, the DirectShow backend opens webcams faster and more reliably.
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(index)
        is_webcam = True
    else:
        if not Path(source).is_file():
            print(f"ERROR: video file not found: {source}")
            return None, False
        cap = cv2.VideoCapture(source)
        is_webcam = False

    if not cap.isOpened():
        cap.release()
        return None, is_webcam
    return cap, is_webcam


# ----------------------------------------------------------------------------
# 4. Video writer (optional saving)
# ----------------------------------------------------------------------------
def create_writer(frame_size, fps):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = OUTPUT_DIR / f"output_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filename), fourcc, fps, frame_size)
    if not writer.isOpened():
        print("WARNING: could not create the output video. Continuing without saving.")
        return None, None
    print(f"Saving processed video to: {filename}")
    return writer, filename


# ----------------------------------------------------------------------------
# 5. Drawing helpers
# ----------------------------------------------------------------------------
def draw_label(frame, text, x, y, color):
    """Draws a filled label box with white text, placed above (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(y - text_h - baseline - 6, 0)  # keep the label inside the frame
    cv2.rectangle(frame, (x, top), (x + text_w + 8, top + text_h + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + text_h + 2), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(frame, result, class_names):
    """Draws boxes, labels, confidence and tracking IDs. Returns object count."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy().astype(int)       # box corners
    confidences = boxes.conf.cpu().numpy()            # confidence 0..1
    classes = boxes.cls.cpu().numpy().astype(int)     # class numbers
    # boxes.id is None when the tracker has not assigned IDs yet
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    for (x1, y1, x2, y2), conf, cls, track_id in zip(xyxy, confidences, classes, ids):
        color = COLORS[track_id % len(COLORS)] if track_id is not None else (160, 160, 160)
        id_text = f"ID {track_id} " if track_id is not None else ""
        label = f"{id_text}{class_names[cls]} {conf * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, x1, y1, color)

    return len(xyxy)


def draw_info_panel(frame, fps, count, saving, paused):
    """Draws the semi-transparent stats panel and the key hints."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (240, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 38), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {count}", (20, 68), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    status = "PAUSED" if paused else ("REC" if saving else "LIVE")
    status_color = (0, 165, 255) if paused else ((0, 0, 255) if saving else (0, 255, 0))
    cv2.putText(frame, status, (20, 92), font, 0.5, status_color, 1, cv2.LINE_AA)

    height = frame.shape[0]
    cv2.putText(frame, "Q: quit   P: pause   S: screenshot", (10, height - 12),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# 6. Main program
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    model = load_model(args.model)

    cap, is_webcam = open_source(args.source)
    if cap is None:
        if is_webcam:
            print("\nERROR: could not open the webcam.")
            print("Tips: close other apps using the camera (Zoom, Teams, Camera app),")
            print("check Windows camera privacy settings, or try '--source 1'.")
        else:
            print("\nERROR: could not open the video file.")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    save_fps = 20.0 if (is_webcam or source_fps <= 0) else source_fps

    writer = None
    if args.save:
        writer, _ = create_writer((frame_w, frame_h), save_fps)

    print("Running... click the video window and press Q to quit.")

    fps = 0.0
    prev_time = time.perf_counter()
    paused = False
    last_frame = None

    try:
        while True:
            if not paused:
                success, frame = cap.read()
                if not success:
                    print("No more frames (end of video, or the camera stopped).")
                    break

                # Detect AND track. persist=True keeps IDs between frames.
                results = model.track(
                    frame, persist=True, tracker=args.tracker,
                    conf=args.conf, verbose=False,
                )

                annotated = frame.copy()
                count = draw_detections(annotated, results[0], model.names)

                # Smoothed FPS so the number does not flicker
                now = time.perf_counter()
                delta = now - prev_time
                prev_time = now
                if delta > 0:
                    instant_fps = 1.0 / delta
                    fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

                draw_info_panel(annotated, fps, count, writer is not None, paused=False)

                if writer is not None:
                    writer.write(annotated)
                last_frame = annotated

            display = last_frame
            if paused and last_frame is not None:
                display = last_frame.copy()
                cv2.putText(display, "PAUSED", (frame_w // 2 - 70, frame_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):          # Q or ESC
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
                prev_time = time.perf_counter()          # avoid a huge FPS jump
            if key in (ord("s"), ord("S")) and last_frame is not None:
                OUTPUT_DIR.mkdir(exist_ok=True)
                path = OUTPUT_DIR / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), last_frame)
                print(f"Screenshot saved: {path}")

            # Stop if the user closes the window with the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        # Always release the camera and close windows, even after an error
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
