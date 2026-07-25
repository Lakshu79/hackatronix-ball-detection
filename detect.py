"""
detect.py - Real-time ball detection using YOLOv8 + OpenCV.
Detects the "sports ball" class from webcam or video, shows FPS live.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

BALL_CLASS_ID = 32  # COCO class id for "sports ball"


def run_detection(model_path: str, source, ball_class_id: int, imgsz: int, resize_width: int):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or camera error.")
            break

        # Optional: shrink the frame before running detection. Lower resolution
        # frames process faster, which helps push FPS up (small accuracy tradeoff).
        if resize_width > 0:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))

        # imgsz controls the resolution YOLO actually runs inference at.
        # Smaller imgsz = faster but slightly less accurate on small objects.
        results = model(frame, classes=[ball_class_id], imgsz=imgsz, verbose=False)
        annotated_frame = results[0].plot()

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Balls detected: {len(results[0].boxes)}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Ball Detection - press q to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ball detection with YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--class-id", type=int, default=BALL_CLASS_ID)
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution passed to YOLO. Lower = faster (default: 640)")
    parser.add_argument("--resize-width", type=int, default=0,
                         help="Resize captured frame to this width before detection, 0 = no resize (default: 0)")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source
    run_detection(args.model, video_source, args.class_id, args.imgsz, args.resize_width)
