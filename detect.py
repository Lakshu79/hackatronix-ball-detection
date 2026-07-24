"""
detect.py - Real-time ball detection using YOLOv8 + OpenCV.
Detects the "sports ball" class from webcam or video, shows FPS live.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

BALL_CLASS_ID = 32  # COCO class id for "sports ball"


def run_detection(model_path: str, source, ball_class_id: int):
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

        results = model(frame, classes=[ball_class_id], verbose=False)
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
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source
    run_detection(args.model, video_source, args.class_id)
