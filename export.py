"""
export.py - Export model to ONNX and benchmark FPS before vs after.
Usage: python export.py --model runs/detect/ball_finetune/weights/best.pt
"""

import argparse
import time

import cv2
from ultralytics import YOLO

BALL_CLASS_ID = 32


def benchmark_fps(model, source, class_id, num_frames=100):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_count = 0
    start_time = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        model(frame, classes=[class_id], verbose=False)
        frame_count += 1

    elapsed = time.time() - start_time
    cap.release()
    return frame_count / elapsed if elapsed > 0 else 0.0


def export_and_benchmark(model_path: str, source, class_id: int, num_frames: int):
    original_model = YOLO(model_path)

    print(f"Benchmarking original model over {num_frames} frames...")
    original_fps = benchmark_fps(original_model, source, class_id, num_frames)
    print(f"Original model FPS: {original_fps:.2f}")

    print("\nExporting model to ONNX format...")
    onnx_path = original_model.export(format="onnx")
    print(f"Exported to: {onnx_path}")

    onnx_model = YOLO(onnx_path)
    print(f"Benchmarking ONNX model over {num_frames} frames...")
    onnx_fps = benchmark_fps(onnx_model, source, class_id, num_frames)
    print(f"ONNX model FPS: {onnx_fps:.2f}")

    print("\n--- Summary ---")
    print(f"{'Model':<20}{'FPS':<10}")
    print(f"{'Original (.pt)':<20}{original_fps:<10.2f}")
    print(f"{'ONNX export':<20}{onnx_fps:<10.2f}")
    speedup = (onnx_fps / original_fps) if original_fps > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX and benchmark FPS")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--class-id", type=int, default=BALL_CLASS_ID)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source
    export_and_benchmark(args.model, video_source, args.class_id, args.frames)
