"""
train.py - Fine-tune YOLOv8 on a custom ball dataset.
Only needed if the pretrained model doesn't detect your ball reliably.
Usage: python train.py --data data/data.yaml --epochs 50
"""

import argparse

from ultralytics import YOLO


def train_model(data_yaml: str, epochs: int, imgsz: int, base_model: str):
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=10,
        batch=-1,
        name="ball_finetune",
    )
    print("\nTraining complete.")
    print("Best weights saved at: runs/detect/ball_finetune/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom ball dataset")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.imgsz, args.base_model)
