"""
evaluate.py - Compute F1, precision, and recall against a validation dataset.
Usage: python evaluate.py --model yolov8n.pt --data data/data.yaml
"""

import argparse

from ultralytics import YOLO


def evaluate_model(model_path: str, data_yaml: str):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    f1 = metrics.box.f1.mean()
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    map50 = metrics.box.map50

    print("\n--- Evaluation Results ---")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"mAP@0.5   : {map50:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1, "map50": map50}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ball detection model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.model, args.data)
