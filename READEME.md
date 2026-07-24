# Ball Detection - HackTronix 2.0 (Track B Qualifier)

This is my submission for Task 1 of the Track B qualifier round - a real-time ball
detection system using YOLOv8 and OpenCV.

## What it does

The script opens the webcam, runs YOLOv8 on each frame, and draws a box around
any ball it detects. It also shows the FPS on screen so I can check how fast it's
running.

## Files in this repo

- `detect.py` - main file, run this to see live ball detection
- `train.py` - used this to fine-tune the model if the pretrained one wasn't
  detecting well enough
- `evaluate.py` - calculates F1 score, precision and recall
- `export.py` - converts the model to ONNX format to check if it runs faster
- `requirements.txt` - all the libraries needed
- `README.md` - this file

## How to run it

1. Install the requirements:
```
pip install -r requirements.txt
```

2. Run the detection script:
```
python detect.py
```

3. A window will open showing the webcam feed. When a ball is in frame, it'll draw
   a box around it with a confidence score, plus the FPS in the corner.

4. Press `q` to close the window.

## If detection isn't accurate enough

I used the pretrained `yolov8n.pt` model first since it already has a "sports ball"
class from COCO. If it wasn't detecting well on my specific ball, I planned to
fine-tune it using a dataset from Roboflow:

```
python train.py --data data/data.yaml --epochs 50
```

Then I'd check the F1 score using:
```
python evaluate.py --model runs/detect/ball_finetune/weights/best.pt --data data/data.yaml
```

## Why I picked YOLOv8n

Went with the nano version because it's the smallest/fastest YOLO model, which
matters since the task wants max FPS along with max F1 score. Since we're
running on a normal laptop without GPU, nano gave the best balance without
needing heavy hardware.

## Notes

- Class 32 in COCO is "sports ball" - that's what I filtered detections to.
- FPS is calculated per frame by timing how long each detection loop takes.
- Tried exporting to ONNX with `export.py` to see if it runs faster on CPU.
-