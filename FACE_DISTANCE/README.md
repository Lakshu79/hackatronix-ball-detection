# Monocular Face Distance Estimation

HackTronix 2.0 — Track B, Task 2

This is my solution for estimating how far away a face is from the camera
(and how far off-center it is) using just a single 2D image. No stereo
camera, no depth sensor — just one image and the pinhole camera model.

## The idea

I detect the face in the frame, then plug its pixel width and position
into two formulas from the problem statement:

```
Z     = (f * W) / w_px
theta = arctan((x - c_x) / f)
```

- `f` = focal length in pixels (depends on your camera — you calibrate this once)
- `W` = real-world face width, I'm assuming ~0.15 m (average adult face)
- `w_px` = how wide the detected face is in the image, in pixels
- `(x, y)` = center of the detected face
- `(c_x, c_y)` = center of the image

Basically: the further away a face is, the smaller it looks, so I can
back out the distance from how many pixels wide it is. Same idea for the
angle — how far the face center is from the middle of the frame tells you
how far off to the side it is.

## What's in here

- `face_distance.py` — the actual logic. Detects faces with OpenCV's
  built-in Haar cascade and runs the depth/angle math.
- `calibrate.py` — run this once to figure out your camera's focal
  length in pixels. You can't skip this step if you want accurate numbers.
- `estimate.py` — the script I actually run, either on a photo or live
  off my webcam.
- `README.md` — this file.

## Setup

```bash
pip install opencv-python numpy
```

That's it. No GPU, no downloading extra models — I'm using the Haar
cascade that ships with OpenCV.

## Step 1: Calibrate

Before this works properly you need to know your camera's focal length
in pixels, which is different for every camera. I did it like this:
stood exactly 1 meter from my laptop camera, took a photo, and ran:

```bash
python3 calibrate.py --image ref.jpg --distance 1.0 --width 0.15
```

or straight from the webcam:

```bash
python3 calibrate.py --webcam --distance 1.0 --width 0.15
```

It'll spit out an `f_px` number — write that down, you need it for the
next step.

(If I'm being lazy and skip calibration, `f_px ≈ image width in pixels`
is a decent rough guess for most laptop/phone cameras, but the numbers
are noticeably less accurate.)

## Step 2: Run it

On a photo:
```bash
python3 estimate.py --image test.jpg --focal 1333.33 --out result.jpg
```
This prints the depth + angle for every face it finds (as JSON) and
saves an annotated version of the image with boxes and labels.

Live, off the webcam:
```bash
python3 estimate.py --webcam --focal 1333.33
```
Hit ESC to close it.

## What the output looks like

```json
[
  {
    "face_id": 0,
    "x_px": 320, "y_px": 240,
    "w_px": 180, "h_px": 180,
    "depth_m": 1.11,
    "angle_deg": 0.0,
    "angle_rad": 0.0
  }
]
```

## Notes on accuracy

The task allows ±50–150 cm error, and this hits that comfortably as
long as I calibrate properly first — most of the error comes from the
face-width assumption (everyone's face is a slightly different width)
rather than from the math itself.

I went with a Haar cascade because it's fast and needs zero setup, but
if I wanted better detection I'd swap in a DNN face detector instead —
the `detect_faces()` function is the only place that would need to
change, the depth/angle math stays exactly the same either way.

## Integrity note

Everything here is built on OpenCV and the Python standard library —
no external pretrained weights other than the Haar cascade that ships
with OpenCV itself.
