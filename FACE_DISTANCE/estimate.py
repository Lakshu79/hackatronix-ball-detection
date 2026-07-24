import argparse
import json
import sys

import cv2

from face_distance import FaceDistanceEstimator, DEFAULT_REAL_FACE_WIDTH_M


def run_on_image(estimator: FaceDistanceEstimator, path: str, out_path: str):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        sys.exit(1)

    estimates = estimator.estimate(frame)
    if not estimates:
        print("No faces detected.")
    else:
        for i, e in enumerate(estimates):
            print(
                f"Face {i}: center=({e.x},{e.y}) w_px={e.w_px} "
                f"depth={e.depth_m:.2f}m angle={e.angle_deg:+.2f}deg"
            )
        results_json = [
            {
                "face_id": i,
                "x_px": e.x,
                "y_px": e.y,
                "w_px": e.w_px,
                "h_px": e.h_px,
                "depth_m": round(e.depth_m, 3),
                "angle_deg": round(e.angle_deg, 2),
                "angle_rad": round(e.angle_rad, 4),
            }
            for i, e in enumerate(estimates)
        ]
        print(json.dumps(results_json, indent=2))

    annotated = estimator.annotate(frame, estimates)
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")


def run_on_webcam(estimator: FaceDistanceEstimator):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    print("Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        estimates = estimator.estimate(frame)
        annotated = estimator.annotate(frame, estimates)
        cv2.imshow("Monocular Face Distance Estimation", annotated)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Monocular face distance + angle estimation")
    ap.add_argument("--image", type=str, help="Path to an input image")
    ap.add_argument("--webcam", action="store_true", help="Use live webcam instead of a static image")
    ap.add_argument("--focal", type=float, required=True, help="Focal length in pixels (from calibrate.py)")
    ap.add_argument("--width", type=float, default=DEFAULT_REAL_FACE_WIDTH_M, help="Assumed real face width, meters")
    ap.add_argument("--out", type=str, default="result.jpg", help="Output path for annotated image (image mode)")
    args = ap.parse_args()

    estimator = FaceDistanceEstimator(focal_length_px=args.focal, real_face_width_m=args.width)

    if args.image:
        run_on_image(estimator, args.image, args.out)
    elif args.webcam:
        run_on_webcam(estimator)
    else:
        print("Provide either --image or --webcam", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
