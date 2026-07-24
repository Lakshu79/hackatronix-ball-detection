import argparse
import sys

import cv2

from face_distance import estimate_focal_length, CASCADE_PATH


def _detect_largest_face_width(frame_bgr) -> int:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return int(faces[0][2])  # w_px of largest face


def main():
    ap = argparse.ArgumentParser(description="Calibrate focal length (pixels) from a reference face image.")
    ap.add_argument("--image", type=str, help="Path to a reference image")
    ap.add_argument("--webcam", action="store_true", help="Capture reference frame from webcam instead")
    ap.add_argument("--distance", type=float, required=True, help="Known distance to face at capture time, meters")
    ap.add_argument("--width", type=float, default=0.15, help="Known/assumed real face width, meters (default 0.15)")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            sys.exit(1)
    elif args.webcam:
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture the reference frame, ESC to cancel.")
        frame = None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.imshow("Calibration - press SPACE", f)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                frame = f
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is None:
            print("No frame captured.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide either --image or --webcam", file=sys.stderr)
        sys.exit(1)

    w_px = _detect_largest_face_width(frame)
    if w_px is None:
        print("No face detected in the reference frame.", file=sys.stderr)
        sys.exit(1)

    f_px = estimate_focal_length(args.distance, args.width, w_px)
    print(f"Detected face width: {w_px}px at {args.distance}m")
    print(f"Calibrated focal length: {f_px:.2f} px")
    print(f"\nUse this with estimate.py, e.g.:")
    print(f"  python3 estimate.py --image test.jpg --focal {f_px:.2f} --width {args.width}")


if __name__ == "__main__":
    main()
