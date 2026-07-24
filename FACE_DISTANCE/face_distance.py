from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

DEFAULT_REAL_FACE_WIDTH_M = 0.15


CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


@dataclass
class FaceEstimate:
    x: int              # face center x, pixels
    y: int               # face center y, pixels
    w_px: int            # face width, pixels
    h_px: int            # face height, pixels
    depth_m: float       # estimated depth Z, meters
    angle_deg: float      # estimated horizontal deviation angle theta, degrees
    angle_rad: float      # same angle in radians
    bbox: Tuple[int, int, int, int]  # (x0, y0, w, h) of the detected face box


class FaceDistanceEstimator:
   

     def __init__(
        self,
        focal_length_px: float,
        real_face_width_m: float = DEFAULT_REAL_FACE_WIDTH_M,
        cascade_path: str = CASCADE_PATH,
    ):
        if focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive")
        self.f = float(focal_length_px)
        self.W = float(real_face_width_m)

        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load cascade at {cascade_path}")

    
     def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x0, y0, w, h) face boxes in pixel coordinates."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [tuple(map(int, f)) for f in faces]

    
     def estimate(self, frame_bgr: np.ndarray) -> List[FaceEstimate]:
        """
        Run detection + the pinhole model on a single frame.
        Returns one FaceEstimate per detected face (largest-first).
        """
        h_img, w_img = frame_bgr.shape[:2]
        c_x, c_y = w_img / 2.0, h_img / 2.0

        boxes = self.detect_faces(frame_bgr)
        # Largest face first (closest / most prominent subject).
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        results: List[FaceEstimate] = []
        for (x0, y0, w_px, h_px) in boxes:
            x_center = x0 + w_px / 2.0
            y_center = y0 + h_px / 2.0

            # Depth: Z = (f * W) / w_px
            depth_m = (self.f * self.W) / w_px

            # Angle: theta = arctan((x - c_x) / f)
            angle_rad = math.atan((x_center - c_x) / self.f)
            angle_deg = math.degrees(angle_rad)

            results.append(
                FaceEstimate(
                    x=int(x_center),
                    y=int(y_center),
                    w_px=w_px,
                    h_px=h_px,
                    depth_m=depth_m,
                    angle_deg=angle_deg,
                    angle_rad=angle_rad,
                    bbox=(x0, y0, w_px, h_px),
                )
            )
        return results

   
     def annotate(self, frame_bgr: np.ndarray, estimates: List[FaceEstimate]) -> np.ndarray:
        """Draw bounding boxes + depth/angle labels on a copy of the frame."""
        out = frame_bgr.copy()
        h_img, w_img = out.shape[:2]
        c_x, c_y = int(w_img / 2), int(h_img / 2)

        # optical-axis marker
        cv2.drawMarker(out, (c_x, c_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        for est in estimates:
            x0, y0, w_px, h_px = est.bbox
            cv2.rectangle(out, (x0, y0), (x0 + w_px, y0 + h_px), (0, 200, 0), 2)
            label = f"Z={est.depth_m:.2f}m  th={est.angle_deg:+.1f}deg"
            cv2.putText(
                out, label, (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA,
            )
            cv2.circle(out, (est.x, est.y), 4, (0, 0, 255), -1)
        return out



def estimate_focal_length(
    known_distance_m: float,
    known_width_m: float,
    width_px: float,
) -> float:
    """
    Standard single-point focal-length calibration:
        f = (w_px * Z) / W
    Take one photo of a face (or reference object) at a KNOWN distance,
    measure its pixel width, and solve for f.
    """
    if known_distance_m <= 0 or known_width_m <= 0 or width_px <= 0:
        raise ValueError("All calibration inputs must be positive")
    return (width_px * known_distance_m) / known_width_m
