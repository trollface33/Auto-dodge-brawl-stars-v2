"""Threat detection based on motion + size + distance + direction to player."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ThreatDetector:
    def __init__(
        self,
        min_contour_area: int = 30,
        min_motion_pixels: float = 2.0,
        max_tracking_jump: float = 220.0,
        min_threat_score: float = 0.35,
        roi: Optional[dict] = None,
    ):
        self.min_contour_area = int(min_contour_area)
        self.min_motion_pixels = float(min_motion_pixels)
        self.max_tracking_jump = float(max_tracking_jump)
        self.min_threat_score = float(min_threat_score)
        self.roi = roi or {"left": 0.08, "top": 0.10, "right": 0.92, "bottom": 0.92}
        self.previous_candidates: List[Dict] = []
        self.last_threat: Optional[Tuple[int, int]] = None

    @staticmethod
    def _to_numpy(frame):
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(frame).__name__}")
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3:
            raise ValueError(f"Unsupported shape: {frame.shape}")
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported channel count: {frame.shape[2]}")

    def _find_candidates(self, image, player_position):
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        left = int(width * self.roi["left"])
        top = int(height * self.roi["top"])
        right = int(width * self.roi["right"])
        bottom = int(height * self.roi["bottom"])

        roi = hsv[top:bottom, left:right]
        red1 = cv2.inRange(roi, np.array([0, 130, 110]), np.array([10, 255, 255]))
        red2 = cv2.inRange(roi, np.array([170, 130, 110]), np.array([180, 255, 255]))
        yellow = cv2.inRange(roi, np.array([15, 140, 120]), np.array([35, 255, 255]))
        mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        player_x, player_y = player_position
        candidates = []
        max_distance = max(width, height) * 0.48

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            _, _, bw, bh = cv2.boundingRect(contour)
            if bw < 5 or bh < 5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"]) + left
            cy = int(moments["m01"] / moments["m00"]) + top
            distance = float(np.hypot(cx - player_x, cy - player_y))
            if distance <= max_distance:
                candidates.append({
                    "position": (cx, cy),
                    "distance": distance,
                    "area": area,
                })
        return candidates

    def update(self, frame, player_position) -> Optional[Tuple[int, int]]:
        try:
            image = self._to_numpy(frame)
            candidates = self._find_candidates(image, player_position)
            if not candidates:
                self.previous_candidates = []
                return None

            best = None
            for current in candidates:
                nearest = None
                nearest_distance = self.max_tracking_jump
                for previous in self.previous_candidates:
                    prev_pos = previous["position"]
                    jump = float(np.hypot(current["position"][0] - prev_pos[0], current["position"][1] - prev_pos[1]))
                    if jump < nearest_distance:
                        nearest = previous
                        nearest_distance = jump

                if nearest is None:
                    continue

                motion_x = current["position"][0] - nearest["position"][0]
                motion_y = current["position"][1] - nearest["position"][1]
                motion = float(np.hypot(motion_x, motion_y))
                approaching = nearest["distance"] - current["distance"]

                if motion < self.min_motion_pixels or approaching < self.min_motion_pixels:
                    continue

                to_player = np.array([
                    player_position[0] - current["position"][0],
                    player_position[1] - current["position"][1],
                ], dtype=float)
                movement = np.array([motion_x, motion_y], dtype=float)
                norm_to_player = np.linalg.norm(to_player)
                norm_movement = np.linalg.norm(movement)
                if norm_to_player < 1e-6 or norm_movement < 1e-6:
                    continue

                alignment = float(np.dot(movement, to_player)) / (norm_movement * norm_to_player)
                if alignment < 0.35:
                    continue

                score = min(1.0, current["area"] / 900.0) * min(1.0, approaching / 20.0)
                if best is None or score > best[0]:
                    best = (score, current["position"][0], current["position"][1])

            self.previous_candidates = candidates
            if best is None:
                return None
            if best[0] < self.min_threat_score:
                return None

            self.last_threat = (int(best[1]), int(best[2]))
            return self.last_threat
        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            return None
