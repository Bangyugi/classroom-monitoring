from __future__ import annotations

import numpy as np
from backend.app.ai.types import (
    CameraProfile,
    PersonDetection,
    PhoneDetectionBox,
    SeatStatus,
    TrackRuntimeState,
)
from backend.app.ai.seat_mapper import KP_CONF_MIN, shoulder_metrics
from backend.app.ai.pose_analyzer import ARM_KP_CONF


def compute_smart_roi_crop(
    person: PersonDetection,
    profile: CameraProfile,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    (sh_x, sh_y), sh_w, _ = shoulder_metrics(person)
    bx1, by1, bx2, by2 = person.bbox_xyxy
    bbox_w = max(1.0, float(bx2 - bx1))
    bbox_h = max(1.0, float(by2 - by1))
    kpts = person.keypoints_xyc

    vis_wrists = [kpts[i] for i in (9, 10) if kpts[i][2] >= ARM_KP_CONF]
    if vis_wrists:
        if len(vis_wrists) == 2 and abs(float(vis_wrists[0][1]) - float(vis_wrists[1][1])) > 0.35 * sh_w:
            active_w = min(vis_wrists, key=lambda w: float(w[1]))
            cx, cy = float(active_w[0]), float(active_w[1])
        else:
            cx = sum(float(w[0]) for w in vis_wrists) / len(vis_wrists)
            cy = sum(float(w[1]) for w in vis_wrists) / len(vis_wrists)
        side = max(2.2 * sh_w, 240.0)
        # Dịch tâm crop lên một chút vì điện thoại thường nằm phía trên cổ tay (trong lòng bàn tay)
        cy_shifted = cy - 0.25 * sh_w
        x1 = int(max(0, cx - side * 0.5))
        y1 = int(max(0, cy_shifted - side * 0.5))
        x2 = int(min(frame_w, cx + side * 0.5))
        y2 = int(min(frame_h, cy_shifted + side * 0.5))
        return x1, y1, x2, y2

    pad_x = 0.35 * bbox_w
    pad_top = 0.25 * bbox_h if profile == CameraProfile.REAR_VIEW else 0.0
    pad_bottom = 0.20 * bbox_h
    x1 = int(max(0, bx1 - pad_x))
    y1 = int(max(0, by1 - pad_top))
    x2 = int(min(frame_w, bx2 + pad_x))
    y2 = int(min(frame_h, by2 + pad_bottom))
    return x1, y1, x2, y2


def map_crop_box_to_frame(
    crop_box: tuple[float, float, float, float],
    roi_xyxy: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    rx1, ry1, _, _ = roi_xyxy
    cx1, cy1, cx2, cy2 = crop_box
    return (cx1 + rx1, cy1 + ry1, cx2 + rx1, cy2 + ry1)


def update_level2_verification(
    state: TrackRuntimeState,
    detections: list[PhoneDetectionBox],
    conf_thres: float = 0.60,
    required_hits: int = 2,
) -> TrackRuntimeState:
    valid = [d for d in detections if d.confidence >= conf_thres]
    if valid:
        best = max(valid, key=lambda d: d.confidence)
        state.phone_hits = min(required_hits + 1, state.phone_hits + 1)
        state.last_confidence = best.confidence
        state.last_phone_box = best.bbox_xyxy
        if state.phone_hits >= required_hits:
            state.status = SeatStatus.VIOLATION
    else:
        if not state.is_posture_suspect:
            state.phone_hits = 0
            state.last_phone_box = None
            state.last_confidence = None
            state.suspect_seconds = 0.0
            state.status = SeatStatus.NORMAL
        else:
            state.phone_hits = max(0, state.phone_hits - 1)
            if state.phone_hits == 0:
                state.last_phone_box = None
                state.last_confidence = None
                if state.status == SeatStatus.VIOLATION:
                    state.status = SeatStatus.SUSPICIOUS
    return state


class YoloPhoneDetector:
    """Wrapper chạy YOLOv8n trên vùng ROI Crop (hỗ trợ cả model COCO lớp 67 và model fine-tuned lớp 0)."""

    def __init__(self, weights_path: str = "yolov8n.pt"):
        self.weights_path = weights_path
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)

    def detect_on_roi(
        self, frame: np.ndarray, roi_xyxy: tuple[int, int, int, int], conf_thres: float = 0.40
    ) -> list[PhoneDetectionBox]:
        self._ensure_loaded()
        x1, y1, x2, y2 = roi_xyxy
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame[y1:y2, x1:x2]
        nc = len(self._model.names)
        target_classes = [0] if nc == 1 else [67]
        results = self._model.predict(crop, conf=conf_thres, classes=target_classes, imgsz=480, verbose=False)
        boxes: list[PhoneDetectionBox] = []
        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                xyxy = tuple(float(v) for v in b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                boxes.append(
                    PhoneDetectionBox(
                        bbox_xyxy=map_crop_box_to_frame(xyxy, roi_xyxy),
                        confidence=conf,
                    )
                )
        return boxes

