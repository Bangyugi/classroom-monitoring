from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class CameraProfile(str, Enum):
    FRONT_VIEW = "FRONT_VIEW"
    REAR_VIEW = "REAR_VIEW"


class SeatStatus(str, Enum):
    EMPTY = "EMPTY"
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    VIOLATION = "VIOLATION"


@dataclass
class PersonDetection:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_xyc: np.ndarray  # shape (17, 3): x, y, conf


@dataclass
class SeatPolygon:
    seat_id: int
    label: str
    polygon_norm: list[tuple[float, float]]
    row_idx: int = 0
    col_idx: int = 0


@dataclass
class PhoneDetectionBox:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass
class TrackRuntimeState:
    track_id: int
    seat_id: int | None = None
    suspect_seconds: float = 0.0
    phone_hits: int = 0
    is_posture_suspect: bool = False
    status: SeatStatus = SeatStatus.NORMAL
    last_confidence: float | None = None
    last_phone_box: tuple[float, float, float, float] | None = None
    last_roi_box: tuple[int, int, int, int] | None = None
    last_snapshot_ts: float = 0.0
    head_ratio: float = 0.0

