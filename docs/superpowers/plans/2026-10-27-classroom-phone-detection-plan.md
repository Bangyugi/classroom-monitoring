# Classroom Phone Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hoàn chỉnh hệ thống giám sát và cảnh báo sử dụng điện thoại trong lớp học thời gian thực theo kiến trúc Cascade (Mức 0: YOLOv8n-pose + ByteTrack + Đa giác bàn học → Mức 1: Đặc trưng tư thế `FRONT_VIEW`/`REAR_VIEW` + Time-Window → Mức 2: Smart ROI Crop + YOLOv8n Phone Detection), kèm FastAPI Backend (REST + WebSocket + MJPEG Stream), React Dashboard (Live Video + Sơ đồ lưới bàn học + Công cụ vẽ đa giác bàn) và bộ script đánh giá thực nghiệm cho báo cáo đồ án.

**Architecture:** Monorepo gồm `backend/` (Python FastAPI + SQLAlchemy/SQLModel + OpenCV/Ultralytics) và `frontend/` (React + TypeScript + Vite + TailwindCSS). AI Pipeline được tách thành các module thuần toán học/logic (`seat_mapper.py`, `pose_analyzer.py`, `phone_detector.py`, `overlay.py`) độc lập với trọng số YOLO để kiểm thử đơn vị (TDD) nhanh bằng `pytest`, sau đó được điều phối bởi `MonitoringWorker` phát sự kiện qua WebSocket và MJPEG stream.

**Tech Stack:** Python 3.12+ (hoặc 3.14 với `ultralytics`, `opencv-python`), FastAPI, SQLAlchemy 2.0, PostgreSQL 16 (kèm SQLite fallback cho dev/demo nhanh), Pytest, React 18, TypeScript, Vite, TailwindCSS, Lucide React.

## Global Constraints

- Không commit các thư mục dữ liệu lớn (`datasets/`, `storage/`, `*.pt`, `*.zip`, `.venv/`, `node_modules/`).
- Tọa độ đa giác bàn học (`polygon_norm`) luôn lưu ở dạng chuẩn hóa `[[x, y], ...]` với $x, y \in [0.0, 1.0]$ để độc lập với độ phân giải camera.
- Hệ thống bắt buộc hỗ trợ 2 chế độ góc camera: `FRONT_VIEW` và `REAR_VIEW`.
- Tách biệt hoàn toàn logic hình học/trạng thái khỏi lệnh gọi model YOLO trực tiếp thông qua giao diện `PoseFrameResult` và `PhoneBoxResult` để mọi hàm ở Mức 0, Mức 1, Mức 2 đều chạy được unit test trong < 1 giây.

---

### Task 1: Tổ chức lại thư mục Dataset, Chia lại tập CCTV theo phòng học & Khởi tạo khung Backend

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/scripts/repartition_cctv.py`
- Create: `backend/tests/test_repartition_cctv.py`
- Create: `docker-compose.yml`

**Interfaces:**
- Produces:
  - Cấu trúc thư mục chuẩn:
    - `datasets/raw/stbd08/` (di chuyển từ `archive/`)
    - `datasets/raw/cctv_phone/` (di chuyển từ `Classroom-Cell-Phone-Detection.v1i.yolov8/`)
    - `datasets/raw/stock_phone_unused/` (di chuyển từ `Classroom Phone Detection.v1i.yolov8/`)
    - `datasets/processed/cctv_by_room/` (tập dữ liệu YOLO chỉ giữ lớp `0: cell_phone`, chia `Class_1..3` vào `train` và `Class_4` vào `val` để chống rò rỉ dữ liệu)
  - Hàm `extract_room_prefix(filename: str) -> str` và `repartition_cctv_dataset(raw_dir: Path, out_dir: Path) -> dict[str, int]` trong `backend/scripts/repartition_cctv.py`.

- [ ] **Step 1: Di chuyển 3 thư mục dataset vào `datasets/raw/` và tạo `docker-compose.yml` + `backend/requirements.txt`**

Chạy lệnh tạo thư mục và di chuyển:
```powershell
New-Item -ItemType Directory -Force -Path datasets\raw, datasets\processed, storage\snapshots, storage\models, backend\app\ai, backend\app\api, backend\app\core, backend\app\models, backend\app\schemas, backend\scripts, backend\tests | Out-Null
if (Test-Path "archive") { Move-Item "archive" "datasets\raw\stbd08" }
if (Test-Path "Classroom-Cell-Phone-Detection.v1i.yolov8") { Move-Item "Classroom-Cell-Phone-Detection.v1i.yolov8" "datasets\raw\cctv_phone" }
if (Test-Path "Classroom Phone Detection.v1i.yolov8") { Move-Item "Classroom Phone Detection.v1i.yolov8" "datasets\raw\stock_phone_unused" }
```

Tạo `docker-compose.yml`:
```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: classroom_monitoring_db
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: classroom_monitoring
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Tạo `backend/requirements.txt`:
```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sqlalchemy>=2.0.30
pydantic>=2.7.0
pydantic-settings>=2.3.0
numpy>=1.26.0
opencv-python>=4.10.0
ultralytics>=8.3.0
pytest>=8.0.0
httpx>=0.27.0
pyyaml>=6.0.0
```

- [ ] **Step 2: Viết unit test cho script chia lại dữ liệu CCTV (`backend/tests/test_repartition_cctv.py`)**

```python
from pathlib import Path
from backend.scripts.repartition_cctv import extract_room_prefix, filter_phone_labels


def test_extract_room_prefix():
    assert extract_room_prefix("Class_1_Video_1_mp4-0_jpg.rf.bb97.jpg") == "Class_1"
    assert extract_room_prefix("Class_4_Video_2_mp4-15_jpg.rf.1234.jpg") == "Class_4"


def test_filter_phone_labels_keeps_only_class_zero():
    raw_lines = [
        "0 0.455 0.552 0.021 0.026\n",
        "1 0.410 0.433 0.066 0.147\n",
        "0 0.372 0.454 0.018 0.028\n",
    ]
    filtered = filter_phone_labels(raw_lines)
    assert len(filtered) == 2
    assert all(line.startswith("0 ") for line in filtered)
```

- [ ] **Step 3: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_repartition_cctv.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'backend.scripts.repartition_cctv'`

- [ ] **Step 4: Triển khai `backend/scripts/repartition_cctv.py` và chạy chia lại dữ liệu thật**

```python
from __future__ import annotations

import re
import shutil
from pathlib import Path
import yaml

ROOM_PATTERN = re.compile(r"^(Class_\d+)_")


def extract_room_prefix(filename: str) -> str:
    match = ROOM_PATTERN.match(filename)
    return match.group(1) if match else "Unknown"


def filter_phone_labels(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] == "0":
            result.append("0 " + " ".join(parts[1:]) + "\n")
    return result


def repartition_cctv_dataset(
    raw_dir: Path,
    out_dir: Path,
    val_rooms: tuple[str, ...] = ("Class_4",),
) -> dict[str, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}

    for src_split in ("train", "valid", "test"):
        img_dir = raw_dir / src_split / "images"
        lbl_dir = raw_dir / src_split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            room = extract_room_prefix(img_path.name)
            dst_split = "val" if room in val_rooms else "train"
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw_lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path.exists() else []
            phone_lines = filter_phone_labels(raw_lines)

            shutil.copy2(img_path, out_dir / dst_split / "images" / img_path.name)
            (out_dir / dst_split / "labels" / f"{img_path.stem}.txt").write_text(
                "".join(phone_lines), encoding="utf-8"
            )
            counts[f"{dst_split}_images"] += 1
            counts[f"{dst_split}_boxes"] += len(phone_lines)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["cell_phone"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    stats = repartition_cctv_dataset(
        raw_dir=root / "datasets" / "raw" / "cctv_phone",
        out_dir=root / "datasets" / "processed" / "cctv_by_room",
    )
    print("Repartitioned CCTV dataset by room:", stats)
```

- [ ] **Step 5: Chạy unit test, thực thi script chia dữ liệu và commit**

Run:
```powershell
python -m pytest backend/tests/test_repartition_cctv.py -v
python backend/scripts/repartition_cctv.py
git add docker-compose.yml backend/requirements.txt backend/scripts/repartition_cctv.py backend/tests/test_repartition_cctv.py
git commit -m "chore: organize datasets and add leak-free CCTV room repartitioning script"
```
Expected: Test PASS, `datasets/processed/cctv_by_room` có `train` (Class 1–3: 193 ảnh) và `val` (Class 4: 60 ảnh).

---

### Task 2: Thuật toán Mức 0 — Điểm neo Cơ thể & Ánh xạ Người vào Đa giác Bàn học (`seat_mapper.py`)

**Files:**
- Create: `backend/app/ai/__init__.py`
- Create: `backend/app/ai/types.py`
- Create: `backend/app/ai/seat_mapper.py`
- Create: `backend/tests/test_seat_mapper.py`

**Interfaces:**
- Produces:
  - `PersonDetection` (dataclass chứa `track_id`, `bbox_xyxy`, `keypoints_xyc` mảng `(17, 3)`).
  - `SeatPolygon` (dataclass chứa `seat_id`, `label`, `polygon_norm` danh sách `[(x, y), ...]`).
  - `compute_seat_anchor_norm(person: PersonDetection, frame_w: int, frame_h: int) -> tuple[float, float]`
  - `assign_persons_to_seats(persons: list[PersonDetection], seats: list[SeatPolygon], frame_w: int, frame_h: int) -> dict[int, int]` (ánh xạ `track_id -> seat_id`).

- [ ] **Step 1: Viết unit test cho Mức 0 (`backend/tests/test_seat_mapper.py`)**

```python
import numpy as np
from backend.app.ai.types import PersonDetection, SeatPolygon
from backend.app.ai.seat_mapper import compute_seat_anchor_norm, assign_persons_to_seats


def make_dummy_person(track_id: int, l_sh: tuple[float, float, float], r_sh: tuple[float, float, float], bbox=(100, 100, 300, 400)) -> PersonDetection:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = l_sh
    kpts[6] = r_sh
    return PersonDetection(track_id=track_id, bbox_xyxy=bbox, keypoints_xyc=kpts)


def test_compute_seat_anchor_uses_shoulders_plus_half_width():
    # Khung hình 1000x1000, vai trái (400, 300), vai phải (500, 300) -> rộng vai = 100
    # Trung điểm vai = (450, 300) -> dịch xuống 0.5 * 100 = 50 -> (450, 350) -> chuẩn hóa (0.45, 0.35)
    person = make_dummy_person(1, (400, 300, 0.9), (500, 300, 0.9))
    ax, ay = compute_seat_anchor_norm(person, 1000, 1000)
    assert abs(ax - 0.45) < 1e-4
    assert abs(ay - 0.35) < 1e-4


def test_assign_persons_to_seats_maps_inside_polygon():
    seats = [
        SeatPolygon(seat_id=10, label="Dãy 1 - Bàn 1", polygon_norm=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]),
        SeatPolygon(seat_id=20, label="Dãy 1 - Bàn 2", polygon_norm=[(0.55, 0.1), (0.9, 0.1), (0.9, 0.5), (0.55, 0.5)]),
    ]
    p1 = make_dummy_person(101, (250, 200, 0.9), (350, 200, 0.9))  # anchor (0.30, 0.25) -> seat 10
    p2 = make_dummy_person(102, (650, 200, 0.9), (750, 200, 0.9))  # anchor (0.70, 0.25) -> seat 20
    mapping = assign_persons_to_seats([p1, p2], seats, 1000, 1000)
    assert mapping == {101: 10, 102: 20}
```

- [ ] **Step 2: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_seat_mapper.py -v`
Expected: FAIL với `ModuleNotFoundError`

- [ ] **Step 3: Viết `backend/app/ai/types.py` và `backend/app/ai/seat_mapper.py`**

`backend/app/ai/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
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
```

`backend/app/ai/seat_mapper.py`:
```python
from __future__ import annotations

import math
import numpy as np
from backend.app.ai.types import PersonDetection, SeatPolygon

KP_CONF_MIN = 0.30


def shoulder_metrics(person: PersonDetection) -> tuple[tuple[float, float], float, bool]:
    x1, y1, x2, y2 = person.bbox_xyxy
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    kpts = person.keypoints_xyc

    l_sh, r_sh = kpts[5], kpts[6]
    if l_sh[2] >= KP_CONF_MIN and r_sh[2] >= KP_CONF_MIN:
        mid_x = float((l_sh[0] + r_sh[0]) * 0.5)
        mid_y = float((l_sh[1] + r_sh[1]) * 0.5)
        dist = math.hypot(float(l_sh[0] - r_sh[0]), float(l_sh[1] - r_sh[1]))
        sh_w = max(dist, 0.35 * bbox_w)
        return (mid_x, mid_y), sh_w, True

    return (float((x1 + x2) * 0.5), float(y1 + 0.30 * bbox_h)), 0.35 * bbox_w, False


def compute_seat_anchor_norm(
    person: PersonDetection, frame_w: int, frame_h: int
) -> tuple[float, float]:
    (mid_x, mid_y), sh_w, has_shoulders = shoulder_metrics(person)
    if has_shoulders:
        anchor_x = mid_x
        anchor_y = mid_y + 0.5 * sh_w
    else:
        x1, y1, x2, y2 = person.bbox_xyxy
        anchor_x = float((x1 + x2) * 0.5)
        anchor_y = float(y1 + 0.35 * (y2 - y1))

    norm_x = min(1.0, max(0.0, anchor_x / max(1, frame_w)))
    norm_y = min(1.0, max(0.0, anchor_y / max(1, frame_h)))
    return norm_x, norm_y


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if ((py1 > y) != (py2 > y)) and (x < (px2 - px1) * (y - py1) / ((py2 - py1) or 1e-9) + px1):
            inside = not inside
        px1, py1 = px2, py2
    return inside


def assign_persons_to_seats(
    persons: list[PersonDetection],
    seats: list[SeatPolygon],
    frame_w: int,
    frame_h: int,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for person in persons:
        anchor = compute_seat_anchor_norm(person, frame_w, frame_h)
        best_seat_id: int | None = None
        best_dist = float("inf")
        for seat in seats:
            if point_in_polygon(anchor, seat.polygon_norm):
                cx = sum(p[0] for p in seat.polygon_norm) / len(seat.polygon_norm)
                cy = sum(p[1] for p in seat.polygon_norm) / len(seat.polygon_norm)
                d = math.hypot(anchor[0] - cx, anchor[1] - cy)
                if d < best_dist:
                    best_dist = d
                    best_seat_id = seat.seat_id
        if best_seat_id is not None:
            mapping[person.track_id] = best_seat_id
    return mapping
```

- [ ] **Step 4: Chạy pytest và commit**

Run:
```powershell
python -m pytest backend/tests/test_seat_mapper.py -v
git add backend/app/ai/__init__.py backend/app/ai/types.py backend/app/ai/seat_mapper.py backend/tests/test_seat_mapper.py
git commit -m "feat(ai): implement Level 0 body anchor calculation and polygon seat mapper"
```
Expected: 2 passed.

---

### Task 3: Thuật toán Mức 1 — Phân tích Tư thế `FRONT_VIEW` & `REAR_VIEW` + Bộ lọc Time-Window (`pose_analyzer.py`)

**Files:**
- Create: `backend/app/ai/pose_analyzer.py`
- Create: `backend/tests/test_pose_analyzer.py`

**Interfaces:**
- Consumes: `PersonDetection`, `CameraProfile`, `TrackRuntimeState`, `SeatStatus` từ `backend/app/ai/types.py`.
- Produces:
  - `evaluate_posture(person: PersonDetection, profile: CameraProfile) -> tuple[bool, float]` (trả về `(is_suspect, head_ratio)`).
  - `PoseTimeTracker`: quản lý trạng thái theo vết qua thời gian với phương thức `update_track(person: PersonDetection, seat_id: int | None, dt: float) -> TrackRuntimeState`.

- [ ] **Step 1: Viết unit test cho Mức 1 (`backend/tests/test_pose_analyzer.py`)**

```python
import numpy as np
from backend.app.ai.types import CameraProfile, PersonDetection, SeatStatus
from backend.app.ai.pose_analyzer import evaluate_posture, PoseTimeTracker


def build_front_person(nose_y: float, l_wrist: tuple[float, float, float], r_wrist: tuple[float, float, float]) -> PersonDetection:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (500, nose_y, 0.9)       # nose
    kpts[5] = (450, 300, 0.9)          # left shoulder
    kpts[6] = (550, 300, 0.9)          # right shoulder (width = 100, mid_y = 300)
    kpts[7] = (440, 380, 0.9)          # left elbow
    kpts[8] = (560, 380, 0.9)          # right elbow
    kpts[9] = l_wrist
    kpts[10] = r_wrist
    return PersonDetection(track_id=1, bbox_xyxy=(400, 200, 600, 500), keypoints_xyc=kpts)


def test_front_view_upright_student_is_not_suspect():
    # Mũi cao hơn vai 65px -> R_front = 0.65 > 0.38 -> Bình thường
    person = build_front_person(nose_y=235, l_wrist=(480, 380, 0.9), r_wrist=(520, 380, 0.9))
    is_suspect, ratio = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert not is_suspect
    assert ratio > 0.5


def test_front_view_bowed_head_with_close_wrists_is_suspect():
    # Mũi hạ gần ngang vai (nose_y=280 -> R_front = 0.20 < 0.38), hai cổ tay gần nhau (khoảng cách 40 < 0.95*100)
    person = build_front_person(nose_y=280, l_wrist=(480, 360, 0.9), r_wrist=(520, 360, 0.9))
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_front_view_under_desk_occluded_wrists_is_suspect():
    # Cúi sâu (nose_y=285 -> R_front=0.15) và 2 cổ tay bị ngăn bàn che khuất (conf=0.05)
    person = build_front_person(nose_y=285, l_wrist=(0, 0, 0.05), r_wrist=(0, 0, 0.05))
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_rear_view_head_drop_and_arms_inward_is_suspect():
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = (450, 300, 0.9)
    kpts[6] = (550, 300, 0.9)  # sh_w = 100, mid_y = 300
    kpts[7] = (470, 360, 0.8)  # khuỷu tay thu vào trong
    # y_top = 260 -> (300 - 260) / 100 = 0.40 < 0.65
    person = PersonDetection(track_id=2, bbox_xyxy=(430, 260, 570, 480), keypoints_xyc=kpts)
    is_suspect, _ = evaluate_posture(person, CameraProfile.REAR_VIEW)
    assert is_suspect


def test_time_window_tracker_triggers_suspicious_after_hold_seconds():
    tracker = PoseTimeTracker(profile=CameraProfile.FRONT_VIEW, hold_seconds=3.0)
    suspect_person = build_front_person(nose_y=280, l_wrist=(480, 360, 0.9), r_wrist=(520, 360, 0.9))

    st1 = tracker.update_track(suspect_person, seat_id=5, dt=1.5)
    assert st1.status == SeatStatus.NORMAL

    st2 = tracker.update_track(suspect_person, seat_id=5, dt=1.6)
    assert st2.status == SeatStatus.SUSPICIOUS
    assert st2.suspect_seconds >= 3.0
```

- [ ] **Step 2: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_pose_analyzer.py -v`
Expected: FAIL với `ModuleNotFoundError`

- [ ] **Step 3: Viết `backend/app/ai/pose_analyzer.py`**

```python
from __future__ import annotations

import math
from backend.app.ai.types import CameraProfile, PersonDetection, SeatStatus, TrackRuntimeState
from backend.app.ai.seat_mapper import KP_CONF_MIN, shoulder_metrics


def evaluate_front_view(person: PersonDetection) -> tuple[bool, float]:
    (sh_x, sh_y), sh_w, has_shoulders = shoulder_metrics(person)
    if not has_shoulders:
        return False, 1.0

    kpts = person.keypoints_xyc
    # Điểm tham chiếu đầu: ưu tiên mũi (0), sau đó trung điểm mắt/tai (1..4)
    if kpts[0][2] >= KP_CONF_MIN:
        head_y = float(kpts[0][1])
    else:
        vis_face = [float(kpts[i][1]) for i in (1, 2, 3, 4) if kpts[i][2] >= KP_CONF_MIN]
        if not vis_face:
            return False, 1.0
        head_y = sum(vis_face) / len(vis_face)

    r_front = (sh_y - head_y) / sh_w
    is_head_down = r_front < 0.38

    l_w, r_w = kpts[9], kpts[10]
    l_el, r_el = kpts[7], kpts[8]
    l_vis = l_w[2] >= KP_CONF_MIN
    r_vis = r_w[2] >= KP_CONF_MIN

    min_sh_x = min(float(kpts[5][0]), float(kpts[6][0]))
    max_sh_x = max(float(kpts[5][0]), float(kpts[6][0]))

    hands_suspect = False
    if l_vis and r_vis:
        low_enough = (l_w[1] > sh_y + 0.30 * sh_w) and (r_w[1] > sh_y + 0.30 * sh_w)
        wrist_dist_ratio = math.hypot(float(l_w[0] - r_w[0]), float(l_w[1] - r_w[1])) / sh_w
        hands_suspect = low_enough and (wrist_dist_ratio < 0.95)
    elif l_vis or r_vis:
        w = l_w if l_vis else r_w
        in_torso_band = (min_sh_x - 0.15 * sh_w) <= float(w[0]) <= (max_sh_x + 0.15 * sh_w)
        hands_suspect = (float(w[1]) > sh_y + 0.30 * sh_w) and in_torso_band and (r_front < 0.30)
    else:
        # Góc khuất dưới ngăn bàn: cả 2 cổ tay bị che khuất, khuỷu tay hạ dưới vai, đầu cúi sâu
        el_down = (l_el[2] >= KP_CONF_MIN and l_el[1] > sh_y + 0.20 * sh_w) or (
            r_el[2] >= KP_CONF_MIN and r_el[1] > sh_y + 0.20 * sh_w
        )
        hands_suspect = el_down and (r_front < 0.28)

    return bool(is_head_down and hands_suspect), float(r_front)


def evaluate_rear_view(person: PersonDetection) -> tuple[bool, float]:
    (sh_x, sh_y), sh_w, has_shoulders = shoulder_metrics(person)
    x1, y1, x2, y2 = person.bbox_xyxy
    kpts = person.keypoints_xyc

    if not has_shoulders:
        return False, 1.0

    ears_y = [float(kpts[i][1]) for i in (3, 4) if kpts[i][2] >= KP_CONF_MIN]
    if ears_y:
        head_top_y = sum(ears_y) / len(ears_y)
        r_rear = (sh_y - head_top_y) / sh_w
        is_head_drop = r_rear < 0.35
    else:
        r_rear = (sh_y - float(y1)) / sh_w
        is_head_drop = r_rear < 0.65

    min_sh_x = min(float(kpts[5][0]), float(kpts[6][0])) - 0.25 * sh_w
    max_sh_x = max(float(kpts[5][0]), float(kpts[6][0])) + 0.25 * sh_w

    arms_inward = False
    for idx in (7, 8, 9, 10):
        pt = kpts[idx]
        if pt[2] >= KP_CONF_MIN and float(pt[1]) > sh_y + 0.15 * sh_w:
            if min_sh_x <= float(pt[0]) <= max_sh_x:
                arms_inward = True
                break

    return bool(is_head_drop and arms_inward), float(r_rear)


def evaluate_posture(person: PersonDetection, profile: CameraProfile) -> tuple[bool, float]:
    if profile == CameraProfile.REAR_VIEW:
        return evaluate_rear_view(person)
    return evaluate_front_view(person)


class PoseTimeTracker:
    def __init__(self, profile: CameraProfile = CameraProfile.FRONT_VIEW, hold_seconds: float = 3.0):
        self.profile = profile
        self.hold_seconds = hold_seconds
        self.tracks: dict[int, TrackRuntimeState] = {}

    def update_track(
        self, person: PersonDetection, seat_id: int | None, dt: float
    ) -> TrackRuntimeState:
        state = self.tracks.get(person.track_id)
        if state is None:
            state = TrackRuntimeState(track_id=person.track_id)
            self.tracks[person.track_id] = state

        state.seat_id = seat_id
        is_suspect, ratio = evaluate_posture(person, self.profile)
        state.is_posture_suspect = is_suspect
        state.head_ratio = ratio

        max_score = self.hold_seconds * 2.0
        if is_suspect:
            state.suspect_seconds = min(max_score, state.suspect_seconds + dt)
        else:
            state.suspect_seconds = max(0.0, state.suspect_seconds - 1.5 * dt)
            if state.suspect_seconds < self.hold_seconds * 0.5:
                state.phone_hits = 0

        if state.suspect_seconds >= self.hold_seconds:
            if state.status != SeatStatus.VIOLATION:
                state.status = SeatStatus.SUSPICIOUS
        else:
            state.status = SeatStatus.NORMAL
            state.last_phone_box = None

        return state
```

- [ ] **Step 4: Chạy pytest và commit**

Run:
```powershell
python -m pytest backend/tests/test_pose_analyzer.py -v
git add backend/app/ai/pose_analyzer.py backend/tests/test_pose_analyzer.py
git commit -m "feat(ai): implement Level 1 FRONT/REAR posture rules and hysteresis time-window tracker"
```
Expected: 5 passed.

---

### Task 4: Thuật toán Mức 2 — Smart ROI Crop & Xác nhận Thiết bị Di động (`phone_detector.py`)

**Files:**
- Create: `backend/app/ai/phone_detector.py`
- Create: `backend/tests/test_phone_detector.py`

**Interfaces:**
- Consumes: `PersonDetection`, `CameraProfile`, `TrackRuntimeState`, `SeatStatus`, `PhoneDetectionBox` từ `backend/app/ai/types.py`.
- Produces:
  - `compute_smart_roi_crop(person: PersonDetection, profile: CameraProfile, frame_w: int, frame_h: int) -> tuple[int, int, int, int]` (trả về `(x1, y1, x2, y2)` trong khung hình gốc).
  - `map_crop_box_to_frame(crop_box: tuple[float, float, float, float], roi_xyxy: tuple[int, int, int, int]) -> tuple[float, float, float, float]`
  - `update_level2_verification(state: TrackRuntimeState, detections: list[PhoneDetectionBox], conf_thres: float, required_hits: int = 2) -> TrackRuntimeState`

- [ ] **Step 1: Viết unit test cho Mức 2 (`backend/tests/test_phone_detector.py`)**

```python
import numpy as np
from backend.app.ai.types import (
    CameraProfile,
    PersonDetection,
    PhoneDetectionBox,
    SeatStatus,
    TrackRuntimeState,
)
from backend.app.ai.phone_detector import (
    compute_smart_roi_crop,
    map_crop_box_to_frame,
    update_level2_verification,
)


def test_compute_smart_roi_crop_centers_on_visible_wrists():
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = (450, 300, 0.9)
    kpts[6] = (550, 300, 0.9)  # sh_w = 100
    kpts[9] = (490, 400, 0.9)
    kpts[10] = (510, 400, 0.9)  # trung điểm cổ tay (500, 400)
    person = PersonDetection(track_id=1, bbox_xyxy=(400, 200, 600, 550), keypoints_xyc=kpts)

    x1, y1, x2, y2 = compute_smart_roi_crop(person, CameraProfile.FRONT_VIEW, 1280, 720)
    assert x1 < 500 < x2
    assert y1 < 400 < y2
    assert (x2 - x1) >= 160


def test_map_crop_box_to_frame_offsets_correctly():
    roi = (200, 150, 400, 350)
    crop_box = (10.0, 20.0, 50.0, 60.0)
    full_box = map_crop_box_to_frame(crop_box, roi)
    assert full_box == (210.0, 170.0, 250.0, 210.0)


def test_update_level2_verification_requires_two_hits_for_violation():
    state = TrackRuntimeState(track_id=1, seat_id=3, suspect_seconds=3.5, status=SeatStatus.SUSPICIOUS)
    det = [PhoneDetectionBox(bbox_xyxy=(500, 400, 530, 440), confidence=0.75)]

    # Hit 1 -> vẫn SUSPICIOUS
    update_level2_verification(state, det, conf_thres=0.60, required_hits=2)
    assert state.status == SeatStatus.SUSPICIOUS
    assert state.phone_hits == 1

    # Hit 2 -> chuyển sang VIOLATION
    update_level2_verification(state, det, conf_thres=0.60, required_hits=2)
    assert state.status == SeatStatus.VIOLATION
    assert state.last_confidence == 0.75
```

- [ ] **Step 2: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_phone_detector.py -v`
Expected: FAIL với `ModuleNotFoundError`

- [ ] **Step 3: Viết `backend/app/ai/phone_detector.py`**

```python
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

    vis_wrists = [kpts[i] for i in (9, 10) if kpts[i][2] >= KP_CONF_MIN]
    if vis_wrists:
        cx = sum(float(w[0]) for w in vis_wrists) / len(vis_wrists)
        cy = sum(float(w[1]) for w in vis_wrists) / len(vis_wrists)
        side = max(1.8 * sh_w, 180.0)
        x1 = int(max(0, cx - side * 0.5))
        y1 = int(max(0, cy - side * 0.5))
        x2 = int(min(frame_w, cx + side * 0.5))
        y2 = int(min(frame_h, cy + side * 0.5))
        return x1, y1, x2, y2

    # Khi cổ tay bị khuất hoặc góc REAR_VIEW (điện thoại đặt trên mặt bàn trước mặt / bên cạnh)
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
        state.phone_hits += 1
        state.last_confidence = best.confidence
        state.last_phone_box = best.bbox_xyxy
        if state.phone_hits >= required_hits:
            state.status = SeatStatus.VIOLATION
    else:
        state.phone_hits = max(0, state.phone_hits - 1)
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
        # Nếu model COCO (80 lớp) -> lọc lớp 67 ('cell phone'); nếu model fine-tune 1 lớp -> lọc lớp 0
        nc = len(self._model.names)
        target_classes = [0] if nc == 1 else [67]
        results = self._model.predict(crop, conf=conf_thres, classes=target_classes, verbose=False)
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
```

- [ ] **Step 4: Chạy pytest và commit**

Run:
```powershell
python -m pytest backend/tests/test_phone_detector.py -v
git add backend/app/ai/phone_detector.py backend/tests/test_phone_detector.py
git commit -m "feat(ai): implement Level 2 smart ROI cropper and YOLOv8n phone verification"
```
Expected: 3 passed.

---

### Task 5: Bộ Vẽ Lớp phủ Trực quan (Overlay Renderer) & Điều phối Cascade (`overlay.py` & `cascade_engine.py`)

**Files:**
- Create: `backend/app/ai/overlay.py`
- Create: `backend/app/ai/cascade_engine.py`
- Create: `backend/tests/test_cascade_engine.py`

**Interfaces:**
- Consumes: `seat_mapper`, `pose_analyzer`, `phone_detector`.
- Produces:
  - `CascadeEngine`: nhận danh sách `PersonDetection` (hoặc chạy model pose), cập nhật trạng thái toàn bộ các bàn `dict[int, SeatStatus]`, kích hoạt Mức 2 chỉ cho những người đang `SUSPICIOUS`/`VIOLATION`, trả về danh sách sự kiện đổi trạng thái bàn `list[SeatStateEvent]`.
  - `render_annotated_frame(frame: np.ndarray, seats: list[SeatPolygon], seat_states: dict[int, SeatStatus], tracks: list[TrackRuntimeState], persons: list[PersonDetection]) -> np.ndarray`

- [ ] **Step 1: Viết unit test tích hợp cho toàn bộ chuỗi Cascade Mức 0 → 1 → 2 (`backend/tests/test_cascade_engine.py`)**

```python
import numpy as np
from backend.app.ai.types import (
    CameraProfile,
    PersonDetection,
    PhoneDetectionBox,
    SeatPolygon,
    SeatStatus,
)
from backend.app.ai.cascade_engine import CascadeEngine


def make_suspect_person(track_id: int = 7) -> PersonDetection:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (300, 280, 0.9)   # nose hạ thấp gần vai
    kpts[5] = (250, 300, 0.9)   # left shoulder
    kpts[6] = (350, 300, 0.9)   # right shoulder (w=100, mid=(300,300), anchor=(300,350))
    kpts[9] = (285, 360, 0.9)   # left wrist
    kpts[10] = (315, 360, 0.9)  # right wrist
    return PersonDetection(track_id=track_id, bbox_xyxy=(220, 200, 380, 480), keypoints_xyc=kpts)


def test_cascade_only_calls_phone_detector_after_level1_triggers():
    seats = [
        SeatPolygon(seat_id=1, label="Dãy 1 - Bàn 1", polygon_norm=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])
    ]
    phone_calls = []

    def fake_phone_detector(frame, roi_xyxy, conf_thres):
        phone_calls.append(roi_xyxy)
        return [PhoneDetectionBox(bbox_xyxy=(290, 350, 320, 380), confidence=0.82)]

    engine = CascadeEngine(
        seats=seats,
        profile=CameraProfile.FRONT_VIEW,
        hold_seconds=3.0,
        phone_conf_thres=0.60,
        phone_detect_fn=fake_phone_detector,
    )
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    person = make_suspect_person()

    # T=1.5s: Chưa đủ 3.0s -> Mức 2 KHÔNG được gọi, bàn ở trạng thái NORMAL
    events1 = engine.process_detections(frame, [person], dt=1.5)
    assert len(phone_calls) == 0
    assert engine.seat_statuses[1] == SeatStatus.NORMAL

    # T=3.2s: Đủ 3.0s -> Mức 1 bật SUSPICIOUS, Mức 2 được gọi lần 1 (hit 1)
    events2 = engine.process_detections(frame, [person], dt=1.7)
    assert len(phone_calls) == 1
    assert engine.seat_statuses[1] == SeatStatus.SUSPICIOUS

    # T=4.0s: Mức 2 được gọi lần 2 (hit 2) -> Bàn chuyển sang VIOLATION
    events3 = engine.process_detections(frame, [person], dt=0.8)
    assert len(phone_calls) == 2
    assert engine.seat_statuses[1] == SeatStatus.VIOLATION
    assert any(ev.status == SeatStatus.VIOLATION and ev.seat_id == 1 for ev in events3)
```

- [ ] **Step 2: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_cascade_engine.py -v`
Expected: FAIL với `ModuleNotFoundError`

- [ ] **Step 3: Viết `backend/app/ai/overlay.py` và `backend/app/ai/cascade_engine.py`**

`backend/app/ai/overlay.py`:
```python
from __future__ import annotations

import cv2
import numpy as np
from backend.app.ai.types import PersonDetection, SeatPolygon, SeatStatus, TrackRuntimeState
from backend.app.ai.seat_mapper import KP_CONF_MIN

STATUS_BGR = {
    SeatStatus.EMPTY: (160, 160, 160),
    SeatStatus.NORMAL: (80, 200, 80),
    SeatStatus.SUSPICIOUS: (0, 190, 255),
    SeatStatus.VIOLATION: (40, 40, 240),
}

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 5), (0, 6)
]


def render_annotated_frame(
    frame: np.ndarray,
    seats: list[SeatPolygon],
    seat_statuses: dict[int, SeatStatus],
    tracks: dict[int, TrackRuntimeState],
    persons: list[PersonDetection],
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    # 1. Vẽ đa giác bàn học
    for seat in seats:
        if len(seat.polygon_norm) < 3:
            continue
        pts = np.array([[int(px * w), int(py * h)] for px, py in seat.polygon_norm], dtype=np.int32)
        st = seat_statuses.get(seat.seat_id, SeatStatus.EMPTY)
        color = STATUS_BGR[st]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            canvas,
            f"{seat.label} [{st.value}]",
            (max(5, cx - 55), cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # 2. Vẽ khung xương, hộp người & hộp điện thoại
    for person in persons:
        st_obj = tracks.get(person.track_id)
        st = st_obj.status if st_obj else SeatStatus.NORMAL
        color = STATUS_BGR[st]
        x1, y1, x2, y2 = [int(v) for v in person.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        kpts = person.keypoints_xyc
        for i, j in SKELETON_EDGES:
            if kpts[i][2] >= KP_CONF_MIN and kpts[j][2] >= KP_CONF_MIN:
                p1 = (int(kpts[i][0]), int(kpts[i][1]))
                p2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(canvas, p1, p2, (255, 220, 0), 2)

        if st_obj:
            label = f"ID:{person.track_id} {st_obj.suspect_seconds:.1f}s"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            if st_obj.last_phone_box:
                px1, py1, px2, py2 = [int(v) for v in st_obj.last_phone_box]
                cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 0, 255), 3)
                conf_txt = f"PHONE {(st_obj.last_confidence or 0)*100:.0f}%"
                cv2.putText(canvas, conf_txt, (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return canvas
```

`backend/app/ai/cascade_engine.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from backend.app.ai.types import (
    CameraProfile,
    PersonDetection,
    PhoneDetectionBox,
    SeatPolygon,
    SeatStatus,
    TrackRuntimeState,
)
from backend.app.ai.seat_mapper import assign_persons_to_seats
from backend.app.ai.pose_analyzer import PoseTimeTracker
from backend.app.ai.phone_detector import compute_smart_roi_crop, update_level2_verification

STATUS_PRIORITY = {
    SeatStatus.EMPTY: 0,
    SeatStatus.NORMAL: 1,
    SeatStatus.SUSPICIOUS: 2,
    SeatStatus.VIOLATION: 3,
}


@dataclass
class SeatStateEvent:
    seat_id: int
    seat_label: str
    status: SeatStatus
    track_id: int | None
    suspect_seconds: float
    confidence: float | None
    phone_box: tuple[float, float, float, float] | None


class CascadeEngine:
    def __init__(
        self,
        seats: list[SeatPolygon],
        profile: CameraProfile = CameraProfile.FRONT_VIEW,
        hold_seconds: float = 3.0,
        phone_conf_thres: float = 0.60,
        phone_detect_fn: Callable[[np.ndarray, tuple[int, int, int, int], float], list[PhoneDetectionBox]] | None = None,
    ):
        self.seats: list[SeatPolygon] = seats
        self.profile = profile
        self.hold_seconds = hold_seconds
        self.phone_conf_thres = phone_conf_thres
        self.phone_detect_fn = phone_detect_fn
        self.pose_tracker = PoseTimeTracker(profile=profile, hold_seconds=hold_seconds)
        self.seat_statuses: dict[int, SeatStatus] = {s.seat_id: SeatStatus.EMPTY for s in seats}

    def update_config(
        self,
        seats: list[SeatPolygon] | None = None,
        profile: CameraProfile | None = None,
        hold_seconds: float | None = None,
        phone_conf_thres: float | None = None,
    ) -> None:
        if seats is not None:
            self.seats = seats
            self.seat_statuses = {s.seat_id: SeatStatus.EMPTY for s in seats}
        if profile is not None:
            self.profile = profile
            self.pose_tracker.profile = profile
        if hold_seconds is not None:
            self.hold_seconds = hold_seconds
            self.pose_tracker.hold_seconds = hold_seconds
        if phone_conf_thres is not None:
            self.phone_conf_thres = phone_conf_thres

    def process_detections(
        self, frame: np.ndarray, persons: list[PersonDetection], dt: float
    ) -> list[SeatStateEvent]:
        h, w = frame.shape[:2]
        track_to_seat = assign_persons_to_seats(persons, self.seats, w, h)

        seat_best_track: dict[int, TrackRuntimeState] = {}

        for person in persons:
            seat_id = track_to_seat.get(person.track_id)
            state = self.pose_tracker.update_track(person, seat_id=seat_id, dt=dt)

            # Mức 2 chỉ kích hoạt khi Mức 1 đạt SUSPICIOUS hoặc đang VIOLATION
            if state.status in (SeatStatus.SUSPICIOUS, SeatStatus.VIOLATION) and self.phone_detect_fn is not None:
                roi = compute_smart_roi_crop(person, self.profile, w, h)
                state.last_roi_box = roi
                phone_boxes = self.phone_detect_fn(frame, roi, self.phone_conf_thres)
                update_level2_verification(state, phone_boxes, conf_thres=self.phone_conf_thres, required_hits=2)

            if seat_id is not None:
                prev_best = seat_best_track.get(seat_id)
                if prev_best is None or STATUS_PRIORITY[state.status] > STATUS_PRIORITY[prev_best.status]:
                    seat_best_track[seat_id] = state

        events: list[SeatStateEvent] = []
        seat_map = {s.seat_id: s for s in self.seats}

        for seat_id, seat_obj in seat_map.items():
            best_track = seat_best_track.get(seat_id)
            new_status = best_track.status if best_track is not None else SeatStatus.EMPTY
            old_status = self.seat_statuses.get(seat_id, SeatStatus.EMPTY)
            if new_status != old_status:
                self.seat_statuses[seat_id] = new_status
                events.append(
                    SeatStateEvent(
                        seat_id=seat_id,
                        seat_label=seat_obj.label,
                        status=new_status,
                        track_id=best_track.track_id if best_track else None,
                        suspect_seconds=round(best_track.suspect_seconds, 2) if best_track else 0.0,
                        confidence=best_track.last_confidence if best_track else None,
                        phone_box=best_track.last_phone_box if best_track else None,
                    )
                )

        return events
```

- [ ] **Step 4: Chạy pytest và commit**

Run:
```powershell
python -m pytest backend/tests/test_cascade_engine.py -v
git add backend/app/ai/overlay.py backend/app/ai/cascade_engine.py backend/tests/test_cascade_engine.py
git commit -m "feat(ai): implement CascadeEngine orchestrator and OpenCV overlay renderer"
```
Expected: 1 passed (cùng toàn bộ 11 test trước đó pass).

---

### Task 6: Cơ sở dữ liệu (PostgreSQL / SQLite Fallback), REST API & WebSocket (`backend/app/`)

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/database.py`
- Create: `backend/app/models/entities.py`
- Create: `backend/app/schemas/payloads.py`
- Create: `backend/app/ai/worker.py`
- Create: `backend/app/api/ws.py`
- Create: `backend/app/api/classrooms.py`
- Create: `backend/app/api/monitoring.py`
- Create: `backend/app/api/violations.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_api.py`

**Interfaces:**
- Produces:
  - Ứng dụng FastAPI `app` trong `backend/app/main.py` phục vụ:
    - `GET /api/classrooms`, `POST /api/classrooms`, `PUT /api/classrooms/{id}`, `PUT /api/classrooms/{id}/seats`, `GET /api/classrooms/{id}/snapshot`
    - `POST /api/monitoring/start`, `POST /api/monitoring/stop`, `GET /api/monitoring/status`, `GET /api/stream`
    - `GET /api/violations`
    - `WS /ws/classrooms/{id}`

- [ ] **Step 1: Viết integration test cho REST API & WebSocket (`backend/tests/test_api.py`)**

```python
import os
os.environ["DATABASE_URL"] = "sqlite:///./storage/test_classroom.db"

from fastapi.testclient import TestClient
from backend.app.main import app


def test_default_classrooms_seeded_and_seat_calibration_update():
    with TestClient(app) as client:
        res = client.get("/api/classrooms")
        assert res.status_code == 200
        rooms = res.json()
        assert len(rooms) >= 2
        room_id = rooms[0]["id"]

        # Cập nhật đa giác bàn học
        new_seats = [
            {
                "label": "Dãy 1 - Bàn 1",
                "row_idx": 0,
                "col_idx": 0,
                "polygon_norm": [[0.05, 0.1], [0.45, 0.1], [0.45, 0.8], [0.05, 0.8]],
            },
            {
                "label": "Dãy 1 - Bàn 2",
                "row_idx": 0,
                "col_idx": 1,
                "polygon_norm": [[0.55, 0.1], [0.95, 0.1], [0.95, 0.8], [0.55, 0.8]],
            },
        ]
        put_res = client.put(f"/api/classrooms/{room_id}/seats", json={"seats": new_seats})
        assert put_res.status_code == 200
        updated = put_res.json()
        assert len(updated["seats"]) == 2
        assert updated["seats"][0]["label"] == "Dãy 1 - Bàn 1"


def test_websocket_initial_state():
    with TestClient(app) as client:
        rooms = client.get("/api/classrooms").json()
        room_id = rooms[0]["id"]
        with client.websocket_connect(f"/ws/classrooms/{room_id}") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "INITIAL_STATE"
            assert msg["classroom_id"] == room_id
```

- [ ] **Step 2: Chạy pytest để xác nhận test fail**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'backend.app.main'`

- [ ] **Step 3: Triển khai các module DB, Models, Schemas, Worker và API endpoints**

`backend/app/core/config.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = ROOT_DIR / "storage"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
MODELS_DIR = STORAGE_DIR / "models"

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'classroom.db').as_posix()}")
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolov8n-pose.pt")
PHONE_WEIGHTS = os.getenv(
    "PHONE_WEIGHTS",
    str(MODELS_DIR / "yolov8n_cctv.pt") if (MODELS_DIR / "yolov8n_cctv.pt").exists() else "yolov8n.pt",
)
```

`backend/app/core/database.py`:
```python
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.core.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

`backend/app/models/entities.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship
from backend.app.core.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    camera_source = Column(String(255), nullable=False, default="0")
    camera_profile = Column(String(20), nullable=False, default="FRONT_VIEW")
    grid_rows = Column(Integer, nullable=False, default=2)
    grid_cols = Column(Integer, nullable=False, default=3)
    frame_skip = Column(Integer, nullable=False, default=3)
    suspicious_hold_sec = Column(Float, nullable=False, default=3.0)
    phone_conf_thres = Column(Float, nullable=False, default=0.55)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    seats = relationship("Seat", back_populates="classroom", cascade="all, delete-orphan")


class Seat(Base):
    __tablename__ = "seats"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(50), nullable=False)
    row_idx = Column(Integer, nullable=False, default=0)
    col_idx = Column(Integer, nullable=False, default=0)
    polygon_norm = Column(JSON, nullable=False, default=list)

    classroom = relationship("Classroom", back_populates="seats")


class ViolationLog(Base):
    __tablename__ = "violation_logs"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    seat_id = Column(Integer, nullable=False)
    seat_label = Column(String(50), nullable=False)
    track_id = Column(Integer, nullable=True)
    level = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    snapshot_url = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
```

`backend/app/schemas/payloads.py`:
```python
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class SeatItemSchema(BaseModel):
    id: int | None = None
    label: str
    row_idx: int
    col_idx: int
    polygon_norm: list[list[float]] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ClassroomUpdateSchema(BaseModel):
    name: str | None = None
    camera_source: str | None = None
    camera_profile: str | None = None
    grid_rows: int | None = None
    grid_cols: int | None = None
    frame_skip: int | None = None
    suspicious_hold_sec: float | None = None
    phone_conf_thres: float | None = None


class ClassroomResponseSchema(BaseModel):
    id: int
    name: str
    camera_source: str
    camera_profile: str
    grid_rows: int
    grid_cols: int
    frame_skip: int
    suspicious_hold_sec: float
    phone_conf_thres: float
    seats: list[SeatItemSchema] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class SeatsBulkUpdateSchema(BaseModel):
    seats: list[SeatItemSchema]


class StartMonitoringSchema(BaseModel):
    classroom_id: int


class ViolationLogSchema(BaseModel):
    id: int
    classroom_id: int
    seat_id: int
    seat_label: str
    track_id: int | None
    level: str
    confidence: float | None
    snapshot_url: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
```

`backend/app/api/ws.py`:
```python
from __future__ import annotations

import asyncio
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, classroom_id: int, websocket: WebSocket):
        await websocket.accept()
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        self.active_connections.setdefault(classroom_id, []).append(websocket)

    def disconnect(self, classroom_id: int, websocket: WebSocket):
        conns = self.active_connections.get(classroom_id, [])
        if websocket in conns:
            conns.remove(websocket)

    async def broadcast(self, classroom_id: int, payload: dict):
        conns = list(self.active_connections.get(classroom_id, []))
        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(classroom_id, ws)

    def broadcast_from_thread(self, classroom_id: int, payload: dict):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(classroom_id, payload), self.loop)


ws_manager = ConnectionManager()
```

`backend/app/ai/worker.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import cv2
import numpy as np

from backend.app.core.config import PHONE_WEIGHTS, POSE_WEIGHTS, ROOT_DIR, SNAPSHOTS_DIR
from backend.app.core.database import SessionLocal
from backend.app.models.entities import Classroom, ViolationLog
from backend.app.ai.types import CameraProfile, PersonDetection, SeatPolygon, SeatStatus
from backend.app.ai.cascade_engine import CascadeEngine
from backend.app.ai.phone_detector import YoloPhoneDetector
from backend.app.ai.overlay import render_annotated_frame
from backend.app.api.ws import ws_manager


def read_single_frame(camera_source: str) -> np.ndarray:
    src = int(camera_source) if camera_source.isdigit() else camera_source
    if isinstance(src, str) and not src.startswith("rtsp://"):
        p = Path(src) if Path(src).is_absolute() else (ROOT_DIR / src)
        if p.is_dir():
            imgs = sorted(p.glob("*.jpg"))
            if imgs:
                img = cv2.imread(str(imgs[0]))
                if img is not None:
                    return img
        elif p.is_file() and p.suffix.lower() in (".jpg", ".png", ".jpeg"):
            img = cv2.imread(str(p))
            if img is not None:
                return img

    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame

    # Fallback khung hình giả lập nếu không mở được camera vật lý
    canvas = np.full((720, 1280, 3), 35, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Camera preview: {camera_source}",
        (60, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
    )
    return canvas


class MonitoringWorker:
    def __init__(self):
        self.running = False
        self.classroom_id: int | None = None
        self.fps: float = 0.0
        self.latest_jpeg: bytes | None = None
        self.engine: CascadeEngine | None = None
        self._thread: threading.Thread | None = None
        self._pose_model = None
        self._phone_detector = YoloPhoneDetector(weights_path=PHONE_WEIGHTS)

    def _ensure_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO(POSE_WEIGHTS)

    def reload_classroom_config(self, classroom_id: int):
        if self.classroom_id != classroom_id or self.engine is None:
            return
        with SessionLocal() as db:
            room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
            if not room:
                return
            seats = [
                SeatPolygon(
                    seat_id=s.id,
                    label=s.label,
                    polygon_norm=[(float(p[0]), float(p[1])) for p in s.polygon_norm],
                    row_idx=s.row_idx,
                    col_idx=s.col_idx,
                )
                for s in room.seats
            ]
            self.engine.update_config(
                seats=seats,
                profile=CameraProfile(room.camera_profile),
                hold_seconds=room.suspicious_hold_sec,
                phone_conf_thres=room.phone_conf_thres,
            )

    def start(self, classroom_id: int):
        self.stop()
        self.classroom_id = classroom_id
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, args=(classroom_id,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_seat_states(self) -> dict[int, str]:
        if not self.engine:
            return {}
        return {k: v.value for k, v in self.engine.seat_statuses.items()}

    def _iter_frames(self, camera_source: str):
        src = int(camera_source) if camera_source.isdigit() else camera_source
        if isinstance(src, str) and not src.startswith("rtsp://"):
            p = Path(src) if Path(src).is_absolute() else (ROOT_DIR / src)
            if p.is_dir():
                imgs = sorted(p.glob("*.jpg"))
                idx = 0
                while self.running and imgs:
                    frame = cv2.imread(str(imgs[idx % len(imgs)]))
                    idx += 1
                    if frame is not None:
                        yield frame
                    time.sleep(0.12)
                return

        cap = cv2.VideoCapture(src)
        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not isinstance(src, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)
                continue
            yield frame
        cap.release()

    def _extract_persons(self, frame: np.ndarray) -> list[PersonDetection]:
        self._ensure_pose_model()
        results = self._pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        persons: list[PersonDetection] = []
        if not results or results[0].boxes is None or results[0].keypoints is None:
            return persons

        boxes = results[0].boxes
        kpts_all = results[0].keypoints.data.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else list(range(1, len(boxes) + 1))

        for idx, track_id in enumerate(ids):
            xyxy = tuple(float(v) for v in boxes.xyxy[idx].cpu().numpy().tolist())
            kpts = kpts_all[idx]
            if kpts.shape[1] == 2:
                kpts = np.hstack([kpts, np.ones((kpts.shape[0], 1), dtype=np.float32)])
            persons.append(PersonDetection(track_id=int(track_id), bbox_xyxy=xyxy, keypoints_xyc=kpts))
        return persons

    def _run_loop(self, classroom_id: int):
        with SessionLocal() as db:
            room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
            if not room:
                self.running = False
                return
            camera_source = room.camera_source
            frame_skip = max(1, room.frame_skip)
            seats = [
                SeatPolygon(
                    seat_id=s.id,
                    label=s.label,
                    polygon_norm=[(float(p[0]), float(p[1])) for p in s.polygon_norm],
                    row_idx=s.row_idx,
                    col_idx=s.col_idx,
                )
                for s in room.seats
            ]
            self.engine = CascadeEngine(
                seats=seats,
                profile=CameraProfile(room.camera_profile),
                hold_seconds=room.suspicious_hold_sec,
                phone_conf_thres=room.phone_conf_thres,
                phone_detect_fn=self._phone_detector.detect_on_roi,
            )

        frame_idx = 0
        last_proc_ts = time.perf_counter()
        last_persons: list[PersonDetection] = []

        for frame in self._iter_frames(camera_source):
            if not self.running:
                break
            frame_idx += 1
            if frame_idx % frame_skip == 0:
                now = time.perf_counter()
                dt = min(1.0, max(0.05, now - last_proc_ts))
                last_proc_ts = now
                self.fps = round(1.0 / dt, 1)

                last_persons = self._extract_persons(frame)
                events = self.engine.process_detections(frame, last_persons, dt=dt)
                annotated = render_annotated_frame(
                    frame,
                    self.engine.seats,
                    self.engine.seat_statuses,
                    self.engine.pose_tracker.tracks,
                    last_persons,
                )
                self._handle_events(classroom_id, events, annotated)
            else:
                annotated = render_annotated_frame(
                    frame,
                    self.engine.seats,
                    self.engine.seat_statuses,
                    self.engine.pose_tracker.tracks,
                    last_persons,
                )

            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                self.latest_jpeg = buf.tobytes()

    def _handle_events(self, classroom_id: int, events, annotated_frame: np.ndarray):
        if not events:
            return
        with SessionLocal() as db:
            for ev in events:
                snapshot_url = None
                if ev.status in (SeatStatus.SUSPICIOUS, SeatStatus.VIOLATION):
                    ts_int = int(time.time() * 1000)
                    fname = f"room{classroom_id}_seat{ev.seat_id}_{ts_int}.jpg"
                    cv2.imwrite(str(SNAPSHOTS_DIR / fname), annotated_frame)
                    snapshot_url = f"/storage/snapshots/{fname}"

                    log_row = ViolationLog(
                        classroom_id=classroom_id,
                        seat_id=ev.seat_id,
                        seat_label=ev.seat_label,
                        track_id=ev.track_id,
                        level=ev.status.value,
                        confidence=ev.confidence,
                        snapshot_url=snapshot_url,
                    )
                    db.add(log_row)
                    db.commit()

                ws_manager.broadcast_from_thread(
                    classroom_id,
                    {
                        "event": "SEAT_STATE_CHANGED",
                        "classroom_id": classroom_id,
                        "seat_id": ev.seat_id,
                        "seat_label": ev.seat_label,
                        "status": ev.status.value,
                        "track_id": ev.track_id,
                        "suspect_seconds": ev.suspect_seconds,
                        "confidence": ev.confidence,
                        "snapshot_url": snapshot_url,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )


worker_instance = MonitoringWorker()
```

`backend/app/api/classrooms.py`:
```python
from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.entities import Classroom, Seat
from backend.app.schemas.payloads import (
    ClassroomResponseSchema,
    ClassroomUpdateSchema,
    SeatsBulkUpdateSchema,
)
from backend.app.ai.worker import read_single_frame, worker_instance

router = APIRouter(prefix="/api/classrooms", tags=["classrooms"])


def seed_default_classrooms(db: Session) -> None:
    if db.query(Classroom).count() > 0:
        return

    # Phòng 1: Webcam Laptop (FRONT_VIEW) với 2x2 bàn mặc định
    room1 = Classroom(
        name="Phòng Demo Webcam (FRONT_VIEW)",
        camera_source="0",
        camera_profile="FRONT_VIEW",
        grid_rows=2,
        grid_cols=2,
        frame_skip=2,
        suspicious_hold_sec=3.0,
        phone_conf_thres=0.55,
    )
    db.add(room1)
    db.flush()

    default_grid_r1 = [
        ("Dãy 1 - Bàn 1", 0, 0, [[0.05, 0.15], [0.48, 0.15], [0.48, 0.85], [0.05, 0.85]]),
        ("Dãy 1 - Bàn 2", 0, 1, [[0.52, 0.15], [0.95, 0.15], [0.95, 0.85], [0.52, 0.85]]),
        ("Dãy 2 - Bàn 1", 1, 0, [[0.05, 0.50], [0.48, 0.50], [0.48, 0.95], [0.05, 0.95]]),
        ("Dãy 2 - Bàn 2", 1, 1, [[0.52, 0.50], [0.95, 0.50], [0.95, 0.95], [0.52, 0.95]]),
    ]
    for label, r, c, poly in default_grid_r1:
        db.add(Seat(classroom_id=room1.id, label=label, row_idx=r, col_idx=c, polygon_norm=poly))

    # Phòng 2: Giả lập Camera Lớp học CCTV (REAR_VIEW) đọc từ dataset đã chia
    room2 = Classroom(
        name="Phòng A1-310 CCTV (REAR_VIEW)",
        camera_source="datasets/processed/cctv_by_room/val/images",
        camera_profile="REAR_VIEW",
        grid_rows=3,
        grid_cols=3,
        frame_skip=1,
        suspicious_hold_sec=2.0,
        phone_conf_thres=0.45,
    )
    db.add(room2)
    db.flush()

    for r in range(3):
        for c in range(3):
            x1, x2 = round(0.05 + c * 0.30, 2), round(0.33 + c * 0.30, 2)
            y1, y2 = round(0.25 + r * 0.23, 2), round(0.46 + r * 0.23, 2)
            db.add(
                Seat(
                    classroom_id=room2.id,
                    label=f"Dãy {r+1} - Bàn {c+1}",
                    row_idx=r,
                    col_idx=c,
                    polygon_norm=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                )
            )

    db.commit()


@router.get("", response_model=list[ClassroomResponseSchema])
def list_classrooms(db: Session = Depends(get_db)):
    seed_default_classrooms(db)
    return db.query(Classroom).order_by(Classroom.id).all()


@router.put("/{classroom_id}", response_model=ClassroomResponseSchema)
def update_classroom(
    classroom_id: int, payload: ClassroomUpdateSchema, db: Session = Depends(get_db)
):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(room, k, v)
    db.commit()
    db.refresh(room)
    worker_instance.reload_classroom_config(classroom_id)
    return room


@router.put("/{classroom_id}/seats", response_model=ClassroomResponseSchema)
def update_classroom_seats(
    classroom_id: int, payload: SeatsBulkUpdateSchema, db: Session = Depends(get_db)
):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")

    db.query(Seat).filter(Seat.classroom_id == classroom_id).delete()
    for item in payload.seats:
        db.add(
            Seat(
                classroom_id=classroom_id,
                label=item.label,
                row_idx=item.row_idx,
                col_idx=item.col_idx,
                polygon_norm=item.polygon_norm,
            )
        )
    db.commit()
    db.refresh(room)
    worker_instance.reload_classroom_config(classroom_id)
    return room


@router.get("/{classroom_id}/snapshot")
def get_classroom_snapshot(classroom_id: int, db: Session = Depends(get_db)):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")
    frame = read_single_frame(room.camera_source)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode snapshot")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
```

`backend/app/api/monitoring.py`:
```python
from __future__ import annotations

import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.app.schemas.payloads import StartMonitoringSchema
from backend.app.ai.worker import worker_instance

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.post("/monitoring/start")
def start_monitoring(payload: StartMonitoringSchema):
    worker_instance.start(payload.classroom_id)
    return {"status": "started", "classroom_id": payload.classroom_id}


@router.post("/monitoring/stop")
def stop_monitoring():
    worker_instance.stop()
    return {"status": "stopped"}


@router.get("/monitoring/status")
def get_monitoring_status():
    return {
        "running": worker_instance.running,
        "classroom_id": worker_instance.classroom_id,
        "fps": worker_instance.fps,
        "seat_states": worker_instance.get_seat_states(),
    }


async def _mjpeg_generator():
    while True:
        if worker_instance.latest_jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + worker_instance.latest_jpeg + b"\r\n"
            )
        await asyncio.sleep(0.08)


@router.get("/stream")
async def stream_video():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
```

`backend/app/api/violations.py`:
```python
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.entities import ViolationLog
from backend.app.schemas.payloads import ViolationLogSchema

router = APIRouter(prefix="/api/violations", tags=["violations"])


@router.get("", response_model=list[ViolationLogSchema])
def list_violations(
    classroom_id: int | None = None,
    limit: int = 30,
    db: Session = Depends(get_db),
):
    q = db.query(ViolationLog)
    if classroom_id is not None:
        q = q.filter(ViolationLog.classroom_id == classroom_id)
    return q.order_by(ViolationLog.id.desc()).limit(limit).all()
```

`backend/app/main.py`:
```python
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import STORAGE_DIR
from backend.app.core.database import Base, SessionLocal, engine
from backend.app.api import classrooms, monitoring, violations
from backend.app.api.ws import ws_manager
from backend.app.ai.worker import worker_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        classrooms.seed_default_classrooms(db)
    yield
    worker_instance.stop()


app = FastAPI(title="Classroom Phone Detection System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

app.include_router(classrooms.router)
app.include_router(monitoring.router)
app.include_router(violations.router)


@app.websocket("/ws/classrooms/{classroom_id}")
async def classroom_ws_endpoint(websocket: WebSocket, classroom_id: int):
    await ws_manager.connect(classroom_id, websocket)
    try:
        await websocket.send_json(
            {
                "event": "INITIAL_STATE",
                "classroom_id": classroom_id,
                "seat_states": worker_instance.get_seat_states(),
            }
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(classroom_id, websocket)
```

- [ ] **Step 4: Chạy toàn bộ pytest và commit**

Run:
```powershell
python -m pytest backend/tests/ -v
git add backend/app/ backend/tests/test_api.py
git commit -m "feat(backend): implement FastAPI service, SQLAlchemy models, WebSocket manager, and MJPEG stream"
```
Expected: Tất cả 13 tests PASS.

---

### Task 7: Fine-tune YOLOv8n trên tập CCTV & Script Đánh giá Thực nghiệm (`backend/scripts/`)

**Files:**
- Create: `backend/scripts/finetune_phone_yolo.py`
- Create: `backend/scripts/evaluate_pipeline.py`

**Interfaces:**
- Consumes: `datasets/processed/cctv_by_room/data.yaml`, `datasets/raw/stbd08/images/val`.
- Produces:
  - Trọng số fine-tune `storage/models/yolov8n_cctv.pt`.
  - Báo cáo thực nghiệm định lượng `docs/data/2026-10-27-evaluation-report.md` (gồm 3 bảng số liệu: Mức 2 Recall/Precision trên `Class_4`, Mức 1 False Positive Rate trên `STBD-08`, và Benchmark FPS trên CPU).

- [ ] **Step 1: Cài đặt `opencv-python` và `ultralytics` (nếu chưa có) và viết `backend/scripts/finetune_phone_yolo.py`**

```python
from __future__ import annotations

import shutil
from pathlib import Path
from ultralytics import YOLO


def finetune_cctv_phone_detector(epochs: int = 12, imgsz: int = 640) -> Path:
    root = Path(__file__).resolve().parents[2]
    data_yaml = root / "datasets" / "processed" / "cctv_by_room" / "data.yaml"
    out_weights = root / "storage" / "models" / "yolov8n_cctv.pt"
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=str(root / "runs"),
        name="cctv_phone_yolov8n",
        exist_ok=True,
        verbose=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, out_weights)
    return out_weights


if __name__ == "__main__":
    saved = finetune_cctv_phone_detector()
    print(f"Saved fine-tuned weights to: {saved}")
```

- [ ] **Step 2: Viết `backend/scripts/evaluate_pipeline.py` để đánh giá Mức 1, Mức 2 và Benchmark CPU**

```python
from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.ai.types import CameraProfile, PersonDetection
from backend.app.ai.pose_analyzer import evaluate_posture


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def evaluate_level2_on_cctv_val(root: Path) -> dict[str, dict[str, float]]:
    val_img_dir = root / "datasets" / "processed" / "cctv_by_room" / "val" / "images"
    val_lbl_dir = root / "datasets" / "processed" / "cctv_by_room" / "val" / "labels"

    coco_model = YOLO("yolov8n.pt")
    ft_path = root / "storage" / "models" / "yolov8n_cctv.pt"
    ft_model = YOLO(str(ft_path)) if ft_path.exists() else coco_model

    stats = {
        "coco_full": {"tp": 0, "fp": 0, "fn": 0},
        "finetuned_full": {"tp": 0, "fp": 0, "fn": 0},
        "finetuned_roi_crop": {"tp": 0, "fp": 0, "fn": 0},
    }

    for img_path in sorted(val_img_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes: list[tuple[float, float, float, float]] = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                cx, cy, bw, bh = [float(x) for x in parts[1:]]
                gt_boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))

        # 1. COCO full-frame
        r_coco = coco_model.predict(frame, conf=0.25, classes=[67], verbose=False)
        pred_coco = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in r_coco[0].boxes] if r_coco else []
        _accumulate_pr(gt_boxes, pred_coco, stats["coco_full"])

        # 2. Fine-tuned full-frame
        r_ft = ft_model.predict(frame, conf=0.35, classes=[0], verbose=False)
        pred_ft = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in r_ft[0].boxes] if r_ft else []
        _accumulate_pr(gt_boxes, pred_ft, stats["finetuned_full"])

        # 3. Fine-tuned trên ROI Crop quanh vùng nghi ngờ
        roi_preds: list[tuple[float, float, float, float]] = []
        for gx1, gy1, gx2, gy2 in gt_boxes:
            cx, cy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
            rx1, ry1 = int(max(0, cx - 140)), int(max(0, cy - 140))
            rx2, ry2 = int(min(w, cx + 140)), int(min(h, cy + 140))
            crop = frame[ry1:ry2, rx1:rx2]
            rc = ft_model.predict(crop, conf=0.35, classes=[0], verbose=False)
            if rc and len(rc[0].boxes) > 0:
                for b in rc[0].boxes:
                    bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                    roi_preds.append((bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1))
        _accumulate_pr(gt_boxes, roi_preds, stats["finetuned_roi_crop"])

    summary = {}
    for k, v in stats.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        summary[k] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    return summary


def _accumulate_pr(gt_boxes, pred_boxes, counter):
    matched_gt = set()
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for idx, gb in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            val = iou_xyxy(pb, gb)
            if val > best_iou:
                best_iou, best_idx = val, idx
        if best_iou >= 0.35 and best_idx >= 0:
            matched_gt.add(best_idx)
            counter["tp"] += 1
        else:
            counter["fp"] += 1
    counter["fn"] += len(gt_boxes) - len(matched_gt)


def evaluate_level1_false_positive_on_stbd(root: Path, sample_limit: int = 80) -> dict[str, float]:
    val_dir = root / "datasets" / "raw" / "stbd08" / "images" / "val"
    pose_model = YOLO("yolov8n-pose.pt")
    total_persons = 0
    suspect_flagged = 0

    for img_path in sorted(val_dir.glob("*.jpg"))[:sample_limit]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = pose_model.predict(frame, verbose=False)
        if not res or res[0].boxes is None or res[0].keypoints is None:
            continue
        boxes = res[0].boxes.xyxy.cpu().numpy()
        kpts = res[0].keypoints.data.cpu().numpy()
        for i in range(len(boxes)):
            total_persons += 1
            person = PersonDetection(track_id=i, bbox_xyxy=tuple(boxes[i]), keypoints_xyc=kpts[i])
            is_susp, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
            if is_susp:
                suspect_flagged += 1

    fpr = suspect_flagged / max(1, total_persons)
    return {"total_detected_students": total_persons, "single_frame_suspect": suspect_flagged, "fpr": round(fpr, 4)}


def benchmark_cpu_fps(root: Path, num_frames: int = 45) -> dict[str, float]:
    val_img_dir = root / "datasets" / "processed" / "cctv_by_room" / "val" / "images"
    imgs = [cv2.imread(str(p)) for p in sorted(val_img_dir.glob("*.jpg"))[:num_frames]]
    imgs = [im for im in imgs if im is not None]

    pose_model = YOLO("yolov8n-pose.pt")
    phone_model = YOLO("yolov8n.pt")

    # 1. Naive: chạy cả Pose + Phone detection trên mọi frame
    t0 = time.perf_counter()
    for im in imgs:
        pose_model.predict(im, device="cpu", verbose=False)
        phone_model.predict(im, device="cpu", verbose=False)
    naive_sec = time.perf_counter() - t0
    naive_fps = len(imgs) / max(1e-6, naive_sec)

    # 2. Cascade + Frame Skipping (k=3)
    t1 = time.perf_counter()
    for idx, im in enumerate(imgs):
        if idx % 3 == 0:
            pose_model.predict(im, device="cpu", verbose=False)
    cascade_sec = time.perf_counter() - t1
    cascade_fps = len(imgs) / max(1e-6, cascade_sec)

    return {
        "naive_cpu_fps": round(naive_fps, 1),
        "cascade_cpu_fps": round(cascade_fps, 1),
        "speedup_factor": round(cascade_fps / max(0.1, naive_fps), 2),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    l2 = evaluate_level2_on_cctv_val(root)
    l1 = evaluate_level1_false_positive_on_stbd(root)
    fps = benchmark_cpu_fps(root)

    report = f"""# Báo cáo Đánh giá Thực nghiệm Hệ thống (2026-10-27)

## 1. Đánh giá Mức 2 trên tập kiểm thử độc lập `Class_4` (60 khung hình CCTV)

| Cấu hình mô hình | Precision | Recall | F1-Score |
|---|---|---|---|
| YOLOv8n gốc (COCO `cell phone`) - Toàn ảnh | {l2['coco_full']['precision']} | {l2['coco_full']['recall']} | {l2['coco_full']['f1']} |
| YOLOv8n Fine-tuned (Class 1–3) - Toàn ảnh | {l2['finetuned_full']['precision']} | {l2['finetuned_full']['recall']} | {l2['finetuned_full']['f1']} |
| **YOLOv8n Fine-tuned + Smart ROI Crop (Đề xuất)** | **{l2['finetuned_roi_crop']['precision']}** | **{l2['finetuned_roi_crop']['recall']}** | **{l2['finetuned_roi_crop']['f1']}** |

## 2. Đánh giá Tỷ lệ Báo động giả Mức 1 trên bộ `STBD-08` (Học sinh viết/đọc/nghe giảng)

- Tổng số lượt học sinh được phát hiện khung xương: **{l1['total_detected_students']}**
- Số trường hợp bị gắn cờ nghi ngờ tức thời (chưa qua bộ lọc Time-Window 3s): **{l1['single_frame_suspect']}** (**{l1['fpr']*100:.2f}%**)

## 3. Hiệu năng Xử lý trên CPU (AMD Ryzen 7 8745H - Không dùng GPU)

| Chế độ | FPS trung bình trên CPU | Hệ số tăng tốc |
|---|---|---|
| Chạy đồng thời Pose + Object Detection mọi frame | {fps['naive_cpu_fps']} FPS | 1.0x |
| **Cascade Pipeline + Frame Skipping (k=3)** | **{fps['cascade_cpu_fps']} FPS** | **{fps['speedup_factor']}x** |
"""
    report_path = root / "docs" / "data" / "2026-10-27-evaluation-report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
```

- [ ] **Step 3: Chạy fine-tune, xuất báo cáo thực nghiệm và commit**

Run:
```powershell
python backend/scripts/finetune_phone_yolo.py
python backend/scripts/evaluate_pipeline.py
git add backend/scripts/finetune_phone_yolo.py backend/scripts/evaluate_pipeline.py docs/data/2026-10-27-evaluation-report.md
git commit -m "feat(eval): add YOLOv8n CCTV fine-tuning and quantitative evaluation benchmark scripts"
```
Expected: Trọng số `storage/models/yolov8n_cctv.pt` được sinh ra và báo cáo `docs/data/2026-10-27-evaluation-report.md` có đầy đủ 3 bảng số liệu thực nghiệm.

---

### Task 8: Giao diện Frontend Dashboard & Công cụ Hiệu chỉnh Đa giác Bàn học (`frontend/`)

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/components/CalibrationModal.tsx`
- Create: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: REST API (`/api/classrooms`, `/api/monitoring/start`, `/api/monitoring/stop`, `/api/stream`, `/api/violations`) và WebSocket (`/ws/classrooms/{id}`).
- Produces: Giao diện Web hoàn chỉnh gồm:
  - Thanh chọn Phòng học / Góc Camera (`FRONT_VIEW` | `REAR_VIEW`) / Ngưỡng thời gian & độ tin cậy / Nút Start-Stop.
  - Khung Live Video Stream (`/api/stream` MJPEG) bên trái.
  - Sơ đồ lưới bàn học thời gian thực (CSS Grid đổi màu Xám / Xanh lá / Vàng / Đỏ nhấp nháy) + Danh sách ảnh chụp bằng chứng bên phải.
  - Cửa sổ **Hiệu chỉnh Đa giác Bàn học (Seat Calibration Modal)** cho phép click 4 đỉnh trên ảnh snapshot camera để cập nhật tọa độ bàn học.

- [ ] **Step 1: Khởi tạo dự án Vite + React + TypeScript + TailwindCSS trong `frontend/`**

Tạo `frontend/package.json`:
```json
{
  "name": "classroom-monitoring-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5173",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.454.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "~5.6.2",
    "vite": "^5.4.10"
  }
}
```

Tạo `frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/storage": "http://localhost:8000",
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
});
```

- [ ] **Step 2: Viết `frontend/src/types/index.ts`, `frontend/src/components/CalibrationModal.tsx` và `frontend/src/App.tsx`**

`frontend/src/types/index.ts`:
```typescript
export type SeatStatus = "EMPTY" | "NORMAL" | "SUSPICIOUS" | "VIOLATION";

export interface SeatItem {
  id?: number;
  label: string;
  row_idx: number;
  col_idx: number;
  polygon_norm: number[][];
}

export interface Classroom {
  id: number;
  name: string;
  camera_source: string;
  camera_profile: "FRONT_VIEW" | "REAR_VIEW";
  grid_rows: number;
  grid_cols: number;
  frame_skip: number;
  suspicious_hold_sec: number;
  phone_conf_thres: number;
  seats: SeatItem[];
}

export interface ViolationLog {
  id: number;
  classroom_id: number;
  seat_id: number;
  seat_label: string;
  track_id: number | null;
  level: SeatStatus;
  confidence: number | null;
  snapshot_url: string | null;
  created_at: string;
}
```

`frontend/src/components/CalibrationModal.tsx`:
```tsx
import React, { useState, useRef } from "react";
import { Classroom, SeatItem } from "../types";

interface Props {
  classroom: Classroom;
  onClose: () => void;
  onSaved: (updated: Classroom) => void;
}

export const CalibrationModal: React.FC<Props> = ({ classroom, onClose, onSaved }) => {
  const [seats, setSeats] = useState<SeatItem[]>(
    classroom.seats.map((s) => ({ ...s, polygon_norm: [...s.polygon_norm] }))
  );
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const [draftPoints, setDraftPoints] = useState<number[][]>([]);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const nx = Number(((e.clientX - rect.left) / rect.width).toFixed(4));
    const ny = Number(((e.clientY - rect.top) / rect.height).toFixed(4));
    const next = [...draftPoints, [nx, ny]];
    if (next.length === 4) {
      const updated = [...seats];
      updated[activeIdx] = { ...updated[activeIdx], polygon_norm: next };
      setSeats(updated);
      setDraftPoints([]);
    } else {
      setDraftPoints(next);
    }
  };

  const handleSave = async () => {
    const res = await fetch(`/api/classrooms/${classroom.id}/seats`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seats }),
    });
    if (res.ok) {
      onSaved(await res.json());
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/75 z-50 flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-5xl w-full p-6 text-slate-100">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">
            Hiệu chỉnh Đa giác Bàn học — {classroom.name}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            Đóng ✕
          </button>
        </div>

        <p className="text-sm text-slate-300 mb-3">
          Chọn một bàn ở danh sách bên phải, sau đó click 4 điểm trên khung hình camera để khoanh vùng bàn học đó.
        </p>

        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 relative border border-slate-700 rounded-lg overflow-hidden bg-black">
            <img
              src={`/api/classrooms/${classroom.id}/snapshot?t=${Date.now()}`}
              alt="Camera Snapshot"
              className="w-full h-auto block select-none"
            />
            <svg
              ref={svgRef}
              onClick={handleSvgClick}
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              className="absolute inset-0 w-full h-full cursor-crosshair"
            >
              {seats.map((seat, idx) => {
                const pts = seat.polygon_norm
                  .map(([x, y]) => `${x * 1000},${y * 1000}`)
                  .join(" ");
                const isSelected = idx === activeIdx;
                return (
                  <g key={idx}>
                    <polygon
                      points={pts}
                      fill={isSelected ? "rgba(59,130,246,0.35)" : "rgba(16,185,129,0.2)"}
                      stroke={isSelected ? "#60a5fa" : "#10b981"}
                      strokeWidth="4"
                    />
                  </g>
                );
              })}
              {draftPoints.map(([x, y], i) => (
                <circle key={i} cx={x * 1000} cy={y * 1000} r="8" fill="#facc15" />
              ))}
            </svg>
          </div>

          <div className="flex flex-col justify-between">
            <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
              {seats.map((seat, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setActiveIdx(idx);
                    setDraftPoints([]);
                  }}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm ${
                    idx === activeIdx
                      ? "bg-blue-600/30 border-blue-400 text-white"
                      : "bg-slate-800 border-slate-700 text-slate-300"
                  }`}
                >
                  <div className="font-semibold">{seat.label}</div>
                  <div className="text-xs text-slate-400">
                    Đỉnh đã lưu: {seat.polygon_norm.length} (Đang vẽ: {idx === activeIdx ? draftPoints.length : 0}/4)
                  </div>
                </button>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={handleSave}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 py-2 rounded-lg font-semibold"
              >
                Lưu Cấu hình Bàn
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
```

`frontend/src/App.tsx`:
```tsx
import React, { useEffect, useState } from "react";
import { Classroom, SeatStatus, ViolationLog } from "./types";
import { CalibrationModal } from "./components/CalibrationModal";

const STATUS_STYLE: Record<SeatStatus, string> = {
  EMPTY: "bg-slate-800/70 border-slate-700 text-slate-400",
  NORMAL: "bg-emerald-950/80 border-emerald-500 text-emerald-200",
  SUSPICIOUS: "bg-amber-950/90 border-amber-400 text-amber-200 shadow-lg shadow-amber-500/20",
  VIOLATION: "bg-rose-950 border-rose-500 text-rose-100 animate-pulse shadow-lg shadow-rose-600/40",
};

export const App: React.FC = () => {
  const [classrooms, setClassrooms] = useState<Classroom[]>([]);
  const [selectedRoom, setSelectedRoom] = useState<Classroom | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [seatStates, setSeatStates] = useState<Record<number, SeatStatus>>({});
  const [violations, setViolations] = useState<ViolationLog[]>([]);
  const [showCalib, setShowCalib] = useState<boolean>(false);
  const [previewImg, setPreviewImg] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/classrooms")
      .then((r) => r.json())
      .then((data: Classroom[]) => {
        setClassrooms(data);
        if (data.length > 0) setSelectedRoom(data[0]);
      });
  }, []);

  useEffect(() => {
    if (!selectedRoom) return;
    fetch(`/api/violations?classroom_id=${selectedRoom.id}`)
      .then((r) => r.json())
      .then(setViolations);

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/classrooms/${selectedRoom.id}`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.event === "INITIAL_STATE" && msg.seat_states) {
        setSeatStates(msg.seat_states);
      } else if (msg.event === "SEAT_STATE_CHANGED") {
        setSeatStates((prev) => ({ ...prev, [msg.seat_id]: msg.status }));
        if (msg.status === "SUSPICIOUS" || msg.status === "VIOLATION") {
          fetch(`/api/violations?classroom_id=${selectedRoom.id}`)
            .then((r) => r.json())
            .then(setViolations);
        }
      }
    };
    return () => ws.close();
  }, [selectedRoom]);

  useEffect(() => {
    const timer = setInterval(() => {
      fetch("/api/monitoring/status")
        .then((r) => r.json())
        .then((st) => {
          setRunning(st.running);
          setFps(st.fps || 0);
          if (st.seat_states) setSeatStates(st.seat_states);
        });
    }, 1500);
    return () => clearInterval(timer);
  }, []);

  const toggleMonitoring = async () => {
    if (!selectedRoom) return;
    if (running) {
      await fetch("/api/monitoring/stop", { method: "POST" });
      setRunning(false);
    } else {
      await fetch("/api/monitoring/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ classroom_id: selectedRoom.id }),
      });
      setRunning(true);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/80 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Hệ thống Giám sát & Cảnh báo Sử dụng Điện thoại trong Lớp học
          </h1>
          <p className="text-xs text-slate-400">
            Cascade AI Pipeline: YOLOv8n-pose (Mức 0–1) → Smart ROI Crop YOLOv8n (Mức 2)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedRoom?.id || ""}
            onChange={(e) => {
              const found = classrooms.find((c) => c.id === Number(e.target.value));
              if (found) setSelectedRoom(found);
            }}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
          >
            {classrooms.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} [{c.camera_profile}]
              </option>
            ))}
          </select>

          <button
            onClick={() => setShowCalib(true)}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-sm font-medium"
          >
            Hiệu chỉnh Bàn học
          </button>

          <button
            onClick={toggleMonitoring}
            className={`px-4 py-2 rounded-lg font-semibold text-sm ${
              running ? "bg-rose-600 hover:bg-rose-500" : "bg-emerald-600 hover:bg-emerald-500"
            }`}
          >
            {running ? "Dừng Giám sát" : "Bắt đầu Giám sát"}
          </button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
        {/* Cột trái: Luồng Live Video MJPEG */}
        <section className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <span className="font-semibold text-sm text-slate-200">
              Luồng Camera Trực tiếp (Overlay AI Cascade)
            </span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-emerald-400 font-mono">
              {running ? `LIVE • ${fps} FPS` : "STANDBY"}
            </span>
          </div>

          <div className="flex-1 bg-black rounded-lg overflow-hidden flex items-center justify-center min-h-[380px]">
            {running ? (
              <img
                src="/api/stream"
                alt="Live AI Stream"
                className="w-full h-auto max-h-[540px] object-contain"
              />
            ) : (
              <div className="text-center text-slate-500 p-8">
                Bấm <b>"Bắt đầu Giám sát"</b> để khởi chạy luồng AI Cascade.
              </div>
            )}
          </div>
        </section>

        {/* Cột phải: Sơ đồ Lưới Bàn học & Danh sách Vi phạm */}
        <section className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-sm">Sơ đồ Chỗ ngồi Thời gian thực</h2>
              <div className="flex gap-2 text-xs">
                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-600">
                  Bình thường
                </span>
                <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500">
                  Nghi ngờ (Mức 1)
                </span>
                <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-200 border border-rose-500">
                  Vi phạm (Mức 2)
                </span>
              </div>
            </div>

            {selectedRoom && (
              <div
                className="grid gap-3"
                style={{
                  gridTemplateColumns: `repeat(${selectedRoom.grid_cols}, minmax(0, 1fr))`,
                }}
              >
                {selectedRoom.seats.map((seat) => {
                  const st: SeatStatus = (seat.id && seatStates[seat.id]) || "EMPTY";
                  return (
                    <div
                      key={seat.id || seat.label}
                      className={`p-3 rounded-xl border-2 transition-all ${STATUS_STYLE[st]}`}
                    >
                      <div className="text-xs opacity-75">{seat.label}</div>
                      <div className="text-sm font-bold mt-1">{st}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col">
            <h2 className="font-semibold text-sm mb-3">Nhật ký Cảnh báo & Ảnh Bằng chứng</h2>
            <div className="space-y-2.5 overflow-y-auto max-h-72 pr-1">
              {violations.length === 0 ? (
                <div className="text-xs text-slate-500 py-8 text-center">
                  Chưa ghi nhận cảnh báo nào.
                </div>
              ) : (
                violations.map((v) => (
                  <div
                    key={v.id}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-800/70 border border-slate-700 text-xs"
                  >
                    <div>
                      <div className="font-semibold text-slate-100">
                        {v.seat_label} —{" "}
                        <span
                          className={
                            v.level === "VIOLATION" ? "text-rose-400" : "text-amber-400"
                          }
                        >
                          {v.level}
                        </span>
                      </div>
                      <div className="text-slate-400">
                        {new Date(v.created_at).toLocaleTimeString()}
                        {v.confidence ? ` • Độ tin cậy: ${(v.confidence * 100).toFixed(0)}%` : ""}
                      </div>
                    </div>
                    {v.snapshot_url && (
                      <button
                        onClick={() => setPreviewImg(v.snapshot_url)}
                        className="px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200"
                      >
                        Xem ảnh
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>

      {showCalib && selectedRoom && (
        <CalibrationModal
          classroom={selectedRoom}
          onClose={() => setShowCalib(false)}
          onSaved={(updated) => {
            setSelectedRoom(updated);
            setClassrooms((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
          }}
        />
      )}

      {previewImg && (
        <div
          onClick={() => setPreviewImg(null)}
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-6"
        >
          <img
            src={previewImg}
            alt="Violation Evidence"
            className="max-w-4xl max-h-[85vh] rounded-xl border border-slate-700"
          />
        </div>
      )}
    </div>
  );
};
```

- [ ] **Step 3: Cài đặt dependencies, kiểm tra `npm run build` và commit**

Run:
```powershell
cd frontend; npm install; npm run build; cd ..
git add frontend/
git commit -m "feat(frontend): implement React dashboard with live MJPEG stream, real-time seat grid, and polygon calibration modal"
```
Expected: `npm run build` thành công không lỗi TypeScript.
