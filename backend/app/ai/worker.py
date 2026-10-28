from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
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

SNAPSHOT_COOLDOWN_SEC = 10.0


def _natural_frame_sort_key(path: Path) -> tuple[str, int, str]:
    """Sắp xếp ảnh frame video theo đúng thứ tự thời gian (VD: mp4-2 trước mp4-10)."""
    m = re.match(r"^(.*?)_mp4-(\d+)_jpg", path.name)
    if m:
        return (m.group(1), int(m.group(2)), path.name)
    nums = re.findall(r"\d+", path.stem)
    return (path.stem, int(nums[-1]) if nums else 0, path.name)


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
        self.latest_raw_frame: np.ndarray | None = None
        self.engine: CascadeEngine | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pose_model = None
        self._phone_detector = YoloPhoneDetector(weights_path=PHONE_WEIGHTS)
        self._last_snapshot_ts: dict[tuple[int, str], float] = {}

    def _ensure_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO(POSE_WEIGHTS)

    def get_snapshot_frame(self, classroom_id: int, camera_source: str) -> np.ndarray:
        with self._lock:
            if (
                self.running
                and self.classroom_id == classroom_id
                and self.latest_raw_frame is not None
            ):
                return self.latest_raw_frame.copy()
        return read_single_frame(camera_source)

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
            with self._lock:
                self.engine.update_config(
                    seats=seats,
                    profile=CameraProfile(room.camera_profile),
                    hold_seconds=room.suspicious_hold_sec,
                    phone_conf_thres=room.phone_conf_thres,
                )

    def start(self, classroom_id: int):
        self.stop()
        with self._lock:
            self.classroom_id = classroom_id
            self.running = True
            self.fps = 0.0
            self.latest_jpeg = None
            self.latest_raw_frame = None
            self._last_snapshot_ts.clear()
        self._thread = threading.Thread(target=self._run_loop, args=(classroom_id,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        with self._lock:
            self.fps = 0.0
            self.latest_jpeg = None
            self.latest_raw_frame = None

    def get_seat_states(self, classroom_id: int | None = None) -> dict[int, str]:
        with self._lock:
            if not self.engine:
                return {}
            if classroom_id is not None and self.classroom_id != classroom_id:
                return {}
            return {k: v.value for k, v in self.engine.seat_statuses.items()}

    def _iter_frames(self, camera_source: str):
        src = int(camera_source) if camera_source.isdigit() else camera_source
        if isinstance(src, str) and not src.startswith("rtsp://"):
            p = Path(src) if Path(src).is_absolute() else (ROOT_DIR / src)
            if p.is_dir():
                imgs = sorted(p.glob("*.jpg"), key=_natural_frame_sort_key)
                idx = 0
                while self.running and imgs:
                    frame = cv2.imread(str(imgs[idx % len(imgs)]))
                    idx += 1
                    if frame is not None:
                        # Lặp mỗi frame tĩnh 3 nhịp (~0.45s/ảnh gốc) vì ảnh CCTV trong dataset là các frame cắt thưa
                        for _ in range(3):
                            if not self.running:
                                break
                            yield frame
                            time.sleep(0.15)
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
            with self._lock:
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
            with self._lock:
                self.latest_raw_frame = frame
            frame_idx += 1
            if frame_idx % frame_skip == 0:
                now = time.perf_counter()
                dt = min(1.0, max(0.05, now - last_proc_ts))
                last_proc_ts = now
                self.fps = round(1.0 / dt, 1)

                last_persons = self._extract_persons(frame)
                with self._lock:
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
                with self._lock:
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
        now_ts = time.time()
        with SessionLocal() as db:
            for ev in events:
                snapshot_url = None
                if ev.is_escalation and ev.status in (SeatStatus.SUSPICIOUS, SeatStatus.VIOLATION):
                    key = (ev.seat_id, ev.status.value)
                    last_ts = self._last_snapshot_ts.get(key, 0.0)
                    if (now_ts - last_ts) >= SNAPSHOT_COOLDOWN_SEC:
                        self._last_snapshot_ts[key] = now_ts
                        ts_int = int(now_ts * 1000)
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
