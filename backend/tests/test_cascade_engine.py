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
    kpts[0] = (300, 280, 0.9)
    kpts[5] = (250, 300, 0.9)
    kpts[6] = (350, 300, 0.9)
    kpts[9] = (285, 360, 0.9)
    kpts[10] = (315, 360, 0.9)
    return PersonDetection(track_id=track_id, bbox_xyxy=(220, 200, 380, 480), keypoints_xyc=kpts)


def test_cascade_only_calls_phone_detector_after_level1_triggers():
    seats = [
        SeatPolygon(
            seat_id=1,
            label="Dãy 1 - Bàn 1",
            polygon_norm=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
        ),
        SeatPolygon(
            seat_id=2,
            label="Dãy 1 - Bàn 2",
            polygon_norm=[(0.55, 0.1), (0.95, 0.1), (0.95, 0.5), (0.55, 0.5)],
        ),
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

    events1 = engine.process_detections(frame, [person], dt=1.5)
    assert len(phone_calls) == 0
    assert engine.seat_statuses[1] == SeatStatus.NORMAL

    events2 = engine.process_detections(frame, [person], dt=1.7)
    assert len(phone_calls) == 1
    assert engine.seat_statuses[1] == SeatStatus.SUSPICIOUS

    events3 = engine.process_detections(frame, [person], dt=0.8)
    assert len(phone_calls) == 2
    assert engine.seat_statuses[1] == SeatStatus.VIOLATION
    assert any(ev.status == SeatStatus.VIOLATION and ev.seat_id == 1 for ev in events3)

