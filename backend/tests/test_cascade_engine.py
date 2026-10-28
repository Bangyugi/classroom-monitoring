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


def test_cascade_single_seat_webcam_visual_phone_assist():
    # Trong chế độ 1x1 Webcam, ngay cả khi tay bị điện thoại che khuất hoàn toàn (pose trả về False),
    # nếu YOLOv8 phát hiện thấy điện thoại trực tiếp trong vùng ROI thì vẫn tích lũy thời gian và báo VIOLATION
    seats = [
        SeatPolygon(
            seat_id=1,
            label="Vị trí Webcam Cá nhân",
            polygon_norm=[(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)],
        )
    ]
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (500, 220, 0.95)  # Đầu hoàn toàn thẳng (r_front = 0.80)
    kpts[5] = (450, 300, 0.95)
    kpts[6] = (550, 300, 0.95)  # Không thấy cổ tay/khuỷu tay
    upright_person = PersonDetection(track_id=9, bbox_xyxy=(400, 150, 600, 480), keypoints_xyc=kpts)

    def fake_phone_detector(frame, roi_xyxy, conf_thres):
        return [PhoneDetectionBox(bbox_xyxy=(420, 260, 470, 340), confidence=0.58)]

    engine = CascadeEngine(
        seats=seats,
        profile=CameraProfile.FRONT_VIEW,
        hold_seconds=2.0,
        phone_conf_thres=0.35,
        phone_detect_fn=fake_phone_detector,
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    engine.process_detections(frame, [upright_person], dt=1.2)
    assert engine.pose_tracker.tracks[9].suspect_seconds >= 1.2
    assert engine.pose_tracker.tracks[9].last_phone_box is not None

    events = engine.process_detections(frame, [upright_person], dt=1.0)
    assert engine.seat_statuses[1] == SeatStatus.VIOLATION
    assert any(ev.status == SeatStatus.VIOLATION for ev in events)

