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
    kpts[6] = (550, 300, 0.9)
    kpts[9] = (490, 400, 0.9)
    kpts[10] = (510, 400, 0.9)
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

    update_level2_verification(state, det, conf_thres=0.60, required_hits=2)
    assert state.status == SeatStatus.SUSPICIOUS
    assert state.phone_hits == 1

    update_level2_verification(state, det, conf_thres=0.60, required_hits=2)
    assert state.status == SeatStatus.VIOLATION
    assert state.last_confidence == 0.75

