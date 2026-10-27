import numpy as np
from backend.app.ai.types import CameraProfile, PersonDetection, SeatStatus
from backend.app.ai.pose_analyzer import evaluate_posture, PoseTimeTracker


def build_front_person(
    nose_y: float,
    l_wrist: tuple[float, float, float],
    r_wrist: tuple[float, float, float],
) -> PersonDetection:
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
    person = build_front_person(nose_y=235, l_wrist=(480, 380, 0.9), r_wrist=(520, 380, 0.9))
    is_suspect, ratio = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert not is_suspect
    assert ratio > 0.5


def test_front_view_bowed_head_with_close_wrists_is_suspect():
    person = build_front_person(nose_y=280, l_wrist=(480, 360, 0.9), r_wrist=(520, 360, 0.9))
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_front_view_under_desk_occluded_wrists_is_suspect():
    person = build_front_person(nose_y=285, l_wrist=(0, 0, 0.05), r_wrist=(0, 0, 0.05))
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_front_view_holding_phone_up_near_chest_or_face_is_suspect():
    # Ngồi trước webcam, đầu thẳng (nose_y=210, vai y=350, rộng vai=160) nhưng 1 tay giơ điện thoại lên ngang ngực/vai (y=290)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (320, 210, 0.9)   # nose
    kpts[5] = (240, 350, 0.9)   # left shoulder
    kpts[6] = (400, 350, 0.9)   # right shoulder (sh_w = 160, mid_y = 350)
    kpts[7] = (210, 430, 0.8)   # elbow
    kpts[9] = (230, 290, 0.9)   # wrist raised holding phone near shoulder/face level
    person = PersonDetection(track_id=1, bbox_xyxy=(150, 120, 480, 480), keypoints_xyc=kpts)
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_rear_view_head_drop_and_arms_inward_is_suspect():
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = (450, 300, 0.9)
    kpts[6] = (550, 300, 0.9)
    kpts[7] = (470, 360, 0.8)
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

