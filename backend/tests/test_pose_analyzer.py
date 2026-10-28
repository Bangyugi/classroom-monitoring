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


def test_front_view_chest_level_phone_with_tilted_head_and_low_conf_wrist():
    # Thực tế từ webcam: người dùng cầm điện thoại ngang ngực phải (y=252.4, sh_y=192.6 -> +0.33*sh_w),
    # đầu hơi cúi & nghiêng nhìn sang điện thoại (r_front=0.438), cổ tay bị điện thoại che một phần (conf=0.22)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (305.6, 113.4, 0.99)   # nose
    kpts[1] = (330.0, 87.1, 0.99)    # l_eye
    kpts[2] = (292.6, 88.1, 0.98)    # r_eye
    kpts[5] = (429.4, 195.1, 0.74)   # l_sh
    kpts[6] = (248.4, 190.1, 0.95)   # r_sh (sh_w = 181.0, sh_y = 192.6)
    kpts[7] = (453.1, 279.0, 0.04)   # l_el
    kpts[8] = (194.9, 259.6, 0.31)   # r_el
    kpts[9] = (413.8, 253.5, 0.11)   # l_wr
    kpts[10] = (167.2, 252.4, 0.22)  # r_wr (partially occluded by phone)
    person = PersonDetection(track_id=11, bbox_xyxy=(150.0, 40.0, 470.0, 280.0), keypoints_xyc=kpts)
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_front_view_closeup_webcam_bowed_head_hands_out_of_frame():
    # Thực tế từ webcam: ngồi gần màn hình, cúi đầu nhìn điện thoại dưới mặt bàn (r_front = 0.35),
    # cả khuỷu tay và cổ tay đều nằm khuất dưới cạnh dưới khung hình webcam
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (320.0, 188.0, 0.95)   # nose
    kpts[5] = (410.0, 250.0, 0.90)   # l_sh
    kpts[6] = (230.0, 250.0, 0.90)   # r_sh (sh_w = 180, sh_y = 250, r_front = 62/180 = 0.344)
    person = PersonDetection(track_id=12, bbox_xyxy=(160.0, 80.0, 480.0, 350.0), keypoints_xyc=kpts)
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert is_suspect


def test_front_view_closeup_looking_straight_at_screen_is_not_suspect():
    # Thực tế từ webcam (FALSE_1 & FALSE_2): ngồi gần màn hình, mắt nhìn thẳng vào màn hình máy tính,
    # hai mắt cao hơn hai tai (eye_y < ear_y), mặt cân đối (neck_tilt nhỏ, yaw nhỏ) -> KHÔNG PHẢI nghi ngờ
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (248.9, 183.5, 0.99)   # nose
    kpts[1] = (267.9, 163.2, 0.98)   # l_eye
    kpts[2] = (231.2, 160.3, 0.99)   # r_eye
    kpts[3] = (289.6, 165.8, 0.87)   # l_ear
    kpts[4] = (206.0, 161.0, 0.87)   # r_ear
    kpts[5] = (328.2, 263.5, 0.86)   # l_sh
    kpts[6] = (169.1, 254.1, 0.97)   # r_sh (sh_y = 258.8, sh_w = 159.4, r_front = 0.473)
    kpts[8] = (131.6, 323.4, 0.29)   # r_el
    kpts[10] = (113.0, 312.0, 0.21)  # r_wr low conf
    person = PersonDetection(track_id=1, bbox_xyxy=(80.0, 55.0, 360.0, 328.0), keypoints_xyc=kpts)
    is_suspect, _ = evaluate_posture(person, CameraProfile.FRONT_VIEW)
    assert not is_suspect


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


def test_pose_time_tracker_prunes_stale_tracks():
    tracker = PoseTimeTracker(profile=CameraProfile.FRONT_VIEW, hold_seconds=3.0)
    p1 = build_front_person(nose_y=235, l_wrist=(480, 380, 0.9), r_wrist=(520, 380, 0.9))
    tracker.update_track(p1, seat_id=1, dt=0.5)
    assert 1 in tracker.tracks

    tracker.prune_absent_tracks(active_track_ids=set(), dt=35.0, max_absent_sec=30.0)
    assert 1 not in tracker.tracks

