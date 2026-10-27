from __future__ import annotations

import math
from backend.app.ai.types import CameraProfile, PersonDetection, SeatStatus, TrackRuntimeState
from backend.app.ai.seat_mapper import KP_CONF_MIN, shoulder_metrics


def evaluate_front_view(person: PersonDetection) -> tuple[bool, float]:
    (sh_x, sh_y), sh_w, has_shoulders = shoulder_metrics(person)
    if not has_shoulders:
        return False, 1.0

    kpts = person.keypoints_xyc
    if kpts[0][2] >= KP_CONF_MIN:
        head_y = float(kpts[0][1])
    else:
        vis_face = [float(kpts[i][1]) for i in (1, 2, 3, 4) if kpts[i][2] >= KP_CONF_MIN]
        if not vis_face:
            return False, 1.0
        head_y = sum(vis_face) / len(vis_face)

    r_front = (sh_y - head_y) / sh_w
    is_head_down = r_front < 0.42

    l_w, r_w = kpts[9], kpts[10]
    l_el, r_el = kpts[7], kpts[8]
    l_vis = l_w[2] >= KP_CONF_MIN
    r_vis = r_w[2] >= KP_CONF_MIN

    min_sh_x = min(float(kpts[5][0]), float(kpts[6][0]))
    max_sh_x = max(float(kpts[5][0]), float(kpts[6][0]))

    # Trường hợp A: Giơ tay cầm điện thoại ngang ngực / vai / mặt (rất phổ biến trên Webcam & khi giơ máy xem)
    hand_raised_holding = False
    for w in (l_w, r_w):
        if w[2] >= KP_CONF_MIN:
            wx, wy = float(w[0]), float(w[1])
            near_upper_body_x = (min_sh_x - 0.65 * sh_w) <= wx <= (max_sh_x + 0.65 * sh_w)
            at_chest_or_face_y = (head_y - 0.25 * sh_w) <= wy <= (sh_y + 0.28 * sh_w)
            if near_upper_body_x and at_chest_or_face_y:
                hand_raised_holding = True
                break

    if hand_raised_holding:
        return True, float(r_front)

    # Trường hợp B: Cúi đầu xuống thấp + tay chụm hoặc giấu dưới bàn
    hands_suspect = False
    if l_vis and r_vis:
        low_enough = (l_w[1] > sh_y + 0.30 * sh_w) and (r_w[1] > sh_y + 0.30 * sh_w)
        wrist_dist_ratio = math.hypot(float(l_w[0] - r_w[0]), float(l_w[1] - r_w[1])) / sh_w
        hands_suspect = low_enough and (wrist_dist_ratio < 0.95)
    elif l_vis or r_vis:
        w = l_w if l_vis else r_w
        in_torso_band = (min_sh_x - 0.25 * sh_w) <= float(w[0]) <= (max_sh_x + 0.25 * sh_w)
        hands_suspect = (float(w[1]) > sh_y + 0.25 * sh_w) and in_torso_band and (r_front < 0.36)
    else:
        el_down = (l_el[2] >= KP_CONF_MIN and l_el[1] > sh_y + 0.15 * sh_w) or (
            r_el[2] >= KP_CONF_MIN and r_el[1] > sh_y + 0.15 * sh_w
        )
        hands_suspect = (el_down and (r_front < 0.34)) or (r_front < 0.26)

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
        state.absent_seconds = 0.0
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

    def prune_absent_tracks(
        self, active_track_ids: set[int], dt: float, max_absent_sec: float = 30.0
    ) -> None:
        stale_ids: list[int] = []
        for tid, st in self.tracks.items():
            if tid not in active_track_ids:
                st.absent_seconds += dt
                st.suspect_seconds = max(0.0, st.suspect_seconds - 1.5 * dt)
                if st.absent_seconds >= max_absent_sec:
                    stale_ids.append(tid)
        for tid in stale_ids:
            self.tracks.pop(tid, None)
