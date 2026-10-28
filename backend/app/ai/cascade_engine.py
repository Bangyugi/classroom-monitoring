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
from backend.app.ai.seat_mapper import assign_persons_to_seats, shoulder_metrics
from backend.app.ai.pose_analyzer import PoseTimeTracker, evaluate_posture
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
    is_escalation: bool = False


def compute_expanded_person_roi(
    person: PersonDetection, frame_w: int, frame_h: int, scale: float = 1.3
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = person.bbox_xyxy
    cx = (bx1 + bx2) * 0.5
    cy = (by1 + by2) * 0.5
    w = max(180.0, (bx2 - bx1) * scale)
    h = max(180.0, (by2 - by1) * scale)
    x1 = int(max(0, cx - w * 0.5))
    y1 = int(max(0, cy - h * 0.5))
    x2 = int(min(frame_w, cx + w * 0.5))
    y2 = int(min(frame_h, cy + h * 0.5))
    return x1, y1, x2, y2


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
        active_ids = {p.track_id for p in persons}

        seat_best_track: dict[int, TrackRuntimeState] = {}

        primary_track_id: int | None = None
        if len(self.seats) == 1 and persons:
            primary_track_id = max(
                persons,
                key=lambda p: (p.bbox_xyxy[2] - p.bbox_xyxy[0]) * (p.bbox_xyxy[3] - p.bbox_xyxy[1]),
            ).track_id

        for person in persons:
            seat_id = track_to_seat.get(person.track_id)
            roi = compute_smart_roi_crop(person, self.profile, w, h)
            pre_phone_boxes: list[PhoneDetectionBox] | None = None
            visual_phone_hit = False

            is_posture_suspect, _ = evaluate_posture(person, self.profile)
            kpts = person.keypoints_xyc
            both_wrists_occluded = (kpts[9][2] < 0.22) and (kpts[10][2] < 0.22)

            # Trong chế độ 1x1 Webcam, chỉ quét sớm trên người chính diện duy nhất khi:
            # - Tư thế đã nghi ngờ (is_posture_suspect), HOẶC cả 2 cổ tay đều khuất (có thể do điện thoại giơ ngang mặt che tay)
            if (
                len(self.seats) == 1
                and person.track_id == primary_track_id
                and self.phone_detect_fn is not None
                and (is_posture_suspect or both_wrists_occluded)
            ):
                pre_phone_boxes = self.phone_detect_fn(frame, roi, self.phone_conf_thres)
                valid_pre = [d for d in (pre_phone_boxes or []) if d.confidence >= self.phone_conf_thres]
                if valid_pre:
                    (_, sh_y), sh_w, has_sh = shoulder_metrics(person)
                    best_pre = max(valid_pre, key=lambda d: d.confidence)
                    phone_cy = (best_pre.bbox_xyxy[1] + best_pre.bbox_xyxy[3]) * 0.5
                    phone_at_face_level = has_sh and (phone_cy <= sh_y + 0.05 * sh_w)
                    if is_posture_suspect or phone_at_face_level:
                        visual_phone_hit = True
                        pre_phone_boxes = valid_pre
                    else:
                        pre_phone_boxes = []

            state = self.pose_tracker.update_track(
                person, seat_id=seat_id, dt=dt, visual_phone_detected=visual_phone_hit
            )
            state.last_roi_box = roi

            if visual_phone_hit and pre_phone_boxes:
                best_pre = max(pre_phone_boxes, key=lambda d: d.confidence)
                state.last_confidence = best_pre.confidence
                state.last_phone_box = best_pre.bbox_xyxy

            if state.status in (SeatStatus.SUSPICIOUS, SeatStatus.VIOLATION) and self.phone_detect_fn is not None:
                if pre_phone_boxes is not None:
                    phone_boxes = pre_phone_boxes
                else:
                    phone_boxes = self.phone_detect_fn(frame, roi, self.phone_conf_thres)
                    if not phone_boxes:
                        fb_roi = compute_expanded_person_roi(person, w, h, scale=1.3)
                        if fb_roi != roi:
                            phone_boxes = self.phone_detect_fn(frame, fb_roi, self.phone_conf_thres)
                req_hits = 1 if len(self.seats) == 1 else 2
                update_level2_verification(state, phone_boxes, conf_thres=self.phone_conf_thres, required_hits=req_hits)

            if seat_id is not None:
                prev_best = seat_best_track.get(seat_id)
                if prev_best is None or (
                    STATUS_PRIORITY[state.status],
                    state.suspect_seconds,
                ) > (
                    STATUS_PRIORITY[prev_best.status],
                    prev_best.suspect_seconds,
                ):
                    seat_best_track[seat_id] = state

        self.pose_tracker.prune_absent_tracks(active_ids, dt=dt, max_absent_sec=30.0)

        events: list[SeatStateEvent] = []
        seat_map = {s.seat_id: s for s in self.seats}

        for seat_id, seat_obj in seat_map.items():
            best_track = seat_best_track.get(seat_id)
            new_status = best_track.status if best_track is not None else SeatStatus.EMPTY
            old_status = self.seat_statuses.get(seat_id, SeatStatus.EMPTY)
            if new_status != old_status:
                is_esc = STATUS_PRIORITY[new_status] > STATUS_PRIORITY[old_status]
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
                        is_escalation=is_esc,
                    )
                )

        return events
