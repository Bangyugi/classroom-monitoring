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

