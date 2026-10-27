from __future__ import annotations

import math
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
