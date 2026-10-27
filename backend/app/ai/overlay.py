from __future__ import annotations

import cv2
import numpy as np
from backend.app.ai.types import PersonDetection, SeatPolygon, SeatStatus, TrackRuntimeState
from backend.app.ai.seat_mapper import KP_CONF_MIN

STATUS_BGR = {
    SeatStatus.EMPTY: (160, 160, 160),
    SeatStatus.NORMAL: (80, 200, 80),
    SeatStatus.SUSPICIOUS: (0, 190, 255),
    SeatStatus.VIOLATION: (40, 40, 240),
}

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 5), (0, 6)
]


def render_annotated_frame(
    frame: np.ndarray,
    seats: list[SeatPolygon],
    seat_statuses: dict[int, SeatStatus],
    tracks: dict[int, TrackRuntimeState],
    persons: list[PersonDetection],
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    for seat in seats:
        if len(seat.polygon_norm) < 3:
            continue
        pts = np.array([[int(px * w), int(py * h)] for px, py in seat.polygon_norm], dtype=np.int32)
        st = seat_statuses.get(seat.seat_id, SeatStatus.EMPTY)
        color = STATUS_BGR[st]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            canvas,
            f"{seat.label} [{st.value}]",
            (max(5, cx - 55), cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    for person in persons:
        st_obj = tracks.get(person.track_id)
        st = st_obj.status if st_obj else SeatStatus.NORMAL
        color = STATUS_BGR[st]
        x1, y1, x2, y2 = [int(v) for v in person.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        kpts = person.keypoints_xyc
        for i, j in SKELETON_EDGES:
            if kpts[i][2] >= KP_CONF_MIN and kpts[j][2] >= KP_CONF_MIN:
                p1 = (int(kpts[i][0]), int(kpts[i][1]))
                p2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(canvas, p1, p2, (255, 220, 0), 2)

        if st_obj:
            label = f"ID:{person.track_id} {st_obj.suspect_seconds:.1f}s"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            if st_obj.last_phone_box:
                px1, py1, px2, py2 = [int(v) for v in st_obj.last_phone_box]
                cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 0, 255), 3)
                conf_txt = f"PHONE {(st_obj.last_confidence or 0)*100:.0f}%"
                cv2.putText(canvas, conf_txt, (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return canvas
