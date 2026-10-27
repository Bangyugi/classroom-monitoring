import numpy as np
from backend.app.ai.types import PersonDetection, SeatPolygon
from backend.app.ai.seat_mapper import compute_seat_anchor_norm, assign_persons_to_seats


def make_dummy_person(
    track_id: int,
    l_sh: tuple[float, float, float],
    r_sh: tuple[float, float, float],
    bbox=(100, 100, 300, 400),
) -> PersonDetection:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = l_sh
    kpts[6] = r_sh
    return PersonDetection(track_id=track_id, bbox_xyxy=bbox, keypoints_xyc=kpts)


def test_compute_seat_anchor_uses_shoulders_plus_half_width():
    person = make_dummy_person(1, (400, 300, 0.9), (500, 300, 0.9))
    ax, ay = compute_seat_anchor_norm(person, 1000, 1000)
    assert abs(ax - 0.45) < 1e-4
    assert abs(ay - 0.35) < 1e-4


def test_assign_persons_to_seats_maps_inside_polygon():
    seats = [
        SeatPolygon(
            seat_id=10,
            label="Dãy 1 - Bàn 1",
            polygon_norm=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
        ),
        SeatPolygon(
            seat_id=20,
            label="Dãy 1 - Bàn 2",
            polygon_norm=[(0.55, 0.1), (0.9, 0.1), (0.9, 0.5), (0.55, 0.5)],
        ),
    ]
    p1 = make_dummy_person(101, (250, 200, 0.9), (350, 200, 0.9))
    p2 = make_dummy_person(102, (650, 200, 0.9), (750, 200, 0.9))
    mapping = assign_persons_to_seats([p1, p2], seats, 1000, 1000)
    assert mapping == {101: 10, 102: 20}
