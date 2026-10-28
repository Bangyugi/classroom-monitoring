import os
os.environ["DATABASE_URL"] = "sqlite:///./storage/test_classroom.db"

from fastapi.testclient import TestClient
from backend.app.main import app


def test_default_classrooms_seeded_and_seat_calibration_update():
    with TestClient(app) as client:
        res = client.get("/api/classrooms")
        assert res.status_code == 200
        rooms = res.json()
        assert len(rooms) >= 2
        room_id = rooms[0]["id"]

        new_seats = [
            {
                "label": "Dãy 1 - Bàn 1",
                "row_idx": 0,
                "col_idx": 0,
                "polygon_norm": [[0.05, 0.1], [0.45, 0.1], [0.45, 0.8], [0.05, 0.8]],
            },
            {
                "label": "Dãy 1 - Bàn 2",
                "row_idx": 0,
                "col_idx": 1,
                "polygon_norm": [[0.55, 0.1], [0.95, 0.1], [0.95, 0.8], [0.55, 0.8]],
            },
        ]
        put_res = client.put(f"/api/classrooms/{room_id}/seats", json={"seats": new_seats})
        assert put_res.status_code == 200
        updated = put_res.json()
        assert len(updated["seats"]) == 2
        assert updated["seats"][0]["label"] == "Dãy 1 - Bàn 1"


def test_websocket_initial_state():
    with TestClient(app) as client:
        rooms = client.get("/api/classrooms").json()
        room_id = rooms[0]["id"]
        with client.websocket_connect(f"/ws/classrooms/{room_id}") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "INITIAL_STATE"
            assert msg["classroom_id"] == room_id


def test_delete_single_and_all_violations():
    from backend.app.core.config import SNAPSHOTS_DIR
    from backend.app.core.database import SessionLocal
    from backend.app.models.entities import ViolationLog

    snap1 = SNAPSHOTS_DIR / "test_snap_1.jpg"
    snap2 = SNAPSHOTS_DIR / "test_snap_2.jpg"
    snap1.write_bytes(b"fake_jpg_1")
    snap2.write_bytes(b"fake_jpg_2")

    with SessionLocal() as db:
        v1 = ViolationLog(
            classroom_id=1,
            seat_id=1,
            seat_label="Vị trí Webcam Cá nhân",
            track_id=1,
            level="SUSPICIOUS",
            confidence=0.65,
            snapshot_url="/storage/snapshots/test_snap_1.jpg",
        )
        v2 = ViolationLog(
            classroom_id=1,
            seat_id=1,
            seat_label="Vị trí Webcam Cá nhân",
            track_id=1,
            level="VIOLATION",
            confidence=0.88,
            snapshot_url="/storage/snapshots/test_snap_2.jpg",
        )
        db.add_all([v1, v2])
        db.commit()
        v1_id, v2_id = v1.id, v2.id

    with TestClient(app) as client:
        # Xoá 1 bản ghi
        del_one = client.delete(f"/api/violations/{v1_id}")
        assert del_one.status_code == 200
        assert not snap1.exists()

        remaining = client.get("/api/violations?classroom_id=1").json()
        assert any(r["id"] == v2_id for r in remaining)
        assert not any(r["id"] == v1_id for r in remaining)

        # Xoá toàn bộ của phòng 1
        del_all = client.delete("/api/violations?classroom_id=1")
        assert del_all.status_code == 200
        assert del_all.json()["deleted"] >= 1
        assert not snap2.exists()
        assert client.get("/api/violations?classroom_id=1").json() == []

