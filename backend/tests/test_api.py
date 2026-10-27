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

