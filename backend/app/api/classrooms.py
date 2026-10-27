from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.entities import Classroom, Seat
from backend.app.schemas.payloads import (
    ClassroomResponseSchema,
    ClassroomUpdateSchema,
    SeatsBulkUpdateSchema,
)
from backend.app.ai.worker import worker_instance

router = APIRouter(prefix="/api/classrooms", tags=["classrooms"])


def seed_default_classrooms(db: Session) -> None:
    if db.query(Classroom).count() > 0:
        return

    room1 = Classroom(
        name="Phòng Demo Webcam (FRONT_VIEW)",
        camera_source="0",
        camera_profile="FRONT_VIEW",
        grid_rows=1,
        grid_cols=1,
        frame_skip=2,
        suspicious_hold_sec=2.0,
        phone_conf_thres=0.35,
    )
    db.add(room1)
    db.flush()

    default_grid_r1 = [
        ("Vị trí Webcam Cá nhân", 0, 0, [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]),
    ]
    for label, r, c, poly in default_grid_r1:
        db.add(Seat(classroom_id=room1.id, label=label, row_idx=r, col_idx=c, polygon_norm=poly))

    room2 = Classroom(
        name="Phòng A1-310 CCTV (REAR_VIEW)",
        camera_source="datasets/processed/cctv_by_room/val/images",
        camera_profile="REAR_VIEW",
        grid_rows=3,
        grid_cols=3,
        frame_skip=1,
        suspicious_hold_sec=2.0,
        phone_conf_thres=0.45,
    )
    db.add(room2)
    db.flush()

    for r in range(3):
        for c in range(3):
            x1, x2 = round(0.05 + c * 0.30, 2), round(0.33 + c * 0.30, 2)
            y1, y2 = round(0.25 + r * 0.23, 2), round(0.46 + r * 0.23, 2)
            db.add(
                Seat(
                    classroom_id=room2.id,
                    label=f"Dãy {r+1} - Bàn {c+1}",
                    row_idx=r,
                    col_idx=c,
                    polygon_norm=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                )
            )

    db.commit()


@router.get("", response_model=list[ClassroomResponseSchema])
def list_classrooms(db: Session = Depends(get_db)):
    seed_default_classrooms(db)
    return db.query(Classroom).order_by(Classroom.id).all()


@router.post("", response_model=ClassroomResponseSchema)
def create_classroom(payload: ClassroomUpdateSchema, db: Session = Depends(get_db)):
    room = Classroom(
        name=payload.name or "Phòng học mới",
        camera_source=payload.camera_source or "0",
        camera_profile=payload.camera_profile or "FRONT_VIEW",
        grid_rows=payload.grid_rows or 2,
        grid_cols=payload.grid_cols or 2,
        frame_skip=payload.frame_skip or 3,
        suspicious_hold_sec=payload.suspicious_hold_sec or 3.0,
        phone_conf_thres=payload.phone_conf_thres or 0.55,
    )
    db.add(room)
    db.commit()
    db.refresh(room)
    return room


@router.put("/{classroom_id}", response_model=ClassroomResponseSchema)
def update_classroom(
    classroom_id: int, payload: ClassroomUpdateSchema, db: Session = Depends(get_db)
):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(room, k, v)
    db.commit()
    db.refresh(room)
    worker_instance.reload_classroom_config(classroom_id)
    return room


@router.put("/{classroom_id}/seats", response_model=ClassroomResponseSchema)
def update_classroom_seats(
    classroom_id: int, payload: SeatsBulkUpdateSchema, db: Session = Depends(get_db)
):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")

    db.query(Seat).filter(Seat.classroom_id == classroom_id).delete()
    for item in payload.seats:
        db.add(
            Seat(
                classroom_id=classroom_id,
                label=item.label,
                row_idx=item.row_idx,
                col_idx=item.col_idx,
                polygon_norm=item.polygon_norm,
            )
        )
    db.commit()
    db.refresh(room)
    worker_instance.reload_classroom_config(classroom_id)
    return room


@router.get("/{classroom_id}/snapshot")
def get_classroom_snapshot(classroom_id: int, db: Session = Depends(get_db)):
    room = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Classroom not found")
    frame = worker_instance.get_snapshot_frame(classroom_id, room.camera_source)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode snapshot")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
