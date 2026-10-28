from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.core.config import SNAPSHOTS_DIR
from backend.app.core.database import get_db
from backend.app.models.entities import ViolationLog
from backend.app.schemas.payloads import ViolationLogSchema

router = APIRouter(prefix="/api/violations", tags=["violations"])


def _remove_snapshot_file(snapshot_url: str | None) -> None:
    if not snapshot_url:
        return
    fname = Path(snapshot_url).name
    if not fname:
        return
    target = (SNAPSHOTS_DIR / fname).resolve()
    if target.parent == SNAPSHOTS_DIR.resolve() and target.is_file():
        target.unlink(missing_ok=True)


@router.get("", response_model=list[ViolationLogSchema])
def list_violations(
    classroom_id: int | None = None,
    level: str | None = None,
    limit: int = 30,
    db: Session = Depends(get_db),
):
    q = db.query(ViolationLog)
    if classroom_id is not None:
        q = q.filter(ViolationLog.classroom_id == classroom_id)
    if level is not None:
        q = q.filter(ViolationLog.level == level)
    return q.order_by(ViolationLog.id.desc()).limit(limit).all()


@router.delete("/{violation_id}")
def delete_violation(violation_id: int, db: Session = Depends(get_db)):
    row = db.query(ViolationLog).filter(ViolationLog.id == violation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Violation log not found")
    _remove_snapshot_file(row.snapshot_url)
    db.delete(row)
    db.commit()
    return {"deleted": 1, "id": violation_id}


@router.delete("")
def clear_violations(classroom_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(ViolationLog)
    if classroom_id is not None:
        q = q.filter(ViolationLog.classroom_id == classroom_id)
    rows = q.all()
    count = len(rows)
    for row in rows:
        _remove_snapshot_file(row.snapshot_url)
        db.delete(row)
    db.commit()
    return {"deleted": count}
