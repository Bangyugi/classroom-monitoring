from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.entities import ViolationLog
from backend.app.schemas.payloads import ViolationLogSchema

router = APIRouter(prefix="/api/violations", tags=["violations"])


@router.get("", response_model=list[ViolationLogSchema])
def list_violations(
    classroom_id: int | None = None,
    limit: int = 30,
    db: Session = Depends(get_db),
):
    q = db.query(ViolationLog)
    if classroom_id is not None:
        q = q.filter(ViolationLog.classroom_id == classroom_id)
    return q.order_by(ViolationLog.id.desc()).limit(limit).all()

