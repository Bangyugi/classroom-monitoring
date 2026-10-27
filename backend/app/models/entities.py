from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship
from backend.app.core.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    camera_source = Column(String(255), nullable=False, default="0")
    camera_profile = Column(String(20), nullable=False, default="FRONT_VIEW")
    grid_rows = Column(Integer, nullable=False, default=2)
    grid_cols = Column(Integer, nullable=False, default=3)
    frame_skip = Column(Integer, nullable=False, default=3)
    suspicious_hold_sec = Column(Float, nullable=False, default=3.0)
    phone_conf_thres = Column(Float, nullable=False, default=0.55)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    seats = relationship("Seat", back_populates="classroom", cascade="all, delete-orphan")


class Seat(Base):
    __tablename__ = "seats"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(50), nullable=False)
    row_idx = Column(Integer, nullable=False, default=0)
    col_idx = Column(Integer, nullable=False, default=0)
    polygon_norm = Column(JSON, nullable=False, default=list)

    classroom = relationship("Classroom", back_populates="seats")


class ViolationLog(Base):
    __tablename__ = "violation_logs"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id", ondelete="CASCADE"), nullable=False)
    seat_id = Column(Integer, nullable=False)
    seat_label = Column(String(50), nullable=False)
    track_id = Column(Integer, nullable=True)
    level = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    snapshot_url = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

