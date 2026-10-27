from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class SeatItemSchema(BaseModel):
    id: int | None = None
    label: str
    row_idx: int
    col_idx: int
    polygon_norm: list[list[float]] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ClassroomUpdateSchema(BaseModel):
    name: str | None = None
    camera_source: str | None = None
    camera_profile: str | None = None
    grid_rows: int | None = None
    grid_cols: int | None = None
    frame_skip: int | None = None
    suspicious_hold_sec: float | None = None
    phone_conf_thres: float | None = None


class ClassroomResponseSchema(BaseModel):
    id: int
    name: str
    camera_source: str
    camera_profile: str
    grid_rows: int
    grid_cols: int
    frame_skip: int
    suspicious_hold_sec: float
    phone_conf_thres: float
    seats: list[SeatItemSchema] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class SeatsBulkUpdateSchema(BaseModel):
    seats: list[SeatItemSchema]


class StartMonitoringSchema(BaseModel):
    classroom_id: int


class ViolationLogSchema(BaseModel):
    id: int
    classroom_id: int
    seat_id: int
    seat_label: str
    track_id: int | None
    level: str
    confidence: float | None
    snapshot_url: str | None
    created_at: datetime

    model_config = {"from_attributes": True}

