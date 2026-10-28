from __future__ import annotations

import os
import socket
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = ROOT_DIR / "storage"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
MODELS_DIR = STORAGE_DIR / "models"

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_default_database_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    try:
        with socket.create_connection(("localhost", 5432), timeout=0.5):
            return "postgresql+psycopg://postgres:postgres@localhost:5432/classroom_monitoring"
    except OSError:
        return f"sqlite:///{(STORAGE_DIR / 'classroom.db').as_posix()}"


DATABASE_URL = _resolve_default_database_url()
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolov8n-pose.pt")
PHONE_WEIGHTS = os.getenv(
    "PHONE_WEIGHTS",
    str(MODELS_DIR / "yolov8n_cctv.pt") if (MODELS_DIR / "yolov8n_cctv.pt").exists() else "yolov8n.pt",
)

