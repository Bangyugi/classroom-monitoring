from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = ROOT_DIR / "storage"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
MODELS_DIR = STORAGE_DIR / "models"

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(STORAGE_DIR / 'classroom.db').as_posix()}")
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolov8n-pose.pt")
PHONE_WEIGHTS = os.getenv(
    "PHONE_WEIGHTS",
    str(MODELS_DIR / "yolov8n_cctv.pt") if (MODELS_DIR / "yolov8n_cctv.pt").exists() else "yolov8n.pt",
)

