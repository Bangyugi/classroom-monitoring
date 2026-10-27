from __future__ import annotations

import shutil
from pathlib import Path
from ultralytics import YOLO


def finetune_cctv_phone_detector(epochs: int = 12, imgsz: int = 640) -> Path:
    root = Path(__file__).resolve().parents[2]
    data_yaml = root / "datasets" / "processed" / "cctv_by_room" / "data.yaml"
    out_weights = root / "storage" / "models" / "yolov8n_cctv.pt"
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=str(root / "runs"),
        name="cctv_phone_yolov8n",
        exist_ok=True,
        verbose=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, out_weights)
    return out_weights


if __name__ == "__main__":
    saved = finetune_cctv_phone_detector()
    print(f"Saved fine-tuned weights to: {saved}")

