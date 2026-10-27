from __future__ import annotations

import re
import shutil
from pathlib import Path
import yaml

ROOM_PATTERN = re.compile(r"^(Class_\d+)_")


def extract_room_prefix(filename: str) -> str:
    match = ROOM_PATTERN.match(filename)
    return match.group(1) if match else "Unknown"


def filter_phone_labels(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] == "0":
            result.append("0 " + " ".join(parts[1:]) + "\n")
    return result


def repartition_cctv_dataset(
    raw_dir: Path,
    out_dir: Path,
    val_rooms: tuple[str, ...] = ("Class_4",),
) -> dict[str, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}

    for src_split in ("train", "valid", "test"):
        img_dir = raw_dir / src_split / "images"
        lbl_dir = raw_dir / src_split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            room = extract_room_prefix(img_path.name)
            dst_split = "val" if room in val_rooms else "train"
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            raw_lines = lbl_path.read_text(encoding="utf-8").splitlines() if lbl_path.exists() else []
            phone_lines = filter_phone_labels(raw_lines)

            shutil.copy2(img_path, out_dir / dst_split / "images" / img_path.name)
            (out_dir / dst_split / "labels" / f"{img_path.stem}.txt").write_text(
                "".join(phone_lines), encoding="utf-8"
            )
            counts[f"{dst_split}_images"] += 1
            counts[f"{dst_split}_boxes"] += len(phone_lines)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["cell_phone"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    stats = repartition_cctv_dataset(
        raw_dir=root / "datasets" / "raw" / "cctv_phone",
        out_dir=root / "datasets" / "processed" / "cctv_by_room",
    )
    print("Repartitioned CCTV dataset by room:", stats)
