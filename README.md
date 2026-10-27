# Hệ thống Giám sát và Cảnh báo Sử dụng Điện thoại trong Lớp học

Hệ thống phát hiện hành vi sử dụng điện thoại trong lớp học theo thời gian thực dựa trên **Nhận diện Cảnh báo Phân cấp (Cascade AI Pipeline)**:
- **Mức 0 (Định vị chỗ ngồi):** `YOLOv8n-pose` + `ByteTrack` phát hiện khung xương đa người, tính điểm neo cơ thể và ánh xạ vào đa giác bàn học (`Point-in-Polygon`). Hỗ trợ cả 2 góc đặt camera: `FRONT_VIEW` (phía trước / webcam) và `REAR_VIEW` (phía sau lưng như camera CCTV).
- **Mức 1 (Đánh giá tư thế — Cảnh báo Vàng `SUSPICIOUS`):** Phân tích tỷ lệ gập cổ/rụt đầu và vị trí hai bàn tay kết hợp bộ lọc thời gian **Time-Window Tracking** ($X$ giây có Hysteresis) để chống báo động giả khi học sinh cúi viết bài.
- **Mức 2 (Xác nhận vật thể — Cảnh báo Đỏ `VIOLATION`):** Chỉ kích hoạt khi Mức 1 bật Vàng. Cắt vùng ảnh thông minh (`Smart ROI Crop`) quanh cổ tay/thân trước và đưa vào `YOLOv8n` để phát hiện điện thoại kích thước nhỏ, lưu ảnh chụp bằng chứng và đẩy thông báo thời gian thực qua **WebSocket**.

---

## 1. Cấu trúc Dự án

- `backend/` — Dịch vụ Python **FastAPI** + **SQLAlchemy** (PostgreSQL / SQLite fallback) + **AI Cascade Engine** (`backend/app/ai/`) + **Pytest** (`backend/tests/`).
- `frontend/` — Giao diện **React + TypeScript + Vite + TailwindCSS** (Live MJPEG Video Stream, Sơ đồ lưới bàn học thời gian thực, Nhật ký bằng chứng và Công cụ vẽ đa giác hiệu chỉnh bàn học trên trình duyệt).
- `datasets/` — Chứa dữ liệu gốc (`datasets/raw/stbd08`, `datasets/raw/cctv_phone`) và tập dữ liệu CCTV đã chia lại theo phòng học không rò rỉ (`datasets/processed/cctv_by_room`).
- `docs/` — Đặc tả thiết kế (`docs/superpowers/specs/`), kế hoạch triển khai (`docs/superpowers/plans/`) và phân tích dữ liệu (`docs/data/`).

---

## 2. Hướng dẫn Chạy Nhanh (Quick Start)

### Bước 1: Cài đặt thư viện Python cho Backend & AI
> **Lưu ý:** Để chạy suy luận YOLO thực tế với `ultralytics`, khuyến nghị dùng môi trường ảo Python 3.11 hoặc 3.12 (hoặc cài `pip install ultralytics opencv-python`).

```powershell
python -m pip install -r backend/requirements.txt
```

*(Tùy chọn)* Khởi chạy PostgreSQL bằng Docker (nếu không bật Docker, hệ thống tự động dùng SQLite tại `storage/classroom.db`):
```powershell
docker compose up -d
```

### Bước 2: Khởi chạy Backend FastAPI (Cổng `8000`)
```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 3: Khởi chạy Frontend Dashboard (Cổng `5173`)
Mở một cửa sổ terminal thứ hai:
```powershell
cd frontend
npm.cmd run dev
```
Truy cập trình duyệt tại: **http://localhost:5173**

---

## 3. Kiểm thử & Đánh giá Thực nghiệm

- **Chạy toàn bộ Unit & Integration Tests (16 tests):**
  ```powershell
  python -m pytest backend/tests/ -v
  ```
- **Chia lại bộ dữ liệu CCTV theo phòng học (chống Data Leakage):**
  ```powershell
  python backend/scripts/repartition_cctv.py
  ```
- **Fine-tune YOLOv8n trên tập CCTV (`Class_1..3`) và xuất Báo cáo Thực nghiệm trên `Class_4` & `STBD-08`:**
  ```powershell
  python backend/scripts/finetune_phone_yolo.py
  python backend/scripts/evaluate_pipeline.py
  ```

