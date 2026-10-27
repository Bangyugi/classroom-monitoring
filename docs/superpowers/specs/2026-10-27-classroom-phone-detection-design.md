# Đặc tả Thiết kế Kỹ thuật: Hệ thống Giám sát và Cảnh báo Sử dụng Điện thoại trong Lớp học

- **Ngày tạo:** 2026-10-27
- **Trạng thái:** Đã thống nhất thiết kế (Chờ duyệt văn bản Spec)
- **Tài liệu liên quan:**
  - [project_brief_classroom_phone_detection_system.md](../../../project_brief_classroom_phone_detection_system.md)
  - [2026-10-27-dataset-analysis.md](../../data/2026-10-27-dataset-analysis.md)

---

## 1. Mục tiêu & Phạm vi Hệ thống

Xây dựng hệ thống phần mềm giám sát tự động thời gian thực từ camera lớp học (Webcam, File video, hoặc Camera IP RTSP) nhằm phát hiện và cảnh báo hành vi sử dụng điện thoại thông qua kiến trúc **Nhận diện Cảnh báo Phân cấp (Cascade Pipeline: Mức 0 → Mức 1 → Mức 2)**:

1. **Giải quyết bài toán góc khuất & giảm báo động giả (False Positive):**
   - **Mức 0 (Định vị chỗ ngồi):** Phát hiện đa người bằng `YOLOv8n-pose`, theo vết bằng `ByteTrack` và ánh xạ vào đa giác bàn học đã hiệu chỉnh.
   - **Mức 1 (Đánh giá tư thế - Cảnh báo Vàng `SUSPICIOUS`):** Phân tích đặc trưng hình học khung xương thích ứng theo góc đặt camera (`FRONT_VIEW` hoặc `REAR_VIEW`) kết hợp cửa sổ thời gian (Time-Window Tracking $X$ giây). Hoạt động ngay cả khi điện thoại bị che khuất dưới ngăn bàn.
   - **Mức 2 (Xác nhận vật thể - Cảnh báo Đỏ `VIOLATION`):** Chỉ kích hoạt khi Mức 1 bật Vàng. Cắt vùng ảnh quanh tay/thân trước (Smart ROI Crop) và đưa vào `YOLOv8n` (COCO `cell phone` kết hợp bản fine-tune trên dữ liệu CCTV lớp học) để xác nhận thiết bị di động kích thước nhỏ (~23×23 px).
2. **Trực quan hóa cho giáo viên & Phục vụ bảo vệ đồ án:**
   - Giao diện Web hiển thị song song **luồng Live Video (MJPEG)** có vẽ lớp phủ AI (khung xương, đa giác bàn, thanh tích lũy thời gian, hộp cảnh báo) và **Sơ đồ lưới bàn học thời gian thực (CSS Grid)** cập nhật qua WebSocket kèm danh sách ảnh chụp bằng chứng.
   - Tích hợp công cụ vẽ đa giác hiệu chỉnh bàn học (Seat Calibration) trực tiếp trên trình duyệt.

---

## 2. Kiến trúc Tổng thể & Cấu trúc Thư mục

### 2.1. Sơ đồ Kiến trúc

```
[Nguồn Video: Webcam / File / RTSP]
                 |
                 v
+-----------------------------------------------------------------------+
| Backend & AI Service (Python 3.11/3.12 + FastAPI)                     |
|                                                                       |
|  +-----------------------------------------------------------------+  |
|  | AI Worker Pipeline (Background Thread/Process)                  |  |
|  |  1. Frame Skipping (xử lý mỗi k frame, mặc định k=3)           |  |
|  |  2. Mức 0: YOLOv8n-pose + ByteTrack + Point-in-Polygon (Bàn)    |  |
|  |  3. Mức 1: Đặc trưng Tư thế (FRONT/REAR) + Time-Window (X giây) |  |
|  |  4. Mức 2: Smart ROI Crop + YOLOv8n Phone Detector              |  |
|  |  5. Overlay Renderer (vẽ khung xương, bàn, cảnh báo lên frame)  |  |
|  +---------+-------------------------------+-----------------------+  |
|            | (State Change + Snapshot)     | (JPEG Frame Buffer)      |
|            v                               v                          |
|  +--------------------+         +----------------------------------+  |
|  | Event Dispatcher   |         | Multipart MJPEG Streamer         |  |
|  | (Lưu DB + Phát WS) |         | (GET /api/stream)                |  |
|  +---------+----------+         +-----------------+----------------+  |
+------------|--------------------------------------|-------------------+
             |                                      |
   +---------+---------+                            |
   |                   |                            |
   v                   v                            v
[PostgreSQL / SQLite]  [WebSocket /ws/classrooms]  [HTTP MJPEG]
                              \                    /
                               v                  v
                       +-----------------------------------+
                       | Frontend Dashboard (React + Vite) |
                       | - Live Video + Sơ đồ Lưới Bàn học |
                       | - Nhật ký Bằng chứng + Hiệu chỉnh |
                       +-----------------------------------+
```

### 2.2. Cấu trúc Thư mục (Monorepo)

```
classroom-monitoring/
├── backend/
│   ├── app/
│   │   ├── main.py                # Khởi tạo FastAPI, CORS, static files, lifespan
│   │   ├── core/
│   │   │   ├── config.py          # Biến môi trường & cấu hình mặc định
│   │   │   └── database.py        # Kết nối SQLModel (PostgreSQL / SQLite fallback)
│   │   ├── models/
│   │   │   └── entities.py        # Bảng Classroom, Seat, ViolationLog
│   │   ├── schemas/
│   │   │   └── payloads.py        # DTO cho REST API và sự kiện WebSocket
│   │   ├── api/
│   │   │   ├── classrooms.py      # CRUD phòng học, cấu hình bàn, lấy snapshot tĩnh
│   │   │   ├── monitoring.py      # Start/Stop phiên giám sát, MJPEG /api/stream
│   │   │   ├── violations.py      # Truy vấn lịch sử vi phạm
│   │   │   └── ws.py              # Quản lý kết nối WebSocket theo classroom_id
│   │   └── ai/
│   │       ├── video_source.py    # Đọc Webcam / File / RTSP có auto-reconnect
│   │       ├── seat_mapper.py     # Mức 0: Tính điểm neo cơ thể & Point-in-Polygon
│   │       ├── pose_analyzer.py   # Mức 1: Đặc trưng FRONT_VIEW / REAR_VIEW + Time-Window
│   │       ├── phone_detector.py  # Mức 2: Smart ROI Crop + YOLOv8n inference
│   │       ├── overlay.py         # Vẽ đa giác bàn, keypoints, trạng thái lên frame
│   │       └── worker.py          # Vòng lặp điều phối Cascade & phát sự kiện
│   ├── scripts/
│   │   ├── repartition_cctv.py    # Chia lại bộ CCTV theo phòng học (chống data leakage)
│   │   ├── finetune_phone_yolo.py # Fine-tune YOLOv8n trên tập CCTV đã chia lại
│   │   └── evaluate_pipeline.py   # Đánh giá Mức 0, Mức 1 (STBD-08), Mức 2 (CCTV) & FPS CPU
│   ├── tests/                     # Unit & integration tests (pytest)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TopControlBar.tsx  # Chọn phòng, nguồn video, profile, ngưỡng, Start/Stop
│   │   │   ├── LiveStreamPanel.tsx# Hiển thị luồng MJPEG /api/stream
│   │   │   ├── SeatGridPanel.tsx  # Lưới CSS Grid trạng thái bàn (Xám/Xanh/Vàng/Đỏ)
│   │   │   ├── ViolationFeed.tsx  # Danh sách cảnh báo + xem ảnh bằng chứng
│   │   │   └── CalibrationModal.tsx# Vẽ đa giác bàn trên ảnh snapshot camera
│   │   ├── hooks/
│   │   │   └── useClassroomWS.ts  # Kết nối & tự động nối lại WebSocket
│   │   ├── types/
│   │   │   └── index.ts           # Định nghĩa kiểu dữ liệu khớp với Backend
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── datasets/                      # (Git ignored) Chứa dữ liệu gốc và dữ liệu đã chia lại
│   ├── raw/
│   │   ├── stbd08/                # Đổi tên từ archive/
│   │   ├── cctv_phone/            # Đổi tên từ Classroom-Cell-Phone-Detection.v1i.yolov8/
│   │   └── stock_phone_unused/    # Đổi tên từ Classroom Phone Detection.v1i.yolov8/
│   └── processed/
│       └── cctv_by_room/          # Tập train (Class 1-3) / test (Class 4) không rò rỉ
├── storage/                       # (Git ignored)
│   ├── snapshots/                 # Ảnh chụp bằng chứng vi phạm
│   └── models/                    # Trọng số yolov8n-pose.pt, yolov8n.pt, yolov8n-cctv.pt
├── docs/
├── docker-compose.yml             # Dịch vụ PostgreSQL 16
└── .gitignore
```

---

## 3. Mô hình Dữ liệu & Giao tiếp API / WebSocket

### 3.1. Cơ sở dữ liệu (PostgreSQL với Fallback SQLite)

Sử dụng `SQLModel` (kết hợp SQLAlchemy + Pydantic). Chuỗi kết nối lấy từ biến môi trường `DATABASE_URL` (mặc định khi dev nhanh không bật Docker: `sqlite:///./storage/classroom.db`; khi bật Docker: `postgresql+psycopg://postgres:postgres@localhost:5432/classroom_monitoring`).

#### Bảng 1: `classrooms`
| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | `INTEGER` (PK) | Khóa chính tự tăng |
| `name` | `VARCHAR(100)` | Tên phòng học (VD: `"Phòng A1-310"`, `"Demo Webcam"`) |
| `camera_source` | `VARCHAR(255)` | `"0"` (webcam), đường dẫn file `.mp4`/thư mục ảnh, hoặc `rtsp://...` |
| `camera_profile` | `VARCHAR(20)` | `"FRONT_VIEW"` hoặc `"REAR_VIEW"` |
| `grid_rows` | `INTEGER` | Số dãy bàn trên lưới hiển thị (mặc định `4`) |
| `grid_cols` | `INTEGER` | Số bàn mỗi dãy trên lưới hiển thị (mặc định `4`) |
| `frame_skip` | `INTEGER` | Xử lý 1 frame mỗi `frame_skip` frame (mặc định `3`) |
| `suspicious_hold_sec` | `FLOAT` | Ngưỡng thời gian $X$ giây giữ tư thế để bật Vàng (mặc định `3.0`) |
| `phone_conf_thres` | `FLOAT` | Ngưỡng tin cậy YOLO để bật Đỏ (mặc định `0.60`) |
| `created_at` | `TIMESTAMP` | Thời điểm tạo |

#### Bảng 2: `seats`
| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | `INTEGER` (PK) | Khóa chính tự tăng |
| `classroom_id` | `INTEGER` (FK) | Tham chiếu `classrooms.id` |
| `label` | `VARCHAR(50)` | Nhãn hiển thị (VD: `"Dãy 2 - Bàn 3"`) |
| `row_idx` | `INTEGER` | Chỉ số hàng trên lưới CSS Grid (`0 .. grid_rows-1`) |
| `col_idx` | `INTEGER` | Chỉ số cột trên lưới CSS Grid (`0 .. grid_cols-1`) |
| `polygon_norm` | `JSON` | Danh sách đỉnh chuẩn hóa `[[x1, y1], [x2, y2], ...]` với $x, y \in [0, 1]$ |

#### Bảng 3: `violation_logs`
| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | `INTEGER` (PK) | Khóa chính tự tăng |
| `classroom_id` | `INTEGER` (FK) | Tham chiếu `classrooms.id` |
| `seat_id` | `INTEGER` (FK) | Tham chiếu `seats.id` |
| `seat_label` | `VARCHAR(50)` | Lưu kèm nhãn bàn tại thời điểm ghi nhận |
| `track_id` | `INTEGER` | ID theo vết của ByteTrack |
| `level` | `VARCHAR(20)` | `"SUSPICIOUS"` hoặc `"VIOLATION"` |
| `confidence` | `FLOAT` (Nullable) | Độ tin cậy `cell phone` của YOLOv8n (với mức `VIOLATION`) |
| `snapshot_url` | `VARCHAR(255)` | Đường dẫn tĩnh tới ảnh bằng chứng (`/storage/snapshots/...`) |
| `created_at` | `TIMESTAMP` | Thời điểm phát sinh sự kiện |

### 3.2. Đặc tả REST API

| Phương thức & Đường dẫn | Chức năng |
|---|---|
| `GET /api/classrooms` | Danh sách phòng học (kèm cấu hình `seats`). Tự khởi tạo 2 phòng mẫu (`Demo Webcam - FRONT_VIEW` và `CCTV Class 1 - REAR_VIEW`) nếu DB trống. |
| `POST /api/classrooms` | Tạo phòng học mới. |
| `PUT /api/classrooms/{id}` | Cập nhật thông số phòng học (`camera_source`, `camera_profile`, `suspicious_hold_sec`, `phone_conf_thres`, kích thước lưới). |
| `PUT /api/classrooms/{id}/seats` | Cập nhật danh sách đa giác bàn học (`polygon_norm`) từ công cụ hiệu chỉnh và thông báo cho AI Worker nạp lại ngay lập tức. |
| `GET /api/classrooms/{id}/snapshot` | Đọc 1 frame JPEG hiện tại từ nguồn camera của phòng (phục vụ màn hình vẽ đa giác bàn học). |
| `POST /api/monitoring/start` | Bắt đầu phiên giám sát cho `classroom_id`. |
| `POST /api/monitoring/stop` | Dừng phiên giám sát hiện tại. |
| `GET /api/monitoring/status` | Trạng thái phiên giám sát hiện tại (đang chạy phòng nào, FPS thực tế, trạng thái hiện tại của tất cả các bàn). |
| `GET /api/stream` | Luồng `multipart/x-mixed-replace; boundary=frame` phát trực tiếp các frame JPEG đã vẽ overlay AI. |
| `GET /api/violations` | Danh sách lịch sử vi phạm (hỗ trợ lọc theo `classroom_id`, `level`, `limit`). |

### 3.3. Đặc tả Gói tin WebSocket (`WS /ws/classrooms/{id}`)

Khi kết nối thành công, Server gửi ngay gói `INITIAL_STATE` chứa trạng thái hiện tại của toàn bộ các bàn. Khi một bàn đổi trạng thái (`EMPTY` ↔ `NORMAL` ↔ `SUSPICIOUS` ↔ `VIOLATION`), Server phát gói `SEAT_STATE_CHANGED`:

```json
{
  "event": "SEAT_STATE_CHANGED",
  "classroom_id": 1,
  "seat_id": 5,
  "seat_label": "Dãy 2 - Bàn 1",
  "status": "VIOLATION",
  "track_id": 14,
  "suspect_seconds": 3.4,
  "confidence": 0.76,
  "snapshot_url": "/storage/snapshots/room1_seat5_1730043300.jpg",
  "timestamp": "2026-10-27T22:35:00+07:00"
}
```

---

## 4. Chi tiết Thuật toán AI Cascade (Mức 0 → Mức 1 → Mức 2)

### 4.1. Chuẩn hóa Điểm Khung xương (COCO 17 Keypoints)
`YOLOv8n-pose` trả về 17 điểm $(x_i, y_i, c_i)$ với $i \in \{0..16\}$:
- `0`: Mũi (`nose`), `1, 2`: Mắt trái/phải (`eyes`), `3, 4`: Tai trái/phải (`ears`)
- `5, 6`: Vai trái/phải (`shoulders`), `7, 8`: Khuỷu tay trái/phải (`elbows`)
- `9, 10`: Cổ tay trái/phải (`wrists`), `11, 12`: Hông trái/phải (`hips`)

Một điểm $i$ được coi là **nhìn thấy (visible)** nếu $c_i \ge 0.30$.
Gọi trung điểm hai vai là $M_{\text{sh}} = \frac{P_5 + P_6}{2}$ và bề rộng vai là:
$$W_{\text{sh}} = \max\left(\|P_5 - P_6\|_2,\; 0.35 \cdot W_{\text{bbox}}\right)$$
*(Dùng $0.35 \cdot W_{\text{bbox}}$ làm giá trị sàn trong trường hợp người ngồi nghiêng hoặc 1 vai bị khuất).*

### 4.2. Mức 0 — Theo vết & Ánh xạ Người vào Bàn học
1. Chạy `pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)`.
2. Với mỗi người có `track_id`, tính **điểm neo chỗ ngồi (Seat Anchor $A$)**:
   - Nếu hai vai nhìn thấy: $A = (x_{M_{\text{sh}}},\; y_{M_{\text{sh}}} + 0.5 \cdot W_{\text{sh}})$ (vùng giữa ngực/bụng trên).
   - Ngược lại: lấy tâm của nửa trên bounding box người $(x_{\text{center}},\; y_{\min} + 0.35 \cdot H_{\text{bbox}})$.
3. Đổi tọa độ $A$ sang hệ chuẩn hóa $[0, 1]$ và kiểm tra `cv2.pointPolygonTest(polygon, A, False) >= 0` để gán người đó vào `seat_id`.
   - Nếu một bàn có nhiều hơn 1 người lọt vào, ưu tiên người có điểm neo gần tâm đa giác bàn nhất; trạng thái của bàn là mức cảnh báo cao nhất của người ngồi tại bàn đó.

### 4.3. Mức 1 — Quy tắc Tư thế theo `Camera Profile` & Time-Window

#### A. Chế độ `FRONT_VIEW` (Camera phía trước / Webcam)
- **Điểm tham chiếu đầu ($P_{\text{head}}$):** Lấy $P_0$ (mũi) nếu nhìn thấy; nếu không thì lấy trung điểm các điểm nhìn thấy trong $\{P_1, P_2, P_3, P_4\}$.
- **Điều kiện 1 — Cúi đầu (`is_head_down`):**
  $$R_{\text{front}} = \frac{y_{M_{\text{sh}}} - y_{P_{\text{head}}}}{W_{\text{sh}}} < \theta_{\text{front}} \quad (\text{mặc định } \theta_{\text{front}} = 0.38)$$
  *(Lưu ý: trục $y$ hướng xuống nên khi ngồi thẳng đầu cao hơn vai, $y_{M_{\text{sh}}} > y_{P_{\text{head}}}$ và $R_{\text{front}} \approx 0.55..0.80$; khi cúi gầm đầu hạ gần ngang vai hoặc thấp hơn vai, $R_{\text{front}}$ giảm nhỏ hơn $0.38$).*
- **Điều kiện 2 — Tay chụm thấp hoặc giấu dưới bàn (`is_hands_suspect`):** Thỏa mãn **1 trong 3** trường hợp:
  1. *Hai cổ tay cùng nhìn thấy:* cả hai nằm thấp dưới vai ($y_9, y_{10} > y_{M_{\text{sh}}} + 0.30 \cdot W_{\text{sh}}$) **và** khoảng cách giữa hai cổ tay hẹp ($\frac{\|P_9 - P_{10}\|_2}{W_{\text{sh}}} < 0.95$).
  2. *Một cổ tay nhìn thấy ở vùng giữa bụng/ngực:* cổ tay đó nằm thấp dưới vai và nằm trong dải ngang giữa hai vai ($\min(x_5, x_6) \le x_w \le \max(x_5, x_6)$), đồng thời đầu cúi sâu ($R_{\text{front}} < 0.30$).
  3. *Góc khuất dưới ngăn bàn:* Cả hai cổ tay đều bị mặt bàn che khuất ($c_9 < 0.30$ và $c_{10} < 0.30$), ít nhất một khuỷu tay nhìn thấy hướng xuống dưới vai, và đầu cúi sâu ($R_{\text{front}} < 0.28$).
- **Kết luận frame Mức 1 (`FRONT_VIEW`):** `is_posture_suspect = is_head_down AND is_hands_suspect`.

#### B. Chế độ `REAR_VIEW` (Camera phía sau lưng — như CCTV)
- **Điều kiện 1 — Đầu rụt/gập về trước (`is_head_drop`):**
  - Lấy đỉnh đầu tham chiếu $y_{\text{top}} = y_{\min}$ của bounding box người (hoặc trung điểm hai tai $P_3, P_4$ nếu nhìn thấy).
  - Khoảng cách dọc từ đường vai lên đỉnh hộp đầu bị thu ngắn so với chiều rộng vai (hoặc so với chiều cao hộp):
    $$R_{\text{rear}} = \frac{y_{M_{\text{sh}}} - y_{\text{top}}}{W_{\text{sh}}} < \theta_{\text{rear}} \quad (\text{mặc định } \theta_{\text{rear}} = 0.65)$$
- **Điều kiện 2 — Tay thu vào trước thân (`is_arms_inward`):**
  - Ít nhất một cổ tay hoặc khuỷu tay nhìn thấy nằm thấp hơn vai và hướng vào vùng giữa thân (hoặc không dang rộng ra ngoài hai bên vai quá $0.25 \cdot W_{\text{sh}}$).
- **Kết luận frame Mức 1 (`REAR_VIEW`):** `is_posture_suspect = is_head_drop AND is_arms_inward`.

#### C. Bộ lọc Thời gian (Time-Window với Hysteresis)
Với mỗi `track_id`, gọi $\Delta t$ là khoảng thời gian thực giữa 2 lần xử lý frame:
- Nếu `is_posture_suspect == True`:
  $$S_{\text{track}} \leftarrow \min\left(S_{\max},\; S_{\text{track}} + \Delta t\right)$$
- Nếu `is_posture_suspect == False`:
  $$S_{\text{track}} \leftarrow \max\left(0,\; S_{\text{track}} - 1.5 \cdot \Delta t\right)$$
- Khi $S_{\text{track}} \ge \text{suspicious\_hold\_sec}$ (mặc định $3.0\text{s}$): Bật cờ **Mức 1 (`SUSPICIOUS` - Vàng)** và kích hoạt **Mức 2**.

### 4.4. Mức 2 — Smart ROI Crop & Xác nhận bằng YOLOv8n
Chỉ thực thi cho những người đang ở trạng thái `SUSPICIOUS` (hoặc đang duy trì `VIOLATION`):
1. **Xác định vùng cắt (Smart ROI Crop):**
   - Nếu nhìn thấy ít nhất 1 cổ tay: lấy tâm vùng cắt tại cổ tay (hoặc trung điểm 2 cổ tay), mở rộng hình vuông cạnh $L = \max(1.6 \cdot W_{\text{sh}}, 160\text{ px})$.
   - Nếu không nhìn thấy cổ tay (hoặc ở `REAR_VIEW` tay đưa ra trước ngực/bàn): cắt vùng mở rộng quanh nửa trên/trước của bounding box người (bao phủ từ ngang vai lên phía mặt bàn trước mặt $0.4 \cdot H_{\text{bbox}}$ và xuống ngang hông).
2. **Suy luận YOLOv8n:**
   - Đưa ảnh crop vào mô hình phát hiện điện thoại (`yolov8n-cctv.pt` nếu đã fine-tune, hoặc `yolov8n.pt` lớp `cell phone`).
   - Để tránh mất điện thoại khi điện thoại đặt ngay trên mặt bàn cạnh người (như trong bộ CCTV), nếu vùng crop quanh tay không phát hiện thấy, cho phép fallback kiểm tra vùng crop mở rộng quanh toàn bộ hộp người đó ($1.3 \times \text{bbox}$).
3. **Cơ chế Xác nhận & Giữ trạng thái Đỏ:**
   - Nếu phát hiện `cell phone` với $\text{conf} \ge \text{phone\_conf\_thres}$ (mặc định `0.60`): tăng bộ đếm xác nhận `phone_hits += 1`.
   - Khi `phone_hits >= 2` (xác nhận ít nhất 2 frame xử lý để lọc nhiễu lóe sáng): Chuyển trạng thái bàn sang **`VIOLATION` (Đỏ)**, chụp ảnh bằng chứng lưu vào `storage/snapshots/` (tối đa 1 ảnh / 10 giây cho mỗi lần vi phạm của bàn đó để không tràn ổ cứng), ghi vào bảng `violation_logs` và phát sự kiện WebSocket.

---

## 5. Kế hoạch Xử lý Dữ liệu & Đánh giá Thực nghiệm

1. **Tổ chức lại thư mục dữ liệu (`datasets/`):**
   - Di chuyển `archive/` → `datasets/raw/stbd08/`
   - Di chuyển `Classroom-Cell-Phone-Detection.v1i.yolov8/` → `datasets/raw/cctv_phone/`
   - Di chuyển `Classroom Phone Detection.v1i.yolov8/` → `datasets/raw/stock_phone_unused/`
   - Xóa file nén `archive.zip` dư thừa sau khi kiểm tra đủ file (nếu người dùng đồng ý) hoặc đưa `*.zip`, `datasets/`, `storage/` vào `.gitignore`.
2. **Chia lại bộ dữ liệu CCTV theo phòng học (`repartition_cctv.py`):**
   - Gom toàn bộ 253 ảnh trong `train/`, `valid/`, `test/` của bộ `cctv_phone` lại, tách theo tiền tố tên file:
     - **Tập Train (`Class_1`, `Class_2`, `Class_3`):** dùng để fine-tune `yolov8n.pt`.
     - **Tập Val/Test độc lập (`Class_4`):** phòng học hoàn toàn chưa từng xuất hiện trong tập Train, dùng để đo Precision, Recall, mAP@0.5 trung thực.
3. **Các bảng số liệu xuất ra cho Báo cáo Đồ án (`evaluate_pipeline.py`):**
   - **Thí nghiệm 1 (Hiệu quả của Smart ROI Crop & Fine-tune ở Mức 2):** So sánh Recall & Precision trên tập Test `Class_4` giữa 3 cấu hình: (a) YOLOv8n gốc chạy toàn ảnh, (b) YOLOv8n gốc chạy trên ROI Crop, (c) YOLOv8n Fine-tuned chạy trên ROI Crop.
   - **Thí nghiệm 2 (Tỷ lệ Báo động giả của Mức 1 trên STBD-08):** Chạy Mức 1 trên tập `val` của `STBD-08` (các lớp `write`, `read`, `listen`) để chứng minh bộ quy tắc tư thế phân biệt được phần lớn học sinh đang ngồi học/viết bài với tư thế nghi ngờ dùng điện thoại.
   - **Thí nghiệm 3 (Hiệu năng CPU):** Đo FPS trung bình và thời gian xử lý mỗi frame (ms) trên CPU AMD Ryzen 7 8745H giữa chế độ chạy YOLO mọi frame vs. chế độ Cascade + Frame Skipping.
