# Phân tích các bộ dữ liệu (2026-10-27)

Script thống kê: chỉ dùng thư viện chuẩn Python (đếm nhãn, kích thước ảnh, diện tích box, trùng lặp MD5).

## Tổng quan

| Mã | Thư mục | Ảnh | Box | Lớp | Góc camera | Vai trò đề xuất |
|---|---|---|---|---|---|---|
| **STBD** | `archive/` | 8 884 (train 7 995 / val 889) | 267 888 | 8 hành vi, **không có điện thoại** | Phía trước lớp, trên cao, nhìn mặt HS | Âm tính khó cho Mức 1 + đánh giá pose trên người nhỏ |
| **CCTV** | `Classroom-Cell-Phone-Detection.v1i.yolov8/` | 253 (178/51/24) | 2 238 | `cell-phones`, `person` | **Phía sau lớp**, trên cao, CCTV thật | Dữ liệu dương tính chính cho Mức 2 (fine-tune + đánh giá) |
| **STOCK** | `Classroom Phone Detection.v1i.yolov8/` | 66 (48/9/9) | 122 | `Classroom-Phone-Detection` | Cận cảnh, stock footage | Chỉ minh họa định tính — **không dùng cho train/đánh giá** |

## 1. STBD — `archive/`

* Không kèm `data.yaml`. Tổng số box (267 888) và 8 lớp khớp với **STBD-08 (Student–Teacher Behavior Dataset)**; ánh xạ lớp đã đối chiếu trực quan trên ảnh mẫu:

| ID | Hành vi | Box (train) | Ghi chú kiểm chứng |
|---|---|---|---|
| 0 | write (viết) | 65 289 | `val/1004.jpg`: cả lớp đang viết → toàn lớp 0 |
| 1 | read (đọc) | 53 064 | |
| 2 | listen (nghe giảng) | 105 692 | `train/0.jpg`: đa số HS ngồi thẳng → lớp 2 |
| 3 | turn around (quay đầu) | 4 787 | |
| 4 | raise hand (giơ tay) | 3 850 | `train/0.jpg`: HS giơ tay hàng đầu → lớp 4 |
| 5 | stand (đứng) | 3 662 | `train/0.jpg`: GV đứng tiền cảnh → lớp 5 |
| 6 | discuss (thảo luận) | 4 169 | `val/1260.jpg`: hai HS quay vào nhau, box rộng |
| 7 | guide (GV hướng dẫn) | 604 | `val/1260.jpg`: GV cúi xuống bàn HS, ~1 box/ảnh |

* Ảnh 800×450 hoặc 720×404 (frame video), trung bình **30 người/ảnh**, lớp học Trung Quốc (tiểu học).
* Box người trung vị chỉ **~0,8% diện tích ảnh (~45×65 px)** → thử thách lớn cho pose estimation.
* Không có ảnh trùng lặp, kể cả giữa train/val.
* **Giá trị cho dự án:** `write` + `read` (~118k box) chính là các tư thế "cúi đầu nhưng không dùng điện thoại" → bộ **âm tính khó** để đo tỷ lệ báo động giả của Mức 1.

## 2. CCTV — `Classroom-Cell-Phone-Detection.v1i.yolov8/`

* CCTV thật 1280×720 (có timestamp), **4 phòng học × 2 video** (`Class_{1..4}_Video_{1,2}`), camera đặt **phía sau** lưng sinh viên.
* `cell-phones`: trung vị **0,06% diện tích ảnh (~23×23 px)**; `person`: ~1,2%.
* Mỗi ảnh đều có điện thoại; số `person` ≈ số `cell-phones` → nhiều khả năng chỉ người đang dùng điện thoại được gán nhãn `person`. **Không có mẫu âm tính.**
* ⚠️ **Rò rỉ dữ liệu:** Roboflow chia train/valid/test ngẫu nhiên theo frame — frame của cùng một video nằm ở cả ba tập. Phải **chia lại theo phòng học** (ví dụ: train `Class_1–3`, test `Class_4`).
* ⚠️ Điện thoại ~23 px: nếu đưa nguyên ảnh 1280 px vào YOLO 640 px, điện thoại chỉ còn ~12 px → gần như chắc chắn bị bỏ sót. Củng cố quyết định **crop quanh cổ tay rồi phóng to** ở Mức 2.
* Frame lấy thưa (chỉ số frame không liên tục) → không dùng được để đánh giá Time-Window.

## 3. STOCK — `Classroom Phone Detection.v1i.yolov8/`

* 50 frame từ một video stock **có watermark Getty Images** + 16 ảnh lẻ (`suggested-*`).
* Nhãn ghi CC BY 4.0 nhưng nguồn gốc là nội dung Getty có bản quyền → **rủi ro pháp lý khi công bố**.
* Nhãn chất lượng đáng ngờ (ví dụ `phone-video_mp4-0000`: box ở mép trái ảnh, không trùng điện thoại rõ ràng).
* Cận cảnh, không giống góc camera lớp học → giá trị thấp.

## Kế hoạch sử dụng

| Mục tiêu đánh giá | Dữ liệu | Chỉ số |
|---|---|---|
| Mức 2 — baseline YOLOv8n (COCO `cell phone`) vs. fine-tuned | CCTV, chia theo phòng học | Precision / Recall / mAP@0.5 trên crop và trên toàn ảnh |
| Mức 1 — tỷ lệ báo động giả | STBD lớp `write`, `read`, `listen` | % người bị gắn cờ "Nghi ngờ" (càng thấp càng tốt) |
| Mức 0 — độ phủ pose trên người nhỏ | STBD (box người làm ground truth) | Tỷ lệ người được YOLOv8n-pose phát hiện theo kích thước box |
| Time-Window | Chưa có — cần video tự quay (giai đoạn camera IP) | Độ trễ cảnh báo, số cảnh báo giả/giờ |

## Khuyến nghị tổ chức thư mục

Chuyển 3 bộ dữ liệu vào `datasets/raw/` và thêm `datasets/` vào `.gitignore` (≈1,07 GB, không commit lên Git).
