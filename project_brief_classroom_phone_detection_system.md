# TÊN ĐỀ TÀI: Xây dựng hệ thống giám sát và cảnh báo sử dụng điện thoại trong lớp học dựa trên thị giác máy tính và phân tích hành vi

> **Phiên bản:** 1.1 (cập nhật 2026-10-27) — xem mục 8 "Nhật ký quyết định thiết kế" để biết các thay đổi so với bản 1.0.

## 1. Bối cảnh và Đặt vấn đề
Sự phổ biến của điện thoại thông minh đang tạo ra thách thức lớn đối với sự tập trung của học sinh, sinh viên trong môi trường học đường. Việc giáo viên giám sát thủ công một lớp học đông người (30–50 sinh viên) thường không hiệu quả, làm gián đoạn mạch giảng dạy và có thể bỏ sót các hành vi vi phạm ở góc khuất. Các giải pháp camera AI hiện tại chủ yếu tập trung vào nhận diện khuôn mặt hoặc phát hiện trực tiếp vật thể (Object Detection), dẫn đến nhiều sai sót (nhận nhầm máy tính Casio, hộp bút) hoặc không thể hoạt động khi học sinh giấu điện thoại dưới ngăn bàn.

## 2. Mục tiêu dự án
Xây dựng một hệ thống phần mềm giám sát tự động, xử lý video thời gian thực từ camera lớp học nhằm phát hiện và cảnh báo hành vi sử dụng điện thoại. Hệ thống phải **giảm thiểu bài toán góc khuất** bằng cách phân tích tư thế (vẫn đưa ra cảnh báo mức "Nghi ngờ" khi điện thoại bị che khuất hoàn toàn), giảm thiểu tỷ lệ báo động giả (False Positive) nhờ cơ chế xác nhận hai mức, và cung cấp cho giáo viên một giao diện quản lý sơ đồ lớp học trực quan, không gây áp lực theo dõi liên tục.

## 3. Giải pháp đề xuất (Core Logic)
Dự án áp dụng phương pháp **Nhận diện Cảnh báo Phân cấp (Hierarchical Alert System)** kết hợp giữa nhận diện tư thế đa người (Multi-person Pose Estimation) và phát hiện vật thể (Object Detection) theo chuỗi nối tiếp (Cascade Pipeline):

* **Mức 0 - Theo dõi & Định vị chỗ ngồi:** Mô hình **YOLOv8n-pose** phát hiện đồng thời tất cả học sinh trong khung hình cùng 17 điểm khung xương (chuẩn COCO). Mỗi người được gán ID theo thời gian bằng bộ theo vết (ByteTrack) và được ánh xạ vào một ô bàn học dựa trên **vùng đa giác (polygon) đã hiệu chỉnh trước** trên khung hình camera. Hệ thống hỗ trợ cấu hình **Camera Profile** theo góc lắp đặt: `FRONT_VIEW` (phía trước bục giảng / webcam trực diện) hoặc `REAR_VIEW` (phía sau lưng học sinh).
* **Mức 1 - Đánh giá tư thế (Nghi ngờ - Vàng):** Áp dụng công thức đặc trưng thích ứng theo `Camera Profile`:
  * Với `FRONT_VIEW`: Tính góc gập của cổ (vector trung điểm hai vai → mũi/tai so với phương thẳng đứng), khoảng cách giữa hai cổ tay (chuẩn hóa theo độ rộng vai) và vị trí cổ tay thấp dưới mép bàn.
  * Với `REAR_VIEW` (khi mũi/mắt bị khuất): Tính tỷ lệ khoảng cách dọc từ trung điểm hai tai (hoặc đỉnh hộp giới hạn đầu) xuống đường nối hai vai (chuẩn hóa theo độ rộng vai) để phát hiện tư thế rụt cổ/cúi đầu, kết hợp vị trí khuỷu tay/cổ tay hướng vào trong lòng.
  * Nếu tư thế bất thường duy trì liên tục trong $X$ giây (Time-Window Tracking), hệ thống bật cờ "Nghi ngờ".
* **Mức 2 - Xác nhận vật thể (Vi phạm - Đỏ):** Chỉ kích hoạt khi mức 1 thỏa mãn. Hệ thống cắt (crop) vùng ảnh xung quanh hai cổ tay (hoặc vùng trước ngực/đùi tùy góc nhìn) và đưa vào mô hình YOLOv8n để tìm kiếm nhãn `cell phone`. Nếu độ tin cậy $> 60\%$ (ngưỡng có thể cấu hình), hệ thống ghi nhận vi phạm và lưu lại ảnh bằng chứng.

> **Lưu ý phạm vi:** Khi điện thoại bị che khuất hoàn toàn (dưới ngăn bàn), hệ thống chỉ có thể dừng ở **Mức 1 (Vàng)**. Mức 2 (Đỏ) chỉ được xác nhận khi điện thoại lộ ra trong khung hình. Đây là giới hạn vật lý của camera và sẽ được nêu rõ trong phần đánh giá.

## 4. Phạm vi dự án & Sản phẩm bàn giao
* **Module AI Worker:** Tiến trình/luồng xử lý thị giác máy tính (OpenCV, YOLOv8n-pose, YOLOv8n, ByteTrack), đọc luồng video (RTSP / webcam / file), xử lý Time-Window Tracking và xuất luồng **Live Annotated Video Stream (MJPEG)**.
* **Công cụ hiệu chỉnh chỗ ngồi (Seat Calibration):** Cho phép vẽ vùng đa giác của từng bàn học trực tiếp trên khung hình camera ngay trên Web UI và gắn nhãn vị trí (ví dụ *Dãy 2 - Bàn 3*).
* **Backend API Service (Python FastAPI):** Quản lý cấu hình phòng học & ma trận bàn học (Seat Mapping), lưu trữ lịch sử vi phạm và ảnh chụp bằng chứng vào PostgreSQL, đồng thời đẩy sự kiện thời gian thực qua **WebSocket** và phục vụ luồng MJPEG từ AI Worker.
* **Frontend Dashboard:** Giao diện Web (React + TailwindCSS) hiển thị song song **luồng Live Video trực tiếp** (có overlay AI) và **sơ đồ lưới lớp học thời gian thực** (nhấp nháy ô bàn vi phạm qua WebSocket) kèm danh sách lịch sử bằng chứng mà không cần tải lại trang.
* **Tài liệu:** Báo cáo đồ án chi tiết (Cơ sở lý thuyết, kiến trúc hệ thống, đánh giá độ chính xác của thuật toán) kèm bộ mã nguồn mở.

### Thứ tự triển khai
Dự án được chia thành các tiểu dự án, mỗi phần có spec → plan → triển khai riêng:
0. **System Architecture & Contract** — mô hình dữ liệu PostgreSQL, REST API, WebSocket schema, cấu trúc thư mục.
1. **AI Engine & Evaluation** — pipeline Mức 0–1–2, đánh giá trên dataset và chạy thử với webcam/video.
2. **FastAPI Backend & AI Integration** — kết nối DB, REST API, WebSocket, MJPEG stream.
3. **Frontend Dashboard** (bao gồm công cụ hiệu chỉnh chỗ ngồi trên Web).

### Chiến lược dữ liệu (Data Strategy)
Hiện chưa có video lớp học thật. Nguồn video cho phát triển/demo đi theo lộ trình:
1. **Bước 1 — Webcam laptop:** Laptop nhìn thẳng, giả lập 1–2 "bàn". Mục tiêu: chạy thông pipeline end-to-end (AI Worker → Backend → Dashboard) nhanh nhất.
2. **Bước 2 — Điện thoại cũ làm camera IP (RTSP):** Gắn cao ở góc phòng, 2–4 người ngồi giả lập lớp học → góc nhìn gần thực tế, kiểm thử luồng RTSP và quay video có kịch bản để đánh giá Time-Window.

AI Worker trừu tượng hóa nguồn video (webcam / file / RTSP) để chuyển đổi giữa các bước chỉ bằng cấu hình.

**Dataset công khai đã thu thập** (phân tích chi tiết: `docs/data/2026-10-27-dataset-analysis.md`):

| Bộ dữ liệu | Quy mô | Vai trò |
|---|---|---|
| **STBD-08** (`archive/`) — 8 hành vi HS/GV, camera phía trước | 8 884 ảnh, 267 888 box, **không có điện thoại** | Âm tính khó cho Mức 1 (`write`, `read` = cúi đầu nhưng không dùng điện thoại); đánh giá pose trên người nhỏ |
| **Classroom-Cell-Phone-Detection** (Roboflow) — CCTV thật, camera phía sau | 253 ảnh, 4 phòng × 2 video, nhãn `cell-phones`, `person` | Dữ liệu dương tính chính cho Mức 2: baseline vs. fine-tune. **Bắt buộc chia lại theo phòng học** (bản gốc rò rỉ frame giữa train/test) |
| Classroom Phone Detection (Roboflow) — stock footage | 66 ảnh | **Loại bỏ** khỏi train/đánh giá: watermark Getty (rủi ro bản quyền), nhãn kém, góc cận cảnh |

> **Hạn chế:** Các bộ trên đều là **ảnh tĩnh / frame lấy thưa**, không đánh giá được logic Time-Window → dùng video tự quay ở Bước 2. Điện thoại trong dữ liệu CCTV chỉ ~23×23 px, khẳng định sự cần thiết của bước crop-và-phóng-to ở Mức 2.

## 5. Kiến trúc hệ thống và Công nghệ sử dụng
* **Lõi Trí tuệ nhân tạo (AI Engine):** Python, OpenCV (xử lý luồng hình ảnh), Ultralytics **YOLOv8n-pose** (nhận diện tư thế đa người, chạy được trên CPU), Ultralytics YOLOv8n (phát hiện thiết bị di động cục bộ), ByteTrack (theo vết đối tượng, tích hợp sẵn trong Ultralytics).
* **Backend & Xử lý nghiệp vụ:** **Python FastAPI** (REST API bất đồng bộ, quản lý phiên giám sát và cấu hình bàn học), **SQLAlchemy / SQLModel** (ORM), **Native WebSocket** (truyền tải sự kiện thời gian thực tới Dashboard) và **Multipart Streaming** (phục vụ luồng MJPEG).
* **Cơ sở dữ liệu:** **PostgreSQL** (Lưu trữ Entity sơ đồ lớp, cấu hình đa giác bàn học, logs vi phạm, đường dẫn ảnh chụp bằng chứng).
* **Giao diện người dùng (UI/UX):** React.js (Vite + TypeScript) kết hợp TailwindCSS, vẽ sơ đồ lớp học qua CSS Grid, công cụ vẽ đa giác (HTML5 Canvas/SVG) và tiếp nhận kết nối WebSocket.

```
[Camera / Webcam / Video]
           |
           v
+-------------------------------------------------------------+
| Backend & AI Service (Python FastAPI + AI Worker Process)   |
|  - AI Pipeline: YOLOv8n-pose -> Time-Window -> YOLOv8n Crop |
|  - REST API + WebSocket (/ws/alerts) + MJPEG (/api/stream)  |
+--------------------------+----------------------------------+
                           |
             +-------------+-------------+
             v                           v
    [PostgreSQL DB]            [React Dashboard UI]
  (Rooms, Seats, Logs)       (Live MJPEG + Seat Grid)
```

## 6. Tính đổi mới và Điểm nhấn (USPs)
* **Tối ưu hiệu năng:** Xử lý luồng video theo cơ chế Frame Skipping và Cascade (chỉ chạy YOLO phát hiện điện thoại khi Pose có bất thường), giúp hệ thống hoạt động mượt mà ngay cả trên thiết bị phần cứng phổ thông mà không cần GPU đắt tiền.
* **Tôn trọng quyền riêng tư (Privacy-friendly):** Hệ thống giám sát dựa trên hành vi không gian và tọa độ bàn học thay vì định danh khuôn mặt từng cá nhân, giảm thiểu các lo ngại về đạo đức dữ liệu.
* **Ứng dụng thực tế cao:** Trả về kết quả hiển thị trên ma trận sơ đồ bàn ghế thay vì chỉ vẽ Bounding Box lên video, giúp giáo viên định vị chính xác vị trí vi phạm (ví dụ: *Bàn số 3 - Dãy 2*) để có biện pháp can thiệp sư phạm phù hợp.

## 7. Giới hạn đã biết & Rủi ro
* **Che khuất hoàn toàn:** Không thể xác nhận vi phạm mức Đỏ khi điện thoại không xuất hiện trong khung hình (xem lưu ý mục 3).
* **Học sinh ở xa camera:** Người ở hàng cuối có kích thước nhỏ, độ chính xác keypoint giảm; cần đánh giá theo từng vùng khoảng cách trong báo cáo.
* **Nhận nhầm vật thể:** Lớp `cell phone` của COCO có thể nhầm với máy tính bỏ túi, ví trên vùng crop độ phân giải thấp. Giảm thiểu bằng ngưỡng tin cậy + yêu cầu xác nhận lặp lại qua nhiều khung hình; cân nhắc fine-tune nếu dữ liệu thử nghiệm cho thấy cần thiết.
* **Phối cảnh camera:** Bàn học bị méo do góc nhìn chéo; giải quyết bằng hiệu chỉnh đa giác thủ công thay vì chia lưới đều.

## 8. Nhật ký quyết định thiết kế
| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-10-27 | Stack công nghệ được phép điều chỉnh toàn bộ. | Thống nhất với nhóm dự án. |
| 2026-10-27 | Thay **MediaPipe Pose** bằng **YOLOv8n-pose**. | MediaPipe Pose (legacy) chỉ nhận 1 người/khung hình; bản Tasks API hỗ trợ nhiều người nhưng kém với người nhỏ/xa — không phù hợp lớp 30–50 người. YOLOv8n-pose nhận đa người trong một lần suy luận, cùng hệ sinh thái Ultralytics với YOLOv8n, vẫn chạy được trên CPU. |
| 2026-10-27 | Thay **Oracle Database** bằng **PostgreSQL**. | Nhẹ hơn khi phát triển/demo, dễ chạy bằng Docker, tương thích tốt với Spring Data JPA. |
| 2026-10-27 | Định nghĩa lại mục tiêu "góc khuất": Mức 1 hoạt động khi điện thoại bị che, Mức 2 chỉ khi điện thoại lộ ra. | Mâu thuẫn trong bản 1.0: YOLO trên vùng crop không thể thấy điện thoại dưới ngăn bàn. |
| 2026-10-27 | Bổ sung **Mức 0** (theo vết + ánh xạ chỗ ngồi bằng đa giác hiệu chỉnh) và công cụ Seat Calibration. | Cần ID ổn định theo thời gian cho Time-Window Tracking và ánh xạ chính xác người → bàn khi camera nhìn chéo. |
| 2026-10-27 | Dữ liệu: webcam laptop cho phát triển/demo; SCB-Dataset cho đánh giá định lượng theo khung hình. | Chưa có video lớp học thật; SCB-Dataset công khai, có nhãn `using the phone`, chụp trong lớp học thật. |
| 2026-10-27 | Thay SCB-Dataset bằng 3 bộ đã tải: **STBD-08** (âm tính Mức 1), **Classroom-Cell-Phone-Detection** (dương tính Mức 2, chia lại theo phòng học), loại bỏ bộ stock footage. | Kết quả phân tích dữ liệu ngày 2026-10-27 (xem `docs/data/2026-10-27-dataset-analysis.md`). |
| 2026-10-27 | Nguồn video demo: webcam laptop trước → điện thoại làm camera IP (RTSP) đặt trên cao sau. | Ra kết quả end-to-end nhanh, sau đó tiến gần góc camera thực tế và có video để đánh giá Time-Window. |
| 2026-10-27 | Hỗ trợ cả 2 góc đặt camera (`FRONT_VIEW` và `REAR_VIEW`) thông qua cấu hình `Camera Profile`. | Tận dụng được cả 2 bộ dataset (`STBD-08` góc trước và `CCTV` góc sau), đồng thời tương thích với cả webcam trực diện lẫn camera lớp học lắp sau lưng. |
| 2026-10-27 | Dashboard hiển thị song song Live Video Stream (MJPEG) có vẽ overlay AI cùng với sơ đồ lưới bàn học. | Trực quan hóa quá trình xử lý của AI theo thời gian thực (khung xương, vùng bàn, bounding box) khi demo và bảo vệ đồ án. |
| 2026-10-27 | Thay **Java Spring Boot + STOMP** bằng **Python FastAPI + Native WebSocket**. | Đồng nhất một ngôn ngữ Python cho cả AI Worker và Backend API; loại bỏ overhead truyền ảnh/sự kiện liên tiến trình, phục vụ cả REST, WebSocket và MJPEG từ một backend duy nhất. |
| 2026-10-27 | Môi trường AI dùng **Python 3.11/3.12** (virtualenv riêng), không dùng Python 3.14 của hệ thống. | PyTorch/Ultralytics chưa chắc hỗ trợ ổn định Python 3.14. |
| 2026-10-27 | Phát triển trên máy có GPU (RTX 4060 Laptop) nhưng **benchmark hiệu năng bắt buộc chạy trên CPU**. | Giữ đúng USP "không cần GPU đắt tiền"; GPU chỉ để tăng tốc phát triển/fine-tune. |
