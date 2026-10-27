# DỰ ÁN NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM (TRAFFIC SIGN DEMO)
Ứng dụng nhận diện biển báo giao thông bằng mạng nơ-ron tích chập (YOLO) và giao diện trực quan Tkinter trên Python.

---

## 1. TỔNG QUAN DỰ ÁN

* **Mục tiêu:** Xây dựng ứng dụng demo hoàn chỉnh cho phép chọn ảnh từ máy tính, tự động phát hiện vị trí, vẽ khung (bounding box) và phân loại 4 loại biển báo cấm phổ biến tại Việt Nam cùng độ tin cậy (*Confidence*).
* **Phạm vi 4 lớp biển báo lựa chọn:**
  * **ID 0:** `Cấm đỗ xe` *(Biển tròn viền đỏ, nền xanh, 1 vạch chéo đỏ)*
  * **ID 1:** `Cấm dừng và đỗ xe` *(Biển tròn viền đỏ, nền xanh, 2 vạch chéo đỏ)*
  * **ID 2:** `Cấm rẽ trái` *(Biển tròn viền đỏ, mũi tên rẽ trái bị gạch chéo)*
  * **ID 3:** `Cấm rẽ phải` *(Biển tròn viền đỏ, mũi tên rẽ phải bị gạch chéo)*
* **Nhóm thực hiện & Phân công vai trò:**
  * **Tâm:** Phụ trách Dữ liệu, Kiểm định nhãn & Thực nghiệm Xử lý ảnh OpenCV.
  * **Hào:** Phụ trách Huấn luyện mô hình YOLO, Đánh giá chỉ số & Viết hàm nhận diện.
  * **Hảo:** Phụ trách Xây dựng giao diện Tkinter, Tích hợp mô hình & Quản lý Git.
  * **Đạt:** Phụ trách Thu thập tư liệu, Soạn thảo Báo cáo đồ án (Word/PDF) & Slide thuyết trình.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
D:\thiet_ke\X_L_A\traffic-sign-demo\
├── app.py                           # [Hảo] Ứng dụng giao diện Tkinter chính
├── requirements.txt                 # [Hảo] Danh sách thư viện cần cài đặt
├── README.md                        # Tài liệu hướng dẫn chung và bàn giao toàn dự án
├── .gitignore                       # Danh sách các file/thư mục không đưa lên Git
│
├── configs/
│   ├── classes.json                 # [Tâm] Bảng mã ID và tên 4 loại biển báo
│   ├── data.yaml                    # [Tâm] Cấu hình huấn luyện YOLO trên máy cục bộ
│   └── data.example.yaml            # [Tâm] Cấu hình mẫu đưa lên Git
│
├── src/
│   ├── __init__.py                  # Package init
│   ├── paths.py                     # [Tâm / Hảo] Quản lý đường dẫn chung bằng pathlib.Path
│   ├── prepare_data.py              # [Tâm] Tiền xử lý, lọc trùng và chia tập dữ liệu
│   ├── check_and_visualize.py       # [Tâm] Kiểm định 100% dữ liệu và vẽ nhãn trực quan
│   ├── preprocess.py                # [Tâm] Thực nghiệm tiền xử lý ảnh số bằng OpenCV
│   ├── train.py                     # [Hào] Huấn luyện mô hình YOLO11
│   ├── predict.py                   # [Hào] Hàm nhận diện cho giao diện Tkinter gọi
│   └── evaluate.py                  # [Hào] Đánh giá mô hình trên tập Test
│
├── data/
│   ├── README.md                    # [Tâm] Tài liệu quản lý dữ liệu chi tiết
│   ├── raw/                         # Dữ liệu gốc liên kết (10.691 ảnh - không commit)
│   └── processed/                   # [Tâm BÀN GIAO CHO HÀO] Bộ dữ liệu sạch 800 ảnh
│       ├── images/{train,val,test}  # 560 train - 120 val - 120 test
│       └── labels/{train,val,test}  # File nhãn YOLO tương ứng
│
├── models/
│   ├── README.md                    # [Hào] Hướng dẫn tải trọng số và vị trí lưu
│   └── best.pt                      # [Hào BÀN GIAO CHO HẢO] Trọng số mô hình sau huấn luyện
│
├── outputs/
│   ├── label_verification/          # [Tâm BÀN GIAO CHO ĐẠT] Ảnh vẽ bounding box trực quan
│   │   ├── verification_overview_grid.jpg  # Ảnh lưới 12 ô tổng quan 4 lớp x 3 tập
│   │   └── t4_verification_report.json     # Báo cáo kiểm định định lượng 100% dữ liệu
│   ├── preprocessing/               # [Tâm BÀN GIAO CHO ĐẠT] Ảnh thực nghiệm OpenCV
│   │   ├── opencv_experiment_report.md     # Báo cáo lý thuyết & so sánh OpenCV vs YOLO
│   │   └── case_1_bright_light/ ...        # 5 ca thử nghiệm thực tế (7 ảnh/ca + collage)
│   ├── training/                    # [Hào BÀN GIAO CHO ĐẠT] Log, biểu đồ loss/mAP
│   └── predictions/                 # [Hảo] Thư mục lưu ảnh nhận diện từ giao diện
│
└── docs/
    ├── report_outline.md            # [Đạt] Dàn ý chi tiết bài báo cáo
    ├── data_summary.csv             # [Tâm BÀN GIAO CHO ĐẠT] Thống kê số lượng ảnh/đối tượng
    ├── split_manifest.csv           # [Tâm BÀN GIAO CHO ĐẠT] Nhật ký truy vết 800 ảnh
    ├── experiment_log.csv           # [Hào BÀN GIAO CHO ĐẠT] Nhật ký các lần chạy huấn luyện
    ├── test_metrics.csv             # [Hào BÀN GIAO CHO ĐẠT] Bảng điểm số Precision/Recall/mAP
    ├── demo_checklist.md            # [Hảo] Danh sách kiểm thử giao diện
    ├── figures/                     # [Đạt] Các hình ảnh được chọn lọc cho báo cáo
    └── report/                      # [Đạt] Báo cáo đồ án chính thức (Word / PDF)
```

---

## 3. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG & CHẠY LỆNH

### Bước 1: Khởi tạo môi trường ảo Python
Mở Terminal tại thư mục gốc `traffic-sign-demo`:
```bash
# Khởi tạo môi trường ảo (khuyến nghị Python 3.10 - 3.12 hoặc 3.14)
python -m venv .venv

# Kích hoạt môi trường trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Hoặc trên Command Prompt (cmd):
.\.venv\Scripts\activate.bat
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 4. HƯỚNG DẪN CHI TIẾT DÀNH CHO BẠN HÀO (HUẤN LUYỆN & NHẬN DIỆN)

Tâm đã hoàn thành toàn bộ phần chuẩn bị và kiểm định dữ liệu. Hào chỉ cần làm theo các bước chuẩn sau:

### L1. Kiểm tra dữ liệu nhận từ Tâm
* Mở file [`configs/data.yaml`](configs/data.yaml) để xác nhận đường dẫn dữ liệu sạch `data/processed/`.
* Mở file [`configs/classes.json`](configs/classes.json) để xác nhận 4 ID:
  `0: Cấm đỗ xe`, `1: Cấm dừng và đỗ xe`, `2: Cấm rẽ trái`, `3: Cấm rẽ phải`.
* Dữ liệu đã chia sẵn: **560 ảnh Train, 120 ảnh Val, 120 ảnh Test**; mỗi lớp có đúng 200 đối tượng.

### L2. Viết chương trình huấn luyện ([`src/train.py`](src/train.py))
* Sử dụng mô hình `yolo11n.pt` (hoặc `yolov8n.pt`) làm trọng số khởi đầu.
* Khung code chuẩn khuyến nghị:
  ```python
  from ultralytics import YOLO
  from src.paths import DATA_YAML_PATH, ROOT_DIR

  def main():
      model = YOLO("yolo11n.pt")
      results = model.train(
          data=str(DATA_YAML_PATH),
          epochs=30,        # Khởi đầu thử nghiệm 3-5 epoch, sau đó tăng lên 30 epoch
          imgsz=416,        # Kích thước 416x416 giúp train nhanh trên CPU
          batch=4,          # Giảm xuống 2 nếu thiếu RAM
          device="cpu",     # Đặt "cpu" hoặc 0 nếu máy có GPU NVIDIA CUDA
          workers=2,
          fliplr=0.0,       # CỰC KỲ QUAN TRỌNG: Không lật ngang ảnh (tránh làm đảo biển rẽ trái/phải)
          flipud=0.0,       # Không lật dọc
          project=str(ROOT_DIR / "outputs" / "training"),
          name="traffic_sign_yolo11",
          exist_ok=True,
      )

  if __name__ == "__main__":
      main()
  ```
* Chạy huấn luyện bằng lệnh:
  ```bash
  python -m src.train
  ```

### L3. Lưu mô hình & Ghi nhật ký
* Sau khi train xong, sao chép trọng số tốt nhất từ thư mục kết quả:
  `outputs/training/traffic_sign_yolo11/weights/best.pt` $\longrightarrow$ lưu vào **`models/best.pt`**.
* Cập nhật các thông số chạy thực tế (epoch, batch, thời gian chạy, mAP50 trên tập val) vào file **`docs/experiment_log.csv`**.

### L4. Viết hàm nhận diện bàn giao cho Hảo tích hợp GUI ([`src/predict.py`](src/predict.py))
Hảo cần một hàm nạp model một lần và hàm nhận diện trả về kết quả chuẩn để hiển thị lên Tkinter. Hào viết đúng hợp đồng hàm sau:
```python
import cv2
from ultralytics import YOLO

def load_model(model_path):
    """Nạp trọng số YOLO (chỉ gọi 1 lần khi mở app)."""
    return YOLO(model_path)

def predict_image(model, image_path: str, confidence: float = 0.25):
    """
    Suy luận trên 1 ảnh.
    Trả về: (annotated_bgr, detections)
      - annotated_bgr: numpy.ndarray ảnh đã vẽ bounding box (BGR)
      - detections: list các dict [
            {
                "class_id": 0,
                "class_name": "Cấm đỗ xe",
                "confidence": 0.89,
                "bbox": [x1, y1, x2, y2]
            }, ...
        ]
    Nếu không có biển thì trả về detections là list rỗng [].
    """
    results = model.predict(source=image_path, conf=confidence, verbose=False)
    r = results[0]
    annotated_bgr = r.plot()  # Ảnh có sẵn khung của YOLO
    
    detections = []
    for box in r.boxes:
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        cname = model.names.get(cid, f"Class {cid}")
        detections.append({
            "class_id": cid,
            "class_name": cname,
            "confidence": conf,
            "bbox": [int(x) for x in xyxy],
        })
    return annotated_bgr, detections
```

### L5. Đánh giá chất lượng mô hình trên tập Test ([`src/evaluate.py`](src/evaluate.py))
* Đánh giá mô hình `models/best.pt` trên 120 ảnh của tập `data/processed/images/test`.
* Xuất bảng chỉ số Precision, Recall, mAP50, mAP50-95 vào file **`docs/test_metrics.csv`** để bàn giao cho Đạt.

---

## 5. HƯỚNG DẪN CHI TIẾT DÀNH CHO BẠN ĐẠT (VIẾT BÁO CÁO & THUYẾT TRÌNH)

Đạt có thể lấy ngay toàn bộ tư liệu, bảng biểu và hình ảnh đã được chuẩn bị sẵn để đưa vào bài báo cáo:

### Đ1. Dàn ý báo cáo đề xuất ([`docs/report_outline.md`](docs/report_outline.md))
* **Chương 1:** Giới thiệu đề tài, mục tiêu nhận diện 4 loại biển báo cấm, phạm vi ứng dụng.
* **Chương 2:** Cơ sở lý thuyết (Xử lý ảnh số: không gian màu HSV, hình thái học morphology, tìm contour; Học sâu: mạng nơ-ron tích chập và kiến trúc YOLO).
* **Chương 3:** Thu thập, tiền xử lý và tổ chức dữ liệu.
* **Chương 4:** Xây dựng ứng dụng và Thực nghiệm (So sánh phương pháp OpenCV cổ điển vs YOLO; Kết quả huấn luyện mô hình).
* **Chương 5:** Thiết kế giao diện người dùng Tkinter & Thử nghiệm thực tế.
* **Chương 6:** Kết luận, hạn chế và hướng phát triển tương lai.

### Đ2. Tư liệu Chương 3: Dữ liệu & Tiền xử lý (Lấy từ Tâm)
1. **Bảng số liệu thống kê:** Mở file **[`docs/data_summary.csv`](docs/data_summary.csv)**:
   * Tổng số ảnh: **800 ảnh** (Train: 560 ảnh - 70%, Val: 120 ảnh - 15%, Test: 120 ảnh - 15%).
   * Phân bố đối tượng: Mỗi lớp có **đúng 200 đối tượng** (Cấm đỗ xe: 200, Cấm dừng đỗ: 200, Cấm rẽ trái: 200, Cấm rẽ phải: 200) $\rightarrow$ Đạt nhấn mạnh đây là **tập dữ liệu cân bằng hoàn hảo (Class Balance)**, giúp mô hình không bị thiên vị.
2. **Kỹ thuật chống rò rỉ dữ liệu (Data Leakage):**
   * Giải thích: Bộ dữ liệu gốc được trích xuất từ camera hành trình nên có nhiều khung hình liên tiếp của cùng một đoạn đường.
   * Giải pháp: Nhóm đã nhóm các khung hình theo cụm video (`group_id`) trong file [`docs/split_manifest.csv`](docs/split_manifest.csv) để đảm bảo toàn bộ các khung hình của một đoạn đường chỉ nằm trong Train hoặc chỉ nằm trong Test.
3. **Hình ảnh minh họa Bounding Box:**
   * Chèn bức ảnh **[`outputs/label_verification/verification_overview_grid.jpg`](outputs/label_verification/verification_overview_grid.jpg)** vào báo cáo: Đây là ảnh lưới 12 ô hiển thị trực quan 4 lớp trên 3 tập với viền màu và tên tiếng Việt rõ ràng.

### Đ3. Tư liệu Chương 2 & 4: Thực nghiệm Xử lý ảnh OpenCV (Lấy từ Tâm)
* Mở file **[`outputs/preprocessing/opencv_experiment_report.md`](outputs/preprocessing/opencv_experiment_report.md)** để lấy nguyên văn cơ sở lý thuyết và kết quả:
  * **Lý thuyết:** Giải thích tại sao dùng HSV thay cho RGB; kỹ thuật gộp 2 dải màu đỏ $[0, 10]$ và $[160, 179]$; tác dụng của phép Mở (Erosion $\to$ Dilation) và phép Đóng (Dilation $\to$ Erosion) với kernel elip.
  * **Bảng kết quả 5 trường hợp:**
    * *Ảnh đủ sáng:* IoU = 0.75 (OpenCV bắt trúng vị trí biển).
    * *Ảnh thiếu sáng/bóng râm:* IoU = 0.00 (màu đỏ bị chìm vào bóng tối).
    * *Ảnh biển nhỏ ở xa:* IoU = 0.00 (bị phép lọc nhiễu xóa mất).
    * *Ảnh biển lớn:* IoU = 0.73 (bắt chính xác 100%).
    * *Ảnh nền phức tạp:* Bị 10 dương tính giả do bắt nhầm đèn xe ô tô màu đỏ.
  * **Hình ảnh chèn báo cáo:** Lấy các ảnh ghép quy trình 6 bước tại:
    `outputs/preprocessing/case_1_bright_light/pipeline_collage_*.jpg`.
  * **Luận điểm so sánh:** *OpenCV nhanh, nhẹ nhưng dễ bị nhiễu và không đọc được ký hiệu bên trong biển; do đó YOLO là giải pháp tối ưu cho ứng dụng nhận diện hoàn chỉnh.*

### Đ4. Tư liệu Chương 4 & 5: Kết quả Mô hình & Giao diện (Lấy từ Hào và Hảo)
* **Từ Hào:** Lấy biểu đồ huấn luyện trong `outputs/training/`, bảng điểm mAP trong `docs/test_metrics.csv` và ảnh một số ca dự đoán đúng/sai.
* **Từ Hảo:** Lấy ảnh chụp giao diện Tkinter đang chạy thực tế, quy trình chọn ảnh $\rightarrow$ nhận diện $\rightarrow$ hiển thị bảng kết quả $\rightarrow$ lưu ảnh.

---

## 6. DANH SÁCH KIỂM TRA TRƯỚC KHI BÁO CÁO / NỘP BÀI

- [x] **T1:** Bảng nhãn `configs/classes.json` chuẩn 4 ID.
- [x] **T2 & T3:** Dữ liệu 800 ảnh sạch, chuẩn YOLO, không trùng lặp, chia tập 70/15/15.
- [x] **T4:** Kiểm định 100% tọa độ nhãn hợp lệ, có ảnh lưới tổng quan 12 ô `verification_overview_grid.jpg`.
- [x] **T5:** Thực nghiệm OpenCV 5 trường hợp, có báo cáo phân tích Markdown chi tiết.
- [x] **T6:** Toàn bộ tài liệu hướng dẫn và bản đồ vị trí file được cập nhật đầy đủ.
- [ ] **L1 - L6:** Hào hoàn thành huấn luyện `best.pt`, xuất `predict.py` và bảng điểm `test_metrics.csv`.
- [ ] **H1 - H6:** Hảo hoàn thiện giao diện Tkinter gọi `predict_image`, kiểm thử và đóng gói.
- [ ] **Đ1 - Đ6:** Đạt hoàn thiện bản báo cáo Word/PDF và slide thuyết trình.
