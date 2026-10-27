# TÀI LIỆU QUẢN LÝ DỮ LIỆU & BIÊN BẢN BÀN GIAO (TÂM - T1 ĐẾN T6)

Tài liệu này ghi lại toàn bộ quy trình chuẩn bị dữ liệu, thực nghiệm xử lý ảnh số, cấu trúc thư mục và hướng dẫn bàn giao chi tiết cho **bạn Hào** (phụ trách Huấn luyện/Mô hình) và **bạn Đạt** (phụ trách Báo cáo/Thuyết trình).

---

## 1. NGUỒN GỐC DỮ LIỆU GỐC (RAW DATA)

* **Tên bộ dữ liệu:** `vietnam_traffic_signs_dataset-main` (bộ 16 lớp biển báo giao thông Việt Nam theo định dạng YOLO chuẩn).
* **Nguồn tải về:** Giải nén từ file nén `vietnam_traffic_signs_dataset-main.zip` vào thư mục của đồ án.
* **Vị trí thư mục gốc trên máy:** `D:\thiet_ke\X_L_A\vietnam_traffic_signs_dataset-main`
* **Cách kết nối vào dự án (`data/raw/`):**
  Thư mục `data/raw/` được liên kết trực tiếp (Directory Junction) tới bộ dữ liệu gốc để không làm trùng lặp dung lượng ổ cứng:
  ```cmd
  cmd /c mklink /J "D:\thiet_ke\X_L_A\traffic-sign-demo\data\raw" "D:\thiet_ke\X_L_A\vietnam_traffic_signs_dataset-main"
  ```
  *(Lưu ý: Thư mục `data/raw/` đã được thêm vào `.gitignore`, tuyệt đối không commit dữ liệu gốc lên GitHub).*
* **Cấu trúc dữ liệu gốc:**
  ```text
  data/raw/
  ├── data/
  │   ├── images/   (10.691 ảnh gốc định dạng .jpg)
  │   └── labels/   (10.691 file nhãn .txt gốc)
  └── test_dataset/ (1.026 ảnh - chỉ chứa lớp cấm đỗ, không dùng làm tập test chung)
  ```

---

## 2. BẢNG QUY ƯỚC 4 LỚP BIỂN BÁO MỤC TIÊU (ĐÃ CHỐT T1)

Toàn bộ dự án thống nhất dùng bảng mã ID từ $0$ đến $3$ trong file [`configs/classes.json`](../configs/classes.json):

| ID Dự án | Tên Biển Báo (Tiếng Việt) | ID Nguồn (16 lớp) | Tên Nhãn Gốc Trong Dataset | Mã Màu Trực Quan (RGB) |
| :---: | :--- | :---: | :--- | :---: |
| **0** | **Cấm đỗ xe** | 3 | `no parking` | `(30, 144, 255)` (Xanh dương) |
| **1** | **Cấm dừng và đỗ xe** | 2 | `no stop or parking` | `(220, 20, 60)` (Đỏ tươi) |
| **2** | **Cấm rẽ trái** | 11 | `no turn left` | `(255, 140, 0)` (Cam đậm) |
| **3** | **Cấm rẽ phải** | 12 | `no turn right` | `(46, 139, 87)` (Xanh lá) |

---

## 3. CÁC LỆNH VẬN HÀNH TOÀN BỘ PIPELINE

Mọi lệnh được thực thi từ thư mục gốc của repository (`traffic-sign-demo`):

### Bước 1: Lọc dữ liệu, Remap nhãn và Chia tập (T2 & T3)
```bash
python -m src.prepare_data --max_samples 800 --seed 42
```
*Tác vụ:* Quét 4.807 ảnh ứng viên $\rightarrow$ lọc trùng lặp MD5 $\rightarrow$ gom cụm chuỗi video $\rightarrow$ lấy mẫu 800 ảnh cân bằng 4 lớp $\rightarrow$ chia theo tỷ lệ 70% Train (560) - 15% Val (120) - 15% Test (120) $\rightarrow$ xuất vào `data/processed/`.

### Bước 2: Kiểm định chất lượng & Trực quan hóa Bounding Box (T4)
```bash
python -m src.check_and_visualize
```
*Tác vụ:* Kiểm tra 100% file ảnh và nhãn $\rightarrow$ xác thực tọa độ hộp $\in [0, 1]$ $\rightarrow$ vẽ bounding box và tag tên tiếng Việt $\rightarrow$ xuất ảnh lưới tổng quan 12 ô `verification_overview_grid.jpg`.

### Bước 3: Thực nghiệm Xử lý ảnh số OpenCV (T5)
```bash
python -m src.preprocess
```
*Tác vụ:* Thực thi pipeline (BGR $\to$ HSV $\to$ Mặt nạ đỏ $\to$ Morphology Open/Close $\to$ Contours $\to$ Ứng viên ROI) trên 5 điều kiện thực tế $\rightarrow$ xuất báo cáo phân tích Markdown.

---

## 4. BẢN ĐỒ VỊ TRÍ TẤT CẢ CÁC FILE ĐẦU RA

```text
D:\thiet_ke\X_L_A\traffic-sign-demo\
├── configs/
│   ├── classes.json                    # Bảng ID và tên 4 lớp biển báo (T1)
│   ├── data.yaml                       # File cấu hình nạp vào YOLO để train (đường dẫn tuyệt đối máy này)
│   └── data.example.yaml               # File mẫu đưa lên Git
│
├── data/
│   ├── README.md                       # File tài liệu hướng dẫn và bàn giao này (T6)
│   ├── raw/                            # Dữ liệu gốc liên kết (10.691 ảnh)
│   └── processed/                      # BỘ DỮ LIỆU SẠCH 800 ẢNH (Bàn giao cho Hào)
│       ├── images/
│       │   ├── train/                  # 560 ảnh train (70%)
│       │   ├── val/                    # 120 ảnh validation (15%)
│       │   └── test/                   # 120 ảnh test kiểm tra cuối (15%)
│       └── labels/
│           ├── train/                  # 560 file nhãn YOLO remap (0, 1, 2, 3)
│           ├── val/                    # 120 file nhãn YOLO remap (0, 1, 2, 3)
│           └── test/                   # 120 file nhãn YOLO remap (0, 1, 2, 3)
│
├── docs/                               # TÀI LIỆU SỐ LIỆU (Bàn giao cho Đạt)
│   ├── data_summary.csv                # Bảng thống kê số lượng ảnh và đối tượng từng lớp/tập
│   └── split_manifest.csv              # Nhật ký truy vết toàn bộ 800 ảnh (MD5, group_id, split)
│
└── outputs/                            # KẾT QUẢ HÌNH ẢNH MINH CHỨNG (Bàn giao cho Đạt)
    ├── label_verification/             # Hình ảnh kiểm tra nhãn (T4)
    │   ├── train/                      # Các ảnh mẫu tập Train có vẽ khung
    │   ├── val/                        # Các ảnh mẫu tập Val có vẽ khung
    │   ├── test/                       # Các ảnh mẫu tập Test có vẽ khung
    │   ├── verification_overview_grid.jpg  # [QUAN TRỌNG] Ảnh lưới ghép 12 ô tổng quan
    │   └── t4_verification_report.json     # Báo cáo kiểm định định lượng 100% dữ liệu
    │
    └── preprocessing/                  # Thực nghiệm xử lý ảnh OpenCV (T5)
        ├── opencv_experiment_report.md # [QUAN TRỌNG] Báo cáo lý thuyết & phân tích ưu/nhược
        ├── case_1_bright_light/        # Ca 1: Ảnh đủ sáng ban ngày (IoU 0.75)
        ├── case_2_low_light/           # Ca 2: Ảnh thiếu sáng/bóng râm (IoU 0.00)
        ├── case_3_small_sign/          # Ca 3: Ảnh biển nhỏ ở xa (IoU 0.00)
        ├── case_4_large_sign/          # Ca 4: Ảnh biển lớn cận cảnh (IoU 0.73)
        └── case_5_complex_background/  # Ca 5: Ảnh đường phố nền phức tạp (dương tính giả)
```

---

## 5. HƯỚNG DẪN BÀN GIAO CHI TIẾT CHO BẠN HÀO (HUẤN LUYỆN MODEL)

Bạn **Hào** cần lấy các thành phần sau để bắt đầu viết script huấn luyện `src/train.py`:

1. **Dữ liệu huấn luyện:** Nằm tại `data/processed/` (đã chia sẵn 560 train, 120 val, 120 test, định dạng chuẩn YOLO).
2. **File cấu hình YOLO:** Nạp trực tiếp file `configs/data.yaml`. Trong code huấn luyện Ultralytics YOLO:
   ```python
   from ultralytics import YOLO
   from src.paths import DATA_YAML_PATH

   model = YOLO("yolo11n.pt")
   model.train(
       data=str(DATA_YAML_PATH),
       epochs=30,
       imgsz=416,
       batch=4,
       device="cpu", # hoặc 0 nếu máy có GPU
       fliplr=0.0,   # KHÔNG lật ngang (tránh nhầm biển cấm rẽ trái/phải)
       flipud=0.0
   )
   ```
3. **Bảng nhãn:** Đọc `configs/classes.json` để ánh xạ kết quả dự đoán ra tên tiếng Việt:
   * 0: Cấm đỗ xe
   * 1: Cấm dừng và đỗ xe
   * 2: Cấm rẽ trái
   * 3: Cấm rẽ phải
4. **Cam kết chất lượng dữ liệu:**
   * 0 ảnh lỗi, 0 file nhãn lệch.
   * Tọa độ bounding box đã chuẩn hóa an toàn trong $[0, 1]$.
   * Đã chia theo cụm chuỗi video (`group_id`), bảo đảm không có tình trạng rò rỉ dữ liệu (không có khung hình liên tiếp của cùng 1 đoạn đường xuất hiện ở cả Train và Test).

---

## 6. HƯỚNG DẪN BÀN GIAO CHI TIẾT CHO BẠN ĐẠT (VIẾT BÁO CÁO & THUYẾT TRÌNH)

Bạn **Đạt** sử dụng các nguồn tài liệu và hình ảnh sau để đưa vào các mục tương ứng trong bài báo cáo Word/PDF và slide thuyết trình:

### A. Số liệu thống kê dữ liệu (Đưa vào Chương 3: Dữ liệu & Tiền xử lý)
* Mở file **`docs/data_summary.csv`** lấy bảng số liệu:
  * **Train:** 560 ảnh (420 đối tượng chia đều 4 lớp: 149 - 126 - 133 - 152).
  * **Val:** 120 ảnh (14 - 54 - 29 - 23).
  * **Test:** 120 ảnh (37 - 20 - 38 - 25).
  * **Tổng cộng:** 800 ảnh, mỗi lớp có **đúng 200 đối tượng** (phân bố cân bằng đối xứng).
* Lấy nội dung giải thích phương pháp khử trùng (MD5 Hash) và kỹ thuật gom cụm video (`group_id`) chống *Data Leakage* từ mục 6 của file này.

### B. Hình ảnh minh chứng kiểm tra nhãn Bounding Box (Đưa vào Chương 3)
* Chèn hình **`outputs/label_verification/verification_overview_grid.jpg`** vào báo cáo với chú thích:  
  *“Hình 3.x: Bảng kiểm tra trực quan nhãn dữ liệu 4 lớp biển báo trên 3 tập Train, Val, Test.”*

### C. Hình ảnh và Lý thuyết thực nghiệm OpenCV (Đưa vào Chương 2: Cơ sở lý thuyết & Chương 4: Thực nghiệm)
* Mở file **`outputs/preprocessing/opencv_experiment_report.md`**: Đã viết sẵn toàn bộ lý thuyết về không gian màu HSV, kỹ thuật lấy 2 dải đỏ, toán tử hình thái học (Morphology Open/Close), trích xuất Contour và bảng so sánh 5 trường hợp.
* Chọn các hình ảnh ghép quy trình đẹp mắt chèn vào bài:
  * `outputs/preprocessing/case_1_bright_light/pipeline_collage_Ảnh đủ sáng ban ngày (10058.jpg).jpg` (Minh họa quy trình chuẩn 6 bước: BGR $\to$ Gray $\to$ HSV $\to$ Mask $\to$ Morphology $\to$ Candidates).
  * `outputs/preprocessing/case_2_low_light/07_comparison_with_gt.jpg` (Minh họa hạn chế khi thiếu sáng).
  * `outputs/preprocessing/case_3_small_sign/07_comparison_with_gt.jpg` (Minh họa hạn chế khi biển báo quá nhỏ ở xa).
  * `outputs/preprocessing/case_5_complex_background/06_opencv_candidates.jpg` (Minh họa nhiễu do đèn xe ô tô).
* Luận điểm kết luận: *“OpenCV hiệu quả với ảnh sáng rõ, nhưng dễ bị nhiễu và bỏ sót trong điều kiện thực tế phức tạp. Do đó mô hình học sâu YOLO được lựa chọn làm giải pháp nhận diện chính.”*

---

## 7. ĐIỀU KIỆN HOÀN THÀNH VÀ CAM KẾT KỸ THUẬT CỦA TÂM

1. **Tính nhất quán:** 4 ID và tên lớp khớp 100% giữa `configs/classes.json`, `configs/data.yaml`, `data/processed/` và giao diện của Hảo.
2. **Khả năng tái lập:** Script `src/prepare_data.py` cố định random seed 42, chạy lại sẽ cho kết quả chia tập và phân bổ nhãn giống hệt 100%.
3. **Mã nguồn sạch:** Code được tổ chức module hóa, hỗ trợ đường dẫn tương đối qua `src/paths.py`, chạy được trên mọi máy tính khác nhau mà không cần sửa code.
