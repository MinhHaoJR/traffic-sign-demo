# DỰ ÁN NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM (TRAFFIC SIGN DEMO)

Ứng dụng nhận diện biển báo giao thông Việt Nam trên **ảnh và video** bằng YOLO11n, có giao diện Tkinter.

---

## 1. TỔNG QUAN

* **Chức năng:** chọn ảnh hoặc video, phát hiện vị trí biển báo, vẽ khung (bounding box), ghi tên lớp và độ tin cậy (*confidence*), lưu kết quả ra JPG/PNG hoặc MP4.
* **Phạm vi:** **28 loại biển báo** gồm biển cấm, giới hạn tốc độ, biển nguy hiểm, hiệu lệnh và chỉ dẫn (danh sách ở [mục 4](#4-28-lớp-biển-báo)).
* **Dữ liệu:** gộp 3 bộ dữ liệu công khai thành 7.886 ảnh, 12.960 biển (chi tiết ở [mục 5](#5-dữ-liệu)).
* **Kết quả trên tập test** (1.210 ảnh, 2.041 biển): Precision 0,882 · Recall 0,812 · **mAP@0.5 0,867** · mAP@0.5:0.95 0,689.
* **Các phiên bản model:**

  | Bản | Số lớp | Trọng số | Số liệu |
  | --- | :-: | --- | --- |
  | Hiện tại | 28 | `models/best.pt` | `docs/test_metrics.csv`, [`models/README.md`](models/README.md) |
  | Bản 2 | 11 biển cấm | không lưu trong repo | `docs/v2_11lop/` |
  | Bản 1 | 4 biển cấm | `models/best_4lop.pt` | `docs/v1_4lop/` |

  ID 0–3 giống nhau ở cả 3 bản, ID 0–10 giống nhau ở bản 2 và bản hiện tại.

* **Nhóm thực hiện:**
  * **Tâm:** dữ liệu, kiểm định nhãn, thực nghiệm xử lý ảnh OpenCV.
  * **Hào:** huấn luyện YOLO, đánh giá, hàm nhận diện.
  * **Hảo:** giao diện Tkinter, tích hợp model, quản lý Git.
  * **Đạt:** tư liệu, báo cáo đồ án (Word/PDF), slide thuyết trình.

---

## 2. CẤU TRÚC THƯ MỤC

```text
traffic-sign-demo/
├── app.py                      # Giao diện Tkinter (ảnh + video)
├── requirements.txt
├── configs/
│   ├── classes.json            # Bảng ID -> tên 28 lớp
│   ├── data.example.yaml       # Cấu hình dữ liệu YOLO mẫu (đưa lên Git)
│   └── data.yaml               # Sinh tự động trên từng máy (không đưa lên Git)
├── src/
│   ├── paths.py                # Đường dẫn dùng chung
│   ├── prepare_merged.py       # Gộp 3 bộ dữ liệu thành bộ 28 lớp
│   ├── prepare_data.py         # Chuẩn bị bộ 4 lớp ban đầu (bản 1)
│   ├── check_and_visualize.py  # Kiểm định nhãn, vẽ ảnh lưới tổng quan
│   ├── preprocess.py           # Thực nghiệm xử lý ảnh OpenCV
│   ├── train.py                # Huấn luyện YOLO, ghi docs/experiment_log.csv
│   ├── evaluate.py             # Đánh giá trên tập test, ghi docs/test_metrics.csv
│   ├── predict.py              # Nhận diện ảnh: load_model, predict_image, predict_frame
│   └── predict_video.py        # Nhận diện video, ghi MP4
├── models/
│   ├── best.pt                 # Model 28 lớp (được commit)
│   ├── best_4lop.pt            # Model 4 lớp ban đầu
│   └── README.md               # Cấu hình train, kết quả từng lớp, giới hạn
├── data/                       # Dữ liệu gốc và data/processed (không đưa lên Git)
├── outputs/                    # Log train, ảnh kiểm định, ảnh/video dự đoán (không đưa lên Git)
└── docs/
    ├── data_summary.csv        # Số đối tượng mỗi lớp theo train/val/test
    ├── split_manifest.csv      # Ảnh nào thuộc tập nào, mã hash, nhóm khung hình
    ├── experiment_log.csv      # Nhật ký các lần train
    ├── test_metrics.csv        # Điểm trên tập test, tổng và từng lớp
    ├── demo_checklist.md       # Danh sách kiểm thử giao diện
    ├── report_outline.md       # Dàn ý báo cáo
    ├── figures/                # Hình chọn cho báo cáo
    ├── v1_4lop/, v2_11lop/     # Số liệu của các bản cũ
    └── report/                 # Báo cáo chính thức (Word/PDF)
```

---

## 3. CÀI ĐẶT VÀ CHẠY

### Cài đặt

Mở PowerShell ở thư mục dự án (khuyến nghị Python 3.10–3.14):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` cài PyTorch bản CPU, đủ để chạy app. Chỉ cần bản GPU khi muốn train lại (xem [mục 6](#6-huấn-luyện-và-đánh-giá)).

Lần nhận diện đầu tiên, Ultralytics có thể tự cài thêm gói `pi-heif` (cần internet).

### Giao diện

```powershell
.\.venv\Scripts\python.exe app.py
```

Model `models/best.pt` được nạp một lần khi mở app. Nếu thiếu model hoặc thư viện YOLO, app vẫn mở được để xem ảnh, nhưng các nút nhận diện bị tắt.

**Ảnh:**
1. Bấm **Chọn ảnh** (JPG/JPEG/PNG), chỉnh ngưỡng confidence rồi bấm **Nhận diện**.
2. Bảng kết quả hiện tên lớp, confidence và tọa độ khung.
3. Bấm **Lưu kết quả** để lưu ảnh JPG/PNG (mặc định `<tên_ảnh>_ket_qua.jpg`).
4. Khi đổi ảnh hoặc đổi ngưỡng, kết quả cũ bị xóa để không gắn nhầm với đầu vào mới.

**Video:**
1. Bấm **Chọn video** (MP4/AVI/MOV/MKV), rồi chọn nơi lưu video kết quả (mặc định `outputs/predictions/videos/<tên>_ket_qua.mp4`).
2. Trong lúc chạy, hai khung hiển thị khung hình gốc và khung đã nhận diện; bảng hiện các biển của khung hiện tại.
3. Bấm **Dừng video** để dừng sớm; phần đã xử lý vẫn được lưu.
4. Khi xong, dòng trạng thái báo số khung, tốc độ xử lý và các biển đã xuất hiện.
5. Trong lúc chạy, các nút ảnh và thanh ngưỡng bị khóa; ngưỡng dùng là giá trị lúc bấm Chọn video.

Các ca kiểm thử giao diện và kết quả nằm trong [`docs/demo_checklist.md`](docs/demo_checklist.md).

### Dòng lệnh

```powershell
.\.venv\Scripts\python.exe -m src.predict "duong_dan_anh.jpg" 0.25
.\.venv\Scripts\python.exe -m src.predict_video "duong_dan_video.mp4" 0.25 --out "ket_qua.mp4"
```

Dùng trong code Python:

```python
from src.predict import load_model, predict_image

model = load_model()  # nạp 1 lần
annotated_bgr, detections = predict_image(model, "anh.jpg", confidence=0.25)
# detections: [{"class_id", "class_name", "confidence", "bbox": [x1, y1, x2, y2]}, ...]
```

---

## 4. 28 LỚP BIỂN BÁO

Bảng đầy đủ nằm ở [`configs/classes.json`](configs/classes.json); nguồn và điểm từng lớp nằm ở [`models/README.md`](models/README.md).

| Nhóm | ID và tên lớp |
| --- | --- |
| Biển cấm | 0 Cấm đỗ xe · 1 Cấm dừng và đỗ xe · 2 Cấm rẽ trái · 3 Cấm rẽ phải · 4 Cấm đi ngược chiều · 5 Cấm quay đầu · 6 Cấm rẽ trái và quay đầu · 7 Cấm ô tô khách và ô tô tải · 8 Cấm xe hai và ba bánh · 9 Cấm mô tô và xe máy · 22 Cấm ô tô · 23 Cấm ô tô rẽ trái |
| Giới hạn tốc độ | 13 Tốc độ 60 km/h · 15 Tốc độ 40 km/h · 19 Tốc độ 80 km/h |
| Nguy hiểm, cảnh báo | 12 Đi chậm · 14 Đường người đi bộ cắt ngang · 16 Giao nhau với đường không ưu tiên · 21 Chỗ ngoặt nguy hiểm vòng bên trái · 26 Trẻ em · 27 Giao nhau với đường ưu tiên |
| Hiệu lệnh | 10 Dừng lại · 11 Phải đi vòng sang bên phải · 24 Hướng đi phải theo: rẽ phải · 25 Hướng đi phải theo: rẽ trái |
| Chỉ dẫn | 17 Biển gộp làn đường theo phương tiện · 18 Bến xe buýt · 20 Đường một chiều |

Model chỉ nhận diện 28 lớp này. Biển khác cho kết quả rỗng hoặc bị gán nhầm sang lớp gần giống.

---

## 5. DỮ LIỆU

Giải nén 3 bộ dữ liệu vào `data/` (không đưa lên Git):

| Thư mục | Bộ dữ liệu | Ghi chú |
| --- | --- | --- |
| `data/raw/` | `vietnam_traffic_signs_dataset-main.zip` (ds1, 16 lớp) | Ảnh biển to, rõ |
| `data/raw_ds2/` | `Vietnamese traffic signs detection and recognition.zip` (ds2, 29 lớp) | Có nhãn gộp nhiều loại biển |
| `data/raw_ds3/` | `Vietnam Traffic Signs.zip` (ds3, 52 lớp) | Khung hình liên tiếp từ video |

```powershell
.\.venv\Scripts\python.exe -m src.prepare_merged        # ghi data/processed, docs/data_summary.csv, docs/split_manifest.csv, configs/*.yaml
.\.venv\Scripts\python.exe -m src.check_and_visualize   # kiểm định nhãn, ảnh lưới ở outputs/label_verification/
```

`src/prepare_merged.py` làm các bước sau. Bảng map ID nguồn → ID dự án nằm ở `SOURCES` trong file.
1. **Map ID** nhãn của từng bộ về 28 lớp, sau khi xem ảnh cắt để chỉ gộp các biển thật sự giống nhau.
2. **Bỏ ảnh có nhãn mơ hồ**, ví dụ "speed limit" hay "road danger" của ds2 gộp nhiều biển. Giữ lại thì biển trong ảnh thành nền chưa gán nhãn.
3. **Khử trùng** bằng MD5.
4. **Giới hạn 700 đối tượng mỗi lớp** để các lớp nhiều ảnh của ds1 không lấn át.
5. **Chia train/val/test 70/15/15 theo cụm 25 khung hình liên tiếp**, để khung gần giống nhau của cùng một đoạn video không rơi vào cả train và test.

Lớp từ ds3 chỉ được chọn khi biển xuất hiện trong ít nhất 13 cụm khác nhau. ds3 là khung video, nên đếm theo khung sẽ đánh giá quá cao số biển thật.

| Tập | Ảnh | Đối tượng |
| --- | :-: | :-: |
| Train | 5.403 | 8.961 |
| Val | 1.273 | 1.958 |
| Test | 1.210 | 2.041 |

Kiểm định: 0 nhãn sai định dạng, 0 ảnh thiếu nhãn hoặc nhãn thiếu ảnh. `src.prepare_data` (bộ 4 lớp ban đầu) ghi vào cùng `data/processed`, nên chạy nó sẽ thay bộ 28 lớp.

---

## 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ

Cài PyTorch bản CUDA vào `.venv` để train trên GPU NVIDIA, rồi train và đánh giá:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m src.train --epochs 50 --imgsz 640 --batch 16 --device 0 --workers 0 --cache disk --name run_28lop_50ep --save-model
.\.venv\Scripts\python.exe -m src.evaluate --imgsz 640 --device 0 --examples
```

* `fliplr=0` được đặt sẵn trong `src/train.py`: lật ngang sẽ đổi biển rẽ trái thành rẽ phải.
* `--workers 0` là cố ý. Trên Windows mỗi worker nạp lại thư viện CUDA (khoảng 1,4 GB bộ nhớ ảo), nên máy 16 GB RAM bị lỗi `WinError 1455` khi dùng 4 worker. `--cache disk` lưu ảnh đã giải mã thành `.npy` để không chiếm RAM.
* `--save-model` chép trọng số tốt nhất (chọn theo tập val) vào `models/best.pt`. Mỗi lần train thêm một dòng vào `docs/experiment_log.csv`.
* `--imgsz` khi đánh giá phải trùng lúc train. `evaluate.py` ghi đè `docs/test_metrics.csv` và lưu ảnh ví dụ đúng / sai / bỏ sót vào `outputs/predictions/examples/`.

Lần train hiện tại (`run_28lop_50ep`) chạy trên RTX 4050 Laptop, mất 112 phút.

**Kết quả:**
* 16/28 lớp đạt mAP@0.5 ≥ 0,9 trên tập test.
* Lớp yếu nhất:

  | Lớp | mAP@0.5 |
  | --- | :-: |
  | Đường một chiều | 0,32 |
  | Cấm ô tô khách và ô tô tải | 0,62 |
  | Tốc độ 80 km/h | 0,64 |
  | Đi chậm | 0,77 |
  | Tốc độ 60 km/h | 0,77 |

  Các lớp này lấy từ ds3, nên tập train chỉ có ít biển thật lặp lại qua nhiều khung hình.
* "Cấm mô tô và xe máy" (3 đối tượng test) và "Cấm quay đầu" (13) có quá ít mẫu nên điểm không ổn định.

Bảng từng lớp và các giới hạn cần nêu trong báo cáo nằm ở [`models/README.md`](models/README.md).

---

## 7. TƯ LIỆU CHO BÁO CÁO (ĐẠT)

* **Dàn ý:** [`docs/report_outline.md`](docs/report_outline.md).
* **Chương dữ liệu:**
  * Số liệu: [`docs/data_summary.csv`](docs/data_summary.csv) và [mục 5](#5-dữ-liệu).
  * Cách chống rò rỉ dữ liệu: chia theo cụm khung hình, cột `group_id` trong [`docs/split_manifest.csv`](docs/split_manifest.csv).
* **Chương thực nghiệm OpenCV:**
  * Báo cáo đầy đủ: `outputs/preprocessing/opencv_experiment_report.md` (sinh bằng `python -m src.preprocess`). Báo cáo giải thích HSV, cách gộp 2 dải màu đỏ, và phép mở/đóng với kernel elip.
  * Kết quả 5 trường hợp:

    | Trường hợp | Kết quả |
    | --- | --- |
    | Đủ sáng | IoU 0,75 |
    | Thiếu sáng | IoU 0 |
    | Biển nhỏ ở xa | IoU 0 |
    | Biển lớn | IoU 0,73 |
    | Nền phức tạp | 10 dương tính giả (bắt nhầm đèn xe màu đỏ) |

  * Kết luận: OpenCV nhanh nhưng dễ nhiễu và không đọc được ký hiệu trong biển, nên YOLO phù hợp hơn cho ứng dụng hoàn chỉnh.
* **Chương kết quả mô hình:**
  * Bảng điểm: [`docs/test_metrics.csv`](docs/test_metrics.csv), [`models/README.md`](models/README.md).
  * Nhật ký train: [`docs/experiment_log.csv`](docs/experiment_log.csv).
  * Biểu đồ loss/mAP: `outputs/training/run_28lop_50ep/`.
* **Hình đã chọn trong [`docs/figures/`](docs/figures):**

  | Hình | Nội dung |
  | --- | --- |
  | `fig01` | Lưới kiểm định nhãn 28 lớp × 3 tập |
  | `fig02`–`fig06` | 5 ca thực nghiệm OpenCV |
  | `fig07` | Confusion matrix trên tập test |
  | `fig08` | Đường PR |
  | `fig09` | Ví dụ nhận đúng |
  | `fig10` | Ví dụ nhận sai |
  | `fig11` | Ví dụ bỏ sót |

* **Chương giao diện:** ảnh chụp app khi nhận diện ảnh và video; quy trình chọn → nhận diện → xem bảng → lưu.

---

## 8. TRẠNG THÁI

- [x] Dữ liệu 28 lớp gộp từ 3 bộ, kiểm định nhãn đạt.
- [x] Model 28 lớp, đánh giá trên tập test, ghi số liệu.
- [x] Giao diện nhận diện ảnh và video; lệnh `src.predict`, `src.predict_video`.
- [x] Thực nghiệm OpenCV 5 trường hợp.
- [ ] Chạy thử nút video trong app với model 28 lớp (đã kiểm thử với model 11 lớp).
- [ ] Báo cáo Word/PDF và slide thuyết trình.
