# DỰ ÁN NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM (TRAFFIC SIGN DEMO)

Ứng dụng nhận diện biển báo giao thông Việt Nam trên **ảnh và video** bằng YOLO11n, có giao diện Tkinter. Biển chưa được train có thể **dạy cho hệ thống ngay trong app** từ vài ảnh.

---

## 1. TỔNG QUAN

* **Chức năng:** chọn ảnh hoặc video, phát hiện vị trí biển báo, vẽ khung (bounding box), ghi tên lớp và độ tin cậy (*confidence*), lưu kết quả ra JPG/PNG hoặc MP4.
* **Phạm vi:** **28 loại biển báo** gồm biển cấm, giới hạn tốc độ, biển nguy hiểm, hiệu lệnh và chỉ dẫn, cộng lớp "Biển báo khác" để khoanh các biển chưa có tên (danh sách ở [mục 4](#4-28-lớp-biển-báo)).
* **Học biển mới:** biển lạ hiện là "Biển chưa biết"; người dùng đặt tên một lần là app nhận ra biển đó ở ảnh sau, và có thể train lại YOLO với các biển đã dạy (chi tiết ở [mục 7](#7-học-biển-mới)).
* **Dữ liệu:** gộp 3 bộ dữ liệu công khai thành 7.926 ảnh, 13.752 biển (chi tiết ở [mục 5](#5-dữ-liệu)).
* **Kết quả trên tập test** (1.197 ảnh, 2.219 biển): Precision 0,885 · Recall 0,850 · **mAP@0.5 0,908** · mAP@0.5:0.95 0,717. Trên 842 ảnh biển cận cảnh (biển to, có thể bị cắt ở mép): mAP@0.5 0,958.
* **Các phiên bản model:**

  | Bản | Số lớp | Trọng số | Số liệu |
  | --- | :-: | --- | --- |
  | Hiện tại | 28 + "Biển báo khác" | `models/best.pt` | `docs/test_metrics.csv`, [`models/README.md`](models/README.md) |
  | Bản 3 | 28 | không lưu trong repo | `docs/v3_28lop/` |
  | Bản 2 | 11 biển cấm | không lưu trong repo | `docs/v2_11lop/` |
  | Bản 1 | 4 biển cấm | `models/best_4lop.pt` | `docs/v1_4lop/` |

  ID 0–3 giống nhau ở mọi bản, ID 0–10 từ bản 2, ID 0–27 từ bản 3.

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
│   ├── classes.json            # Bảng ID -> tên lớp (29 lớp + các biển dạy trong app)
│   ├── data.example.yaml       # Cấu hình dữ liệu YOLO mẫu (đưa lên Git)
│   └── data.yaml               # Sinh tự động trên từng máy (không đưa lên Git)
├── src/
│   ├── paths.py                # Đường dẫn dùng chung
│   ├── prepare_merged.py       # Gộp 3 bộ dữ liệu thành bộ 29 lớp
│   ├── prepare_data.py         # Chuẩn bị bộ 4 lớp ban đầu (bản 1)
│   ├── check_and_visualize.py  # Kiểm định nhãn, vẽ ảnh lưới tổng quan
│   ├── preprocess.py           # Thực nghiệm xử lý ảnh OpenCV
│   ├── train.py                # Huấn luyện YOLO, ghi docs/experiment_log.csv
│   ├── evaluate.py             # Đánh giá trên tập test, ghi docs/test_metrics.csv
│   ├── predict.py              # Nhận diện ảnh: load_model, predict_image, predict_frame
│   ├── predict_video.py        # Nhận diện video, ghi MP4
│   ├── closeups.py             # Ảnh biển cận cảnh cho train và bộ test cận cảnh
│   ├── few_shot.py             # Học nhanh biển mới: bộ mã hóa ảnh cắt, ảnh mẫu, kho phản hồi
│   └── learn_new.py            # Train lại YOLO với các biển đã dạy
├── models/
│   ├── best.pt                 # Model 29 lớp (được commit)
│   ├── best_4lop.pt            # Model 4 lớp ban đầu
│   ├── sign_encoder.pt         # Bộ mã hóa ảnh cắt biển báo cho học nhanh
│   ├── few_shot.npz            # Ảnh mẫu đã mã hóa của các lớp và biển đã dạy
│   └── README.md               # Cấu hình train, kết quả từng lớp, giới hạn
├── data/                       # Dữ liệu gốc, data/processed, data/feedback (biển đã dạy); không đưa lên Git
├── outputs/                    # Log train, ảnh kiểm định, ảnh/video dự đoán (không đưa lên Git)
└── docs/
    ├── data_summary.csv        # Số đối tượng mỗi lớp theo train/val/test
    ├── split_manifest.csv      # Ảnh nào thuộc tập nào, mã hash, nhóm khung hình
    ├── experiment_log.csv      # Nhật ký các lần train
    ├── test_metrics.csv        # Điểm trên tập test, tổng và từng lớp
    ├── demo_checklist.md       # Danh sách kiểm thử giao diện
    ├── report_outline.md       # Dàn ý báo cáo
    ├── figures/                # Hình chọn cho báo cáo
    ├── v1_4lop/ … v3_28lop/    # Số liệu của các bản cũ
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

**Dạy biển chưa biết:**
1. Sau khi nhận diện ảnh, biển ngoài 28 lớp hiện là **Biển chưa biết**.
2. Chọn hàng đó trong bảng, hoặc kéo chuột vẽ khung quanh biển trên khung **Ảnh kết quả** (dùng khi YOLO không khoanh được biển).
3. Bấm **Đặt tên biển…**, chọn tên có sẵn hoặc gõ tên mới, bấm **Học biển này**.
4. App nhận diện lại ảnh; biển vừa dạy hiện tên kèm "(học nhanh)". Dạy thêm vài ảnh khác của cùng biển để nhận chắc hơn.

Có thể dùng cách này để sửa nhãn sai: chọn hàng bị gán nhầm rồi đặt đúng tên.

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
| Khác | 28 Biển báo khác (mọi biển của 3 bộ dữ liệu không thuộc 28 lớp trên) |

YOLO chỉ gọi tên 28 lớp đầu. Lớp 28 giúp model khoanh được biển chưa có tên; app hiện các biển này là "Biển chưa biết" để người dùng dạy ([mục 7](#7-học-biển-mới)). Biển lạ vẫn có thể bị gán nhầm sang lớp trông gần giống, hoặc bị bỏ sót.

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
.\.venv\Scripts\python.exe -m src.closeups build        # thêm ảnh cận cảnh (chạy sau prepare_merged)
.\.venv\Scripts\python.exe -m src.check_and_visualize   # kiểm định nhãn, ảnh lưới ở outputs/label_verification/
```

`src/prepare_merged.py` làm các bước sau. Bảng map ID nguồn → ID dự án nằm ở `SOURCES` trong file.
1. **Map ID** nhãn của từng bộ về 28 lớp, sau khi xem ảnh cắt để chỉ gộp các biển thật sự giống nhau. Các lớp nguồn còn lại (trừ nhãn gộp mơ hồ của ds2) về lớp 28 "Biển báo khác".
2. **Bỏ ảnh có nhãn mơ hồ**, ví dụ "speed limit" hay "road danger" của ds2 gộp nhiều biển. Giữ lại thì biển trong ảnh thành nền chưa gán nhãn.
3. **Khử trùng** bằng MD5.
4. **Giới hạn 700 đối tượng mỗi lớp** để các lớp nhiều ảnh của ds1 không lấn át.
5. **Chia train/val/test 70/15/15 theo cụm 25 khung hình liên tiếp**, để khung gần giống nhau của cùng một đoạn video không rơi vào cả train và test.

Lớp từ ds3 chỉ được chọn khi biển xuất hiện trong ít nhất 13 cụm khác nhau. ds3 là khung video, nên đếm theo khung sẽ đánh giá quá cao số biển thật.

Hai lớp ds3 "Giới hạn chiều cao" và "Chỗ ngoặt nguy hiểm vòng bên phải" bị **bỏ hẳn** (`HOLDOUT_DS3`): cả YOLO lẫn bộ mã hóa đều không thấy chúng, nên dùng được để thử học biển mới một cách trung thực.

| Tập | Ảnh | Đối tượng |
| --- | :-: | :-: |
| Train | 5.397 | 9.561 |
| Val | 1.332 | 1.972 |
| Test | 1.197 | 2.219 |

**Ảnh cận cảnh** (`src/closeups.py`): dữ liệu gốc gần như toàn biển nhỏ ở xa, nên model cũ bỏ sót hoặc gọi sai biển chụp gần (mAP@0.5 0,33 trên ảnh cận cảnh). `src.closeups build` cắt phóng to quanh các biển có cạnh ≥ 32 px để biển chiếm 30–75% khung, có lúc lệch hẳn ra mép (biển bị cắt, chỉ cần lộ ≥ 60%). Biển khác chỉ lộ dưới 50% trong khung cắt được tô xám để không thành biển chưa gán nhãn.
* Train: thêm 2.005 ảnh `cu_*` (tối đa 80 mỗi lớp), chỉ cắt từ ảnh train.
* Bộ test cận cảnh: 842 ảnh trong `data/processed/images/closeup_test/` (tối đa 40 mỗi lớp), chỉ cắt từ ảnh test, dùng để đo bằng `src.closeups eval`.
* Val và test gốc không đổi. Chạy lại `prepare_merged` sẽ xóa ảnh cận cảnh, nên chạy lại `closeups build` sau đó.

Kiểm định: 0 nhãn sai định dạng, 0 ảnh thiếu nhãn hoặc nhãn thiếu ảnh. `src.prepare_data` (bộ 4 lớp ban đầu) ghi vào cùng `data/processed`, nên chạy nó sẽ thay bộ 29 lớp.

---

## 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ

Cài PyTorch bản CUDA vào `.venv` để train trên GPU NVIDIA, rồi train và đánh giá:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m src.train --epochs 50 --imgsz 640 --batch 16 --device 0 --workers 0 --cache disk --name run_29lop_50ep --save-model
.\.venv\Scripts\python.exe -m src.closeups build
.\.venv\Scripts\python.exe -m src.train --model models/best.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --workers 0 --cache disk --name run_29lop_closeup_ft30 --save-model
.\.venv\Scripts\python.exe -m src.evaluate --imgsz 640 --device 0 --examples
.\.venv\Scripts\python.exe -m src.closeups eval --device 0
```

* `fliplr=0` được đặt sẵn trong `src/train.py`: lật ngang sẽ đổi biển rẽ trái thành rẽ phải.
* `--workers 0` là cố ý. Trên Windows mỗi worker nạp lại thư viện CUDA (khoảng 1,4 GB bộ nhớ ảo), nên máy 16 GB RAM bị lỗi `WinError 1455` khi dùng 4 worker. `--cache disk` lưu ảnh đã giải mã thành `.npy` để không chiếm RAM.
* `--save-model` chép trọng số tốt nhất (chọn theo tập val) vào `models/best.pt`. Mỗi lần train thêm một dòng vào `docs/experiment_log.csv`.
* Khi tập chia train/val/test thay đổi (ví dụ thêm lớp), train lại từ `yolo11n.pt`. Train tiếp từ model cũ làm điểm test bị rò rỉ, vì model cũ đã thấy một phần ảnh của tập test mới.
* Thêm ảnh cận cảnh không đổi tập chia val/test, nên được train tiếp từ model 50 epoch. Bị ngắt giữa chừng thì chạy lại lệnh với `--resume --name <tên lần chạy>`.
* `--imgsz` khi đánh giá phải trùng lúc train. `evaluate.py` ghi đè `docs/test_metrics.csv` và lưu ảnh ví dụ đúng / sai / bỏ sót vào `outputs/predictions/examples/`.

Model hiện tại (`run_29lop_closeup_ft30`) gồm 50 epoch từ `yolo11n.pt` (179 phút) và 30 epoch train tiếp có ảnh cận cảnh (khoảng 108 phút), trên RTX 4050 Laptop.

**Kết quả:**

| Model | Test thường mAP@0.5 | Test cận cảnh mAP@0.5 |
| --- | :-: | :-: |
| 50 epoch, chưa có ảnh cận cảnh | 0,901 | 0,332 |
| Hiện tại (thêm ảnh cận cảnh) | **0,908** | **0,958** |

* 19/29 lớp đạt mAP@0.5 ≥ 0,9 trên tập test; trung bình 28 lớp có tên là 0,912.
* Lớp yếu nhất:

  | Lớp | mAP@0.5 |
  | --- | :-: |
  | Đường một chiều | 0,65 |
  | Giao nhau với đường không ưu tiên | 0,70 |
  | Bến xe buýt | 0,71 |
  | Chỗ ngoặt nguy hiểm vòng bên trái | 0,75 |
  | Biển báo khác | 0,78 |

  Các lớp ds3 lấy từ video, nên tập train chỉ có ít biển thật lặp lại qua nhiều khung hình. "Biển báo khác" gom nhiều kiểu biển nên khó hơn lớp thường.
* "Cấm mô tô và xe máy" (3 đối tượng test), "Bến xe buýt" (20) và "Cấm quay đầu" (25) có ít mẫu nên điểm không ổn định.

Bảng từng lớp và các giới hạn cần nêu trong báo cáo nằm ở [`models/README.md`](models/README.md).

---

## 7. HỌC BIỂN MỚI

Khi gặp biển chưa được train, hệ thống học theo hai tầng:

| Tầng | Khi nào | Cách làm |
| --- | --- | --- |
| **Học nhanh** | Ngay khi người dùng bấm Học biển này | Lưu "dấu vân" của ảnh cắt biển vào `models/few_shot.npz`. Lần nhận diện sau, biển giống đủ mức sẽ được gọi tên. Không train lại gì. |
| **Train lại YOLO** | Khi đã dạy đủ ảnh (mặc định 20 ảnh mỗi biển mới) | `python -m src.learn_new` train tiếp YOLO với `data/processed` + `data/feedback`. Chỉ thay `models/best.pt` khi các lớp cũ không giảm quá 0,02 mAP@0.5 trên tập test. |

**Cách học nhanh hoạt động** (`src/few_shot.py`):
1. YOLO khoanh biển. Lớp "Biển báo khác" dùng ngưỡng 0,10 (thấp hơn ngưỡng chung), vì biển lạ thường được chấm điểm thấp.
2. Ảnh cắt quanh khung (96×96) đi qua bộ mã hóa ResNet18 (`models/sign_encoder.pt`), thành vector 512 chiều.
3. So độ giống (cosine) với ảnh mẫu của mọi lớp: 30 mẫu mỗi lớp có sẵn và các ảnh người dùng đã dạy.
   * Khung "Biển báo khác": giống một ảnh mẫu từ 0,88 trở lên thì lấy tên lớp đó, nếu không thì hiện **Biển chưa biết**.
   * Khung YOLO đã gán một lớp cũ: chỉ đổi tên khi một ảnh **đã dạy** giống từ 0,88 và giống hơn mọi mẫu của lớp YOLO chọn. Nhờ vậy dạy biển mới không làm hỏng các lớp cũ, và người dùng sửa được nhãn sai.
4. Mỗi lần dạy, ảnh và nhãn YOLO được lưu vào `data/feedback/` (cùng các khung khác trong ảnh làm nhãn phụ) để dùng cho `learn_new`.

Bộ mã hóa được train một lần trên ảnh cắt của cả 3 bộ dữ liệu (gồm cả các lớp nguồn không thuộc 28 lớp), nên học được cách phân biệt biển báo nói chung, không chỉ 28 lớp.

**Lệnh:**

```powershell
.\.venv\Scripts\python.exe -m src.few_shot status           # số biển đã dạy trong data/feedback
.\.venv\Scripts\python.exe -m src.learn_new                 # train lại YOLO (20 epoch) với biển đã dạy
.\.venv\Scripts\python.exe -m src.few_shot train-encoder    # chỉ cần khi train lại bộ mã hóa (cần GPU)
.\.venv\Scripts\python.exe -m src.few_shot build            # tạo lại ảnh mẫu, giữ các biển đã dạy
```

`learn_new` có `--epochs`, `--min-per-class`, `--max-drop`, `--force`. Model cũ được giữ ở `models/best_prev.pt`. Mở lại app để dùng model mới.

**Kết quả thử** với 2 biển bị bỏ hẳn khỏi dữ liệu (40 ảnh hỏi mỗi biển, chạy qua đúng code của app):

| Biển | YOLO khoanh được | Số ảnh dạy | Gọi đúng tên sau khi dạy |
| --- | :-: | :-: | :-: |
| Giới hạn chiều cao | 16/40 | 5 | 12/16 |
| Chỗ ngoặt nguy hiểm vòng bên phải | 32/40 | 10 | 29/32 |

Trên 300 ảnh test của các lớp cũ, chỉ 1/574 khung bị gán nhầm thành biển vừa dạy.

**Giới hạn:**
* Học nhanh chỉ đặt tên cho biển YOLO đã khoanh. Biển YOLO bỏ sót (thường là biển rất nhỏ) vẫn cần vẽ khung để dạy, nhưng chỉ được nhận ra tự động sau khi chạy `learn_new`.
* Trước khi dạy, biển lạ trông giống một lớp cũ có thể bị gán nhầm sang lớp đó (ví dụ "Chỗ ngoặt nguy hiểm vòng bên phải" bị gán thành "vòng bên trái"). Khi đó chọn hàng và đặt đúng tên.
* Ngưỡng 0,88 được chọn trên các phép thử trên. Ngưỡng cao hơn ít nhầm hơn nhưng phải dạy nhiều ảnh hơn.
* `learn_new` cần GPU và mất khoảng 1–1,5 giờ với 20 epoch trên RTX 4050 Laptop.

---

## 8. TƯ LIỆU CHO BÁO CÁO (ĐẠT)

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
  * Biểu đồ loss/mAP: `outputs/training/run_29lop_50ep/` và `outputs/training/run_29lop_closeup_ft30/`.
  * Ảnh cận cảnh: [mục 5](#5-dữ-liệu), kết quả ở [mục 6](#6-huấn-luyện-và-đánh-giá).
* **Hình đã chọn trong [`docs/figures/`](docs/figures):**

  | Hình | Nội dung |
  | --- | --- |
  | `fig01` | Lưới kiểm định nhãn 29 lớp × 3 tập |
  | `fig02`–`fig06` | 5 ca thực nghiệm OpenCV |
  | `fig07` | Confusion matrix trên tập test |
  | `fig08` | Đường PR |
  | `fig09` | Ví dụ nhận đúng |
  | `fig10` | Ví dụ nhận sai |
  | `fig11` | Ví dụ bỏ sót |

* **Chương giao diện:** ảnh chụp app khi nhận diện ảnh và video; quy trình chọn → nhận diện → xem bảng → lưu; quy trình dạy biển chưa biết.
* **Chương học biển mới:** [mục 7](#7-học-biển-mới) (cách hoạt động, kết quả thử, giới hạn).

---

## 9. TRẠNG THÁI

- [x] Dữ liệu 29 lớp gộp từ 3 bộ, kiểm định nhãn đạt.
- [x] Model 29 lớp, đánh giá trên tập test, ghi số liệu.
- [x] Ảnh biển cận cảnh: thêm vào train, đo trên bộ test cận cảnh (mAP@0.5 0,33 → 0,96).
- [x] Học biển mới: dạy trong app, học nhanh, train lại YOLO bằng `src.learn_new` (đã chạy thử 1 epoch trên bản sao).
- [x] Giao diện nhận diện ảnh và video; lệnh `src.predict`, `src.predict_video`.
- [x] Thực nghiệm OpenCV 5 trường hợp.
- [ ] Chạy thử nút video trong app với model 29 lớp (đã kiểm thử với model 11 lớp).
- [ ] Chạy `src.learn_new` đầy đủ với biển dạy thật (cần ≥ 20 ảnh mỗi biển).
- [ ] Báo cáo Word/PDF và slide thuyết trình.
