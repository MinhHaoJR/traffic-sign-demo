# Hướng dẫn model (Hào – L6)

`models/best.pt` là trọng số YOLO11n đã fine-tune cho **11 lớp biển báo cấm** (bản mở rộng, gộp 3 dataset). Bản 4 lớp cũ được giữ ở `models/best_4lop.pt`, số liệu cũ ở `docs/v1_4lop/`. Cả hai file (~5,5 MB) **được commit thẳng vào repo** nhờ ngoại lệ `!models/best.pt`, `!models/best_4lop.pt` trong `.gitignore`.

- **Vị trí đặt file:** `models/best.pt` (đường dẫn `MODEL_PATH` trong `src/paths.py`).
- Khi train lại và chọn model mới, ghi đè file này rồi commit lại; ghi rõ lần chạy trong commit.

## 11 lớp hỗ trợ

ID 0–3 giữ nguyên như bản 4 lớp, nên `best_4lop.pt` vẫn hiển thị đúng tên lớp.

| ID | Tên lớp |
| :-: | :-- |
| 0 | Cấm đỗ xe |
| 1 | Cấm dừng và đỗ xe |
| 2 | Cấm rẽ trái |
| 3 | Cấm rẽ phải |
| 4 | Cấm đi ngược chiều |
| 5 | Cấm quay đầu |
| 6 | Cấm rẽ trái và quay đầu |
| 7 | Cấm ô tô khách và ô tô tải |
| 8 | Cấm xe hai và ba bánh |
| 9 | Cấm mô tô và xe máy |
| 10 | Dừng lại |

Model chỉ nhận diện 11 lớp này, không phải mọi biển báo. "Cấm ô tô" và "Cấm xe tải" bị bỏ vì sau khi lọc chỉ còn 52 và 85 đối tượng sạch.

## Cấu hình / phiên bản model

Lần chạy được chọn: `run_11lop_50ep` (dòng tương ứng trong `docs/experiment_log.csv`).

| Mục | Giá trị |
| :-- | :-- |
| Trọng số khởi đầu | `yolo11n.pt` (fine-tune) |
| epochs / imgsz / batch | 50 / 640 / 16 |
| Thiết bị / seed | GPU RTX 4050 Laptop 6 GB / 42 |
| workers / cache | 0 / ram (tránh lỗi WinError 1455 thiếu bộ nhớ ảo trên Windows) |
| `fliplr` / `flipud` | 0.0 / 0.0 |
| Thời gian huấn luyện | 125,2 phút |
| Thư viện | ultralytics 8.4.191, torch 2.14.1+cu126, Python 3.14 |
| Điểm trên tập **val** (581 ảnh, 825 đối tượng) | Precision 0,898 · Recall 0,746 · mAP@0.5 0,798 · mAP@0.5:0.95 0,615 |

Dữ liệu: 3.855 ảnh gộp từ 3 dataset bằng `python -m src.prepare_merged` (2.657 train / 581 val / 617 test), chia theo cụm 25 khung liên tiếp để khung video gần giống nhau không rơi vào cả train và test. Chi tiết ở `docs/data_summary.csv`, `docs/split_manifest.csv` và mục 8 của `README.md`.

## Huấn luyện (chỉ cần một máy)

Cần `data/processed/` do Tâm bàn giao (hoặc tái tạo bằng `python -m src.prepare_data` khi đã có `data/raw`). `configs/data.yaml` tự được sinh từ `data.example.yaml` nếu chưa có.

```bash
python -m src.train                                   # chạy thử 3 epoch, đo thời gian
python -m src.train --epochs 30 --name run_30ep --save-model   # chạy đầy đủ, chép best.pt vào models/
python -m src.train --device 0 ...                    # dùng GPU NVIDIA nếu PyTorch đã hỗ trợ CUDA
```

Mỗi lần chạy thêm một dòng vào `docs/experiment_log.csv`; log, biểu đồ và checkpoint nằm ở `outputs/training/<name>/`. Chỉ số trong log là trên tập **val**. `--save-model` chỉ nên dùng cho lần chạy được chọn bằng val, không dùng test để chọn tham số.

## Chạy thử nhận diện (cho Hảo tích hợp)

```bash
python -m src.predict duong_dan_anh.jpg 0.25
python -m src.predict_video video.mp4 0.25        # lưu outputs/predictions/videos/<tên>_ket_qua.mp4
```

```python
from src.predict import load_model, predict_image

model = load_model()                                   # nạp 1 lần khi mở app
annotated_bgr, detections = predict_image(model, "anh.jpg", confidence=0.25)
# detections: [{"class_id", "class_name", "confidence", "bbox": [x1, y1, x2, y2]}, ...]; rỗng nếu không có biển
```

- Ném `FileNotFoundError` (thiếu model/ảnh) hoặc `ValueError` (ảnh hỏng) để giao diện bắt và hiển thị.
- Đọc được đường dẫn có dấu tiếng Việt; không gọi `cv2.imshow`.
- Ảnh trả về là BGR, tên lớp vẽ bằng Pillow (font Arial/Segoe UI của Windows) nên hiển thị đúng dấu.

## Đánh giá (sau khi chốt model)

```bash
python -m src.evaluate --imgsz 640 --device 0 --examples
```

Ghi `docs/test_metrics.csv` (Precision, Recall, mAP@0.5, mAP@0.5:0.95 toàn bộ và theo lớp) và lưu tối đa 3 ảnh mỗi loại đúng / sai / bỏ sót vào `outputs/predictions/examples/`. `--imgsz` phải trùng lúc huấn luyện.

## Kết quả

Đánh giá một lần trên tập **test** (617 ảnh, 825 đối tượng) bằng `python -m src.evaluate --imgsz 640 --device 0 --examples`. Nguồn: `docs/test_metrics.csv`.

| Lớp | Số đối tượng | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **Tất cả** | 825 | 0,867 | 0,756 | 0,793 | 0,621 |
| Cấm đỗ xe | 98 | 0,866 | 0,793 | 0,880 | 0,653 |
| Cấm dừng và đỗ xe | 165 | 0,939 | 0,752 | 0,906 | 0,650 |
| Cấm rẽ trái | 101 | 0,946 | 0,842 | 0,937 | 0,826 |
| Cấm rẽ phải | 115 | 0,999 | 1,000 | 0,995 | 0,815 |
| Cấm đi ngược chiều | 103 | 0,977 | 0,841 | 0,882 | 0,579 |
| Cấm quay đầu | 10 | 0,866 | 0,300 | 0,327 | 0,276 |
| Cấm rẽ trái và quay đầu | 33 | 0,994 | 0,879 | 0,903 | 0,633 |
| Cấm ô tô khách và ô tô tải | 46 | 0,000 | 0,000 | 0,002 | 0,001 |
| Cấm xe hai và ba bánh | 68 | 0,995 | 1,000 | 0,995 | 0,884 |
| Cấm mô tô và xe máy | 11 | 0,958 | 0,909 | 0,905 | 0,588 |
| Dừng lại | 75 | 0,995 | 1,000 | 0,995 | 0,928 |

Ảnh ví dụ (ngưỡng 0,25): 518 ảnh đúng, 56 ảnh sai, 43 ảnh bỏ sót, trong `outputs/predictions/examples/`.

Không so trực tiếp với bản 4 lớp (mAP@0.5 0,995): tập test mới lớn hơn, nhiều biển nhỏ ở xa, và tách theo cụm khung nên khó hơn nhiều.

## Giới hạn (cần nêu trong báo cáo)

- **"Cấm ô tô khách và ô tô tải" gần như không nhận được trên test.** Toàn bộ ảnh lớp này đến từ một dataset (dạng video), train chỉ có vài biển thật lặp lại qua nhiều khung; biển trong test là biển khác, nhỏ và mờ. Model nhận đúng trên train nhưng ở test gán nhầm sang "Cấm đỗ xe"/"Cấm dừng và đỗ xe". Cần thêm ảnh của nhiều biển khác nhau.
- **Lớp ít mẫu, số liệu không ổn định:** "Cấm quay đầu" chỉ có 10 đối tượng test, "Cấm mô tô và xe máy" 11, "Cấm rẽ trái và quay đầu" 33. Một vài ảnh đúng/sai làm điểm thay đổi lớn.
- **Có thể thiếu nhãn khi gộp:** mỗi dataset chỉ gán nhãn các lớp của nó; một ảnh có thể chứa biển thuộc lớp khác mà không được gán nhãn. Ảnh chứa nhãn mơ hồ đã bị loại, nhưng chưa rà soát hết.
- **Biển nhỏ ở xa vẫn dễ bị bỏ sót** (recall toàn bộ 0,756).
- Chỉ huấn luyện một lần với seed 42; chưa biết độ dao động giữa các lần chạy.
- Video: suy luận từng khung độc lập (không theo dõi đối tượng), khoảng 10 khung/s trên GPU với video 960×540; nhãn có thể chập chờn giữa các khung.
