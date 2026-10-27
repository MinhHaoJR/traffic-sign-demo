# Hướng dẫn model (Hào – L6)

`models/best.pt` là trọng số YOLO11n đã fine-tune cho 4 lớp biển báo cấm. File này (~5,5 MB) **được commit thẳng vào repo**: `.gitignore` chặn `*.pt` nhưng có ngoại lệ `!models/best.pt`, nên clone về là dùng được, không cần tải riêng. Các file `.pt` khác (checkpoint trong `outputs/`, `yolo11n.pt`) vẫn bị bỏ qua.

- **Vị trí đặt file:** `models/best.pt` (đường dẫn `MODEL_PATH` trong `src/paths.py`).
- Khi train lại và chọn model mới, ghi đè file này rồi commit lại; ghi rõ lần chạy trong commit.

## 4 lớp hỗ trợ

| ID | Tên lớp |
| :-: | :-- |
| 0 | Cấm đỗ xe |
| 1 | Cấm dừng và đỗ xe |
| 2 | Cấm rẽ trái |
| 3 | Cấm rẽ phải |

Model chỉ nhận diện 4 lớp này, không phải mọi biển báo.

## Cấu hình / phiên bản model

Lần chạy được chọn: `run_30ep` (dòng tương ứng trong `docs/experiment_log.csv`; lần `smoke_3ep` chỉ là chạy thử).

| Mục | Giá trị |
| :-- | :-- |
| Trọng số khởi đầu | `yolo11n.pt` (fine-tune) |
| epochs / imgsz / batch | 30 / 416 / 4 |
| Thiết bị / seed | CPU (Intel Core i5-12500H) / 42 |
| `fliplr` / `flipud` | 0.0 / 0.0 |
| Thời gian huấn luyện | 65,4 phút |
| Epoch tốt nhất (chọn bằng val) | 28 |
| Thư viện | ultralytics 8.4.190, torch 2.14.1+cpu, Python 3.14 |
| Điểm trên tập **val** (120 ảnh) | Precision 0,9815 · Recall 0,9985 · mAP@0.5 0,9949 · mAP@0.5:0.95 0,8073 |

Dữ liệu: 800 ảnh do Tâm chuẩn bị (560 train / 120 val / 120 test, seed 42), khớp `docs/split_manifest.csv`.

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
python -m src.evaluate --examples
```

Ghi `docs/test_metrics.csv` (Precision, Recall, mAP@0.5, mAP@0.5:0.95 toàn bộ và theo lớp) và lưu tối đa 3 ảnh mỗi loại đúng / sai / bỏ sót vào `outputs/predictions/examples/`. `--imgsz` phải trùng lúc huấn luyện.

## Kết quả

Đánh giá một lần trên tập **test** (120 ảnh, 120 đối tượng) bằng `python -m src.evaluate --examples`, `imgsz` 416. Nguồn: `docs/test_metrics.csv`.

| Lớp | Số đối tượng | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **Tất cả** | 120 | 0,991 | 0,994 | 0,995 | 0,877 |
| Cấm đỗ xe | 37 | 0,997 | 1,000 | 0,995 | 0,841 |
| Cấm dừng và đỗ xe | 20 | 1,000 | 1,000 | 0,995 | 0,768 |
| Cấm rẽ trái | 38 | 1,000 | 0,978 | 0,995 | 0,952 |
| Cấm rẽ phải | 25 | 0,967 | 1,000 | 0,995 | 0,948 |

Ảnh ví dụ (ngưỡng 0,25): 119 ảnh đúng, 1 ảnh sai, 0 ảnh bỏ sót. Ảnh sai là `11722.jpg`: nhận đúng lớp "Cấm đỗ xe" (0,89) nhưng thừa thêm một khung trùng điểm 0,30. Ảnh nằm trong `outputs/predictions/examples/`.

## Giới hạn (cần nêu trong báo cáo)

- **Điểm test có thể quá lạc quan.** Nhóm chỉ loại ảnh trùng hoàn toàn bằng hash và gom ảnh theo cụm số thứ tự file. Ảnh gần giống (cùng cảnh, bản xoay/đổi sáng) có thể vẫn nằm ở cả train và test; chưa rà soát đầy đủ. Mỗi ảnh chỉ chứa 1 biển, test chỉ có 120 ảnh.
- **Biển nhỏ ở xa dễ bị bỏ sót.** Thử trên ảnh `6442` của bộ gốc (2 biển "cấm dừng và đỗ xe" rộng khoảng 3% ảnh): ở `imgsz` 416 model không phát hiện biển nào với ngưỡng 0,25. Chạy suy luận ở `imgsz` 1280 thì thấy cả hai vị trí nhưng độ tin cậy chỉ 0,53 và 0,37, và một biển bị nhận nhầm sang "Cấm đỗ xe". Nếu cần cải thiện, huấn luyện lại với `imgsz` 640.
- Chỉ nhận diện 4 lớp; biển khác (ví dụ ảnh `0` của bộ gốc chứa biển ngoài phạm vi) cho kết quả rỗng, không phải lỗi.
- Huấn luyện một lần với seed 42 trên CPU; chưa lặp lại nhiều seed nên chưa biết độ dao động.
