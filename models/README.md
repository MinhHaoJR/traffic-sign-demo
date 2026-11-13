# Hướng dẫn model (Hào – L6)

`models/best.pt` là trọng số YOLO11n đã fine-tune cho **28 lớp biển báo** (biển cấm, giới hạn tốc độ, nguy hiểm, hiệu lệnh, chỉ dẫn; gộp 3 dataset). Bản 4 lớp cũ được giữ ở `models/best_4lop.pt`, số liệu cũ ở `docs/v1_4lop/` (4 lớp) và `docs/v2_11lop/` (11 lớp). Cả hai file (~5,5 MB) **được commit thẳng vào repo** nhờ ngoại lệ `!models/best.pt`, `!models/best_4lop.pt` trong `.gitignore`.

- **Vị trí đặt file:** `models/best.pt` (đường dẫn `MODEL_PATH` trong `src/paths.py`).
- Khi train lại và chọn model mới, ghi đè file này rồi commit lại; ghi rõ lần chạy trong commit.

## 28 lớp hỗ trợ

ID 0–3 giữ nguyên như bản 4 lớp, ID 0–10 giữ nguyên như bản 11 lớp.

| ID | Tên lớp | Nguồn |
| :-: | :-- | :-- |
| 0 | Cấm đỗ xe | ds1/ds2/ds3 |
| 1 | Cấm dừng và đỗ xe | ds1/ds2/ds3 |
| 2 | Cấm rẽ trái | ds1/ds2/ds3 |
| 3 | Cấm rẽ phải | ds1/ds2/ds3 |
| 4 | Cấm đi ngược chiều | ds1/ds2/ds3 |
| 5 | Cấm quay đầu | ds1/ds2/ds3 |
| 6 | Cấm rẽ trái và quay đầu | ds1/ds2/ds3 |
| 7 | Cấm ô tô khách và ô tô tải | ds1/ds2/ds3 |
| 8 | Cấm xe hai và ba bánh | ds1/ds2/ds3 |
| 9 | Cấm mô tô và xe máy | ds1/ds2/ds3 |
| 10 | Dừng lại | ds1/ds2/ds3 |
| 11 | Phải đi vòng sang bên phải | ds3 |
| 12 | Đi chậm | ds3 + ds2 |
| 13 | Giới hạn tốc độ 60 km/h | ds3 |
| 14 | Đường người đi bộ cắt ngang | ds3 |
| 15 | Giới hạn tốc độ 40 km/h | ds3 |
| 16 | Giao nhau với đường không ưu tiên | ds3 |
| 17 | Biển gộp làn đường theo phương tiện | ds3 + ds2 |
| 18 | Bến xe buýt | ds3 |
| 19 | Giới hạn tốc độ 80 km/h | ds3 |
| 20 | Đường một chiều | ds3 |
| 21 | Chỗ ngoặt nguy hiểm vòng bên trái | ds3 |
| 22 | Cấm ô tô | ds1 + ds3 |
| 23 | Cấm ô tô rẽ trái | ds1 |
| 24 | Hướng đi phải theo: rẽ phải | ds1 |
| 25 | Hướng đi phải theo: rẽ trái | ds1 |
| 26 | Trẻ em | ds1 + ds3 |
| 27 | Giao nhau với đường ưu tiên | ds1 |

Model chỉ nhận diện 28 lớp này, không phải mọi biển báo. Các lớp 11–21 là biển của ds3 xuất hiện trong ít nhất 13 cụm 25 khung khác nhau; các lớp 22–27 lấy từ ds1 sau khi xem ảnh cắt (bỏ các lớp ds1 lẫn nhiều kiểu biển).

## Cấu hình / phiên bản model

Lần chạy được chọn: `run_28lop_50ep` (dòng tương ứng trong `docs/experiment_log.csv`).

| Mục | Giá trị |
| :-- | :-- |
| Trọng số khởi đầu | `yolo11n.pt` (fine-tune) |
| epochs / imgsz / batch | 50 / 640 / 16 |
| Thiết bị / seed | GPU RTX 4050 Laptop 6 GB / 42 |
| workers / cache | 0 / disk (tránh lỗi WinError 1455 thiếu bộ nhớ ảo trên Windows) |
| `fliplr` / `flipud` | 0.0 / 0.0 |
| Thời gian huấn luyện | khoảng 135 phút (bị ngắt ở epoch 8, chạy tiếp bằng `--resume`; nhật ký chỉ ghi 112,3 phút phần chạy tiếp) |
| Thư viện | ultralytics 8.4.191, torch 2.14.1+cu126, Python 3.14 |
| Điểm trên tập **val** (1.273 ảnh, 1.958 đối tượng) | Precision 0,903 · Recall 0,871 · mAP@0.5 0,912 · mAP@0.5:0.95 0,723 |

Dữ liệu: 7.886 ảnh gộp từ 3 dataset bằng `python -m src.prepare_merged` (5.403 train / 1.273 val / 1.210 test), chia theo cụm 25 khung liên tiếp để khung video gần giống nhau không rơi vào cả train và test. Chi tiết ở `docs/data_summary.csv`, `docs/split_manifest.csv` và mục 8 của `README.md`.

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

Đánh giá một lần trên tập **test** (1.210 ảnh, 2041 đối tượng) bằng `python -m src.evaluate --imgsz 640 --device 0 --examples`. Nguồn: `docs/test_metrics.csv`.

| Lớp | Số đối tượng | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **Tất cả** | 2041 | 0,882 | 0,812 | 0,867 | 0,689 |
| 0. Cấm đỗ xe | 206 | 0,977 | 0,819 | 0,929 | 0,767 |
| 1. Cấm dừng và đỗ xe | 184 | 0,836 | 0,891 | 0,855 | 0,661 |
| 2. Cấm rẽ trái | 119 | 0,933 | 0,874 | 0,910 | 0,837 |
| 3. Cấm rẽ phải | 103 | 0,963 | 0,942 | 0,985 | 0,861 |
| 4. Cấm đi ngược chiều | 106 | 0,917 | 0,689 | 0,795 | 0,489 |
| 5. Cấm quay đầu | 13 | 1,000 | 0,804 | 0,928 | 0,670 |
| 6. Cấm rẽ trái và quay đầu | 50 | 0,990 | 1,000 | 0,995 | 0,658 |
| 7. Cấm ô tô khách và ô tô tải | 48 | 0,895 | 0,458 | 0,619 | 0,364 |
| 8. Cấm xe hai và ba bánh | 117 | 0,833 | 0,914 | 0,942 | 0,758 |
| 9. Cấm mô tô và xe máy | 3 | 0,514 | 0,667 | 0,775 | 0,686 |
| 10. Dừng lại | 66 | 0,991 | 1,000 | 0,995 | 0,899 |
| 11. Phải đi vòng sang bên phải | 58 | 0,891 | 0,707 | 0,786 | 0,478 |
| 12. Đi chậm | 35 | 0,552 | 0,740 | 0,766 | 0,603 |
| 13. Giới hạn tốc độ 60 km/h | 96 | 0,794 | 0,750 | 0,768 | 0,541 |
| 14. Đường người đi bộ cắt ngang | 37 | 0,948 | 1,000 | 0,982 | 0,728 |
| 15. Giới hạn tốc độ 40 km/h | 38 | 1,000 | 0,732 | 0,941 | 0,803 |
| 16. Giao nhau với đường không ưu tiên | 34 | 0,960 | 0,703 | 0,881 | 0,705 |
| 17. Biển gộp làn đường theo phương tiện | 57 | 0,944 | 0,754 | 0,798 | 0,653 |
| 18. Bến xe buýt | 20 | 0,949 | 0,937 | 0,969 | 0,726 |
| 19. Giới hạn tốc độ 80 km/h | 29 | 0,620 | 0,517 | 0,637 | 0,449 |
| 20. Đường một chiều | 31 | 0,750 | 0,226 | 0,324 | 0,170 |
| 21. Chỗ ngoặt nguy hiểm vòng bên trái | 20 | 0,599 | 0,750 | 0,802 | 0,568 |
| 22. Cấm ô tô | 112 | 0,994 | 1,000 | 0,995 | 0,967 |
| 23. Cấm ô tô rẽ trái | 25 | 0,976 | 1,000 | 0,995 | 0,938 |
| 24. Hướng đi phải theo: rẽ phải | 100 | 0,909 | 0,900 | 0,924 | 0,827 |
| 25. Hướng đi phải theo: rẽ trái | 100 | 0,994 | 1,000 | 0,995 | 0,913 |
| 26. Trẻ em | 113 | 0,995 | 1,000 | 0,995 | 0,860 |
| 27. Giao nhau với đường ưu tiên | 121 | 0,983 | 0,963 | 0,985 | 0,713 |

Ảnh ví dụ (ngưỡng 0,25): 949 ảnh đúng, 157 ảnh sai, 104 ảnh bỏ sót, trong `outputs/predictions/examples/`.

Không so trực tiếp với bản 11 lớp (test mAP@0.5 0,793) hay 4 lớp (0,995): mỗi bản có tập test khác nhau.

## Giới hạn (cần nêu trong báo cáo)

- **Lớp yếu nhất:** "Đường một chiều" (recall 0,23), "Cấm ô tô khách và ô tô tải" (0,46), "Giới hạn tốc độ 80 km/h" (0,52). Các lớp này lấy từ ds3 (dạng video) nên train chỉ có ít biển thật lặp lại qua nhiều khung; biển trong test là biển khác.
- **Lớp ít mẫu, số liệu không ổn định:** "Cấm mô tô và xe máy" chỉ có 3 đối tượng test, "Cấm quay đầu" 13. Một vài ảnh đúng/sai làm điểm thay đổi lớn.
- **Lớp lấy từ ds1 có điểm rất cao (0,92–0,995)** một phần vì ảnh ds1 thường là biển to, rõ; chưa chắc giữ được mức này trên ảnh đường phố thực.
- **Có thể thiếu nhãn khi gộp:** mỗi dataset chỉ gán nhãn các lớp của nó (ví dụ khung ds3 có biển "rẽ phải" xanh mà ds3 không có lớp này). Ảnh có nhãn mơ hồ đã bị loại, nhưng chưa rà soát hết.
- **Nhầm giữa các biển tam giác cảnh báo:** trên video thử, biển "Giao nhau với đường không ưu tiên" bị gán thành "Chỗ ngoặt nguy hiểm vòng bên trái" (0,88).
- Chỉ huấn luyện một lần với seed 42; chưa biết độ dao động giữa các lần chạy.
- Video: suy luận từng khung độc lập (không theo dõi đối tượng), khoảng 18 khung/s trên GPU với video 960×540; nhãn có thể chập chờn giữa các khung.
