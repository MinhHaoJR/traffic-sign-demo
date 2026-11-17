# Hướng dẫn model (Hào – L6)

`models/best.pt` là trọng số YOLO11n đã fine-tune cho **28 lớp biển báo + 1 lớp "Biển báo khác"** (biển cấm, giới hạn tốc độ, nguy hiểm, hiệu lệnh, chỉ dẫn; gộp 3 dataset). Bản 4 lớp cũ được giữ ở `models/best_4lop.pt`, số liệu cũ ở `docs/v1_4lop/` (4 lớp), `docs/v2_11lop/` (11 lớp) và `docs/v3_28lop/` (28 lớp, chưa có lớp "Biển báo khác"). Các file này **được commit thẳng vào repo** nhờ ngoại lệ `!models/...` trong `.gitignore`.

Hai file của phần học biển mới (README mục 7):
- `models/sign_encoder.pt` (~22 MB): bộ mã hóa ResNet18 cho ảnh cắt biển báo 96×96, train bằng `python -m src.few_shot train-encoder`.
- `models/few_shot.npz`: ảnh mẫu đã mã hóa của các lớp (30 mẫu/lớp, tạo bằng `python -m src.few_shot build`) cùng các biển người dùng đã dạy trong app.

- **Vị trí đặt file:** `models/best.pt` (đường dẫn `MODEL_PATH` trong `src/paths.py`).
- Khi train lại và chọn model mới, ghi đè file này rồi commit lại; ghi rõ lần chạy trong commit.

## 29 lớp hỗ trợ

ID 0–3 giữ nguyên như bản 4 lớp, ID 0–10 như bản 11 lớp, ID 0–27 như bản 28 lớp.

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
| 28 | Biển báo khác | ds1/ds2/ds3 (các lớp nguồn không thuộc 28 lớp trên) |

Model chỉ gọi tên 28 lớp đầu, không phải mọi biển báo. Lớp 28 "Biển báo khác" giúp model khoanh được cả biển chưa có tên; app hiện các biển này là "Biển chưa biết" để người dùng dạy (README mục 7). Hai biển của ds3 ("Giới hạn chiều cao", "Chỗ ngoặt nguy hiểm vòng bên phải") được bỏ hẳn khỏi dữ liệu để làm phép thử học biển mới một cách trung thực. Các lớp 11–21 là biển của ds3 xuất hiện trong ít nhất 13 cụm 25 khung khác nhau; các lớp 22–27 lấy từ ds1 sau khi xem ảnh cắt (bỏ các lớp ds1 lẫn nhiều kiểu biển).

## Cấu hình / phiên bản model

Lần chạy được chọn: `run_29lop_50ep` (dòng tương ứng trong `docs/experiment_log.csv`).

Dòng `run_29lop_ft30` trong nhật ký đã bị **loại**: lần đó train tiếp từ model 28 lớp, nhưng model 28 lớp đã thấy một phần ảnh nay nằm trong tập test mới (tập test đổi vì thêm lớp 28), nên điểm test 0,962 bị rò rỉ. `run_29lop_50ep` train lại từ `yolo11n.pt`.

| Mục | Giá trị |
| :-- | :-- |
| Trọng số khởi đầu | `yolo11n.pt` (fine-tune) |
| epochs / imgsz / batch | 50 / 640 / 16 |
| Thiết bị / seed | GPU RTX 4050 Laptop 6 GB / 42 |
| workers / cache | 0 / disk (tránh lỗi WinError 1455 thiếu bộ nhớ ảo trên Windows) |
| `fliplr` / `flipud` | 0.0 / 0.0 |
| Thời gian huấn luyện | 178,6 phút |
| Thư viện | ultralytics 8.4.191, torch 2.14.1+cu126, Python 3.14 |
| Điểm trên tập **val** (1.332 ảnh, 1.972 đối tượng) | Precision 0,894 · Recall 0,817 · mAP@0.5 0,884 · mAP@0.5:0.95 0,699 |

Dữ liệu: 7.926 ảnh gộp từ 3 dataset bằng `python -m src.prepare_merged` (5.397 train / 1.332 val / 1.197 test), chia theo cụm 25 khung liên tiếp để khung video gần giống nhau không rơi vào cả train và test. Chi tiết ở `docs/data_summary.csv`, `docs/split_manifest.csv` và mục 5 của `README.md`.

## Huấn luyện (chỉ cần một máy)

Cần `data/processed/` (tái tạo bằng `python -m src.prepare_merged` khi đã có 3 bộ dữ liệu gốc). Khi tập chia train/val/test thay đổi, train lại từ `yolo11n.pt`, không train tiếp từ model cũ (model cũ có thể đã thấy ảnh của tập test mới). `configs/data.yaml` tự được sinh từ `data.example.yaml` nếu chưa có.

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

Đánh giá một lần trên tập **test** (1.197 ảnh, 2.219 đối tượng) bằng `python -m src.evaluate --imgsz 640 --device 0 --batch 16 --examples`. Nguồn: `docs/test_metrics.csv`.

| Lớp | Số đối tượng | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :-- | :-: | :-: | :-: | :-: | :-: |
| **Tất cả** | 2219 | 0,887 | 0,850 | 0,901 | 0,718 |
| 0. Cấm đỗ xe | 194 | 0,842 | 0,974 | 0,967 | 0,824 |
| 1. Cấm dừng và đỗ xe | 165 | 0,918 | 0,903 | 0,959 | 0,767 |
| 2. Cấm rẽ trái | 115 | 0,926 | 0,978 | 0,992 | 0,914 |
| 3. Cấm rẽ phải | 91 | 0,988 | 0,902 | 0,979 | 0,823 |
| 4. Cấm đi ngược chiều | 106 | 0,942 | 0,811 | 0,940 | 0,648 |
| 5. Cấm quay đầu | 25 | 0,986 | 0,440 | 0,559 | 0,479 |
| 6. Cấm rẽ trái và quay đầu | 35 | 0,982 | 1,000 | 0,995 | 0,710 |
| 7. Cấm ô tô khách và ô tô tải | 26 | 0,963 | 0,962 | 0,994 | 0,646 |
| 8. Cấm xe hai và ba bánh | 112 | 0,980 | 0,865 | 0,955 | 0,856 |
| 9. Cấm mô tô và xe máy | 3 | 1,000 | 0,575 | 0,912 | 0,796 |
| 10. Dừng lại | 52 | 0,987 | 1,000 | 0,995 | 0,938 |
| 11. Phải đi vòng sang bên phải | 65 | 0,887 | 0,923 | 0,983 | 0,645 |
| 12. Đi chậm | 58 | 0,683 | 0,966 | 0,877 | 0,718 |
| 13. Giới hạn tốc độ 60 km/h | 67 | 0,623 | 0,702 | 0,753 | 0,613 |
| 14. Đường người đi bộ cắt ngang | 46 | 0,938 | 0,657 | 0,896 | 0,582 |
| 15. Giới hạn tốc độ 40 km/h | 33 | 0,954 | 0,849 | 0,934 | 0,721 |
| 16. Giao nhau với đường không ưu tiên | 38 | 0,628 | 0,842 | 0,821 | 0,557 |
| 17. Biển gộp làn đường theo phương tiện | 42 | 0,773 | 0,857 | 0,864 | 0,466 |
| 18. Bến xe buýt | 20 | 0,760 | 0,750 | 0,757 | 0,492 |
| 19. Giới hạn tốc độ 80 km/h | 36 | 0,922 | 0,980 | 0,994 | 0,832 |
| 20. Đường một chiều | 41 | 0,832 | 0,512 | 0,706 | 0,421 |
| 21. Chỗ ngoặt nguy hiểm vòng bên trái | 29 | 0,735 | 0,765 | 0,833 | 0,618 |
| 22. Cấm ô tô | 109 | 0,993 | 1,000 | 0,995 | 0,973 |
| 23. Cấm ô tô rẽ trái | 25 | 0,973 | 1,000 | 0,995 | 0,995 |
| 24. Hướng đi phải theo: rẽ phải | 100 | 0,833 | 0,840 | 0,834 | 0,744 |
| 25. Hướng đi phải theo: rẽ trái | 100 | 0,993 | 1,000 | 0,995 | 0,917 |
| 26. Trẻ em | 108 | 0,985 | 1,000 | 0,995 | 0,884 |
| 27. Giao nhau với đường ưu tiên | 112 | 0,959 | 0,946 | 0,941 | 0,750 |
| 28. Biển báo khác | 266 | 0,725 | 0,639 | 0,703 | 0,503 |

18/29 lớp đạt mAP@0.5 ≥ 0,9; trung bình 28 lớp có tên là 0,908. Ảnh ví dụ (ngưỡng 0,25): 912 ảnh đúng, 222 ảnh sai, 63 ảnh bỏ sót, trong `outputs/predictions/examples/`.

Không so trực tiếp với bản 28 lớp (test mAP@0.5 0,867), 11 lớp (0,793) hay 4 lớp (0,995): mỗi bản có tập test khác nhau.

## Giới hạn (cần nêu trong báo cáo)

- **Lớp yếu nhất:** "Cấm quay đầu" (recall 0,44, chỉ 25 đối tượng test), "Đường một chiều" (0,51), "Giới hạn tốc độ 60 km/h" (mAP@0.5 0,75), "Bến xe buýt" (0,76). Các lớp ds3 lấy từ video nên train chỉ có ít biển thật lặp lại qua nhiều khung; biển trong test là biển khác.
- **"Biển báo khác" (mAP@0.5 0,70)** gom nhiều kiểu biển khác nhau nên khó hơn các lớp thường. Model chỉ cần khoanh được các biển này; tên do phần học biển mới đặt.
- **Lớp ít mẫu, số liệu không ổn định:** "Cấm mô tô và xe máy" chỉ có 3 đối tượng test, "Bến xe buýt" 20, "Cấm quay đầu" 25. Một vài ảnh đúng/sai làm điểm thay đổi lớn.
- **Lớp lấy từ ds1 có điểm rất cao (0,83–0,995)** một phần vì ảnh ds1 thường là biển to, rõ; chưa chắc giữ được mức này trên ảnh đường phố thực.
- **Có thể thiếu nhãn khi gộp:** mỗi dataset chỉ gán nhãn các lớp của nó (ví dụ khung ds3 có biển "rẽ phải" xanh mà ds3 không có lớp này). Ảnh có nhãn mơ hồ đã bị loại, nhưng chưa rà soát hết.
- **Nhầm giữa các biển tam giác cảnh báo:** trên video thử, biển "Giao nhau với đường không ưu tiên" bị gán thành "Chỗ ngoặt nguy hiểm vòng bên trái" (0,88; model 28 lớp).
- Chỉ huấn luyện một lần với seed 42; chưa biết độ dao động giữa các lần chạy.
- Video: suy luận từng khung độc lập (không theo dõi đối tượng), khoảng 18 khung/s trên GPU với video 960×540; nhãn có thể chập chờn giữa các khung.
