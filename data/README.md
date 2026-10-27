# Hướng Dẫn & Quản Lý Dữ Liệu (Tâm)

## 1. Nguồn Dữ Liệu Gốc (Raw Data)
* **Tên bộ dữ liệu:** `vietnam_traffic_signs_dataset-main` (tập 16 lớp biển báo giao thông Việt Nam theo chuẩn YOLO).
* **Vị trí lưu trữ gốc:** `data/raw/` (liên kết hoặc giải nén từ `vietnam_traffic_signs_dataset-main.zip`).
* **Lưu ý:** Thư mục `data/raw/` được cấu hình trong `.gitignore`, không commit lên GitHub.

## 2. Phạm Vi 4 Lớp Biển Báo Lựa Chọn
| ID Dự án | Tên Biển Báo | ID Gốc (16 lớp) | Tên Nhãn Gốc |
| :---: | :--- | :---: | :--- |
| **0** | Cấm đỗ xe | 3 | `no parking` |
| **1** | Cấm dừng và đỗ xe | 2 | `no stop or parking` |
| **2** | Cấm rẽ trái | 11 | `no turn left` |
| **3** | Cấm rẽ phải | 12 | `no turn right` |

## 3. Lệnh Tiền Xử Lý Dữ Liệu
Chạy lệnh từ thư mục gốc của dự án:
```bash
python -m src.prepare_data
```
Hoặc tùy chỉnh các tham số (ví dụ số lượng ảnh hoặc random seed):
```bash
python -m src.prepare_data --max_samples 800 --seed 42
```

## 4. Cấu Trúc Dữ Liệu Đầu Ra (data/processed/)
Sau khi chạy script `src/prepare_data.py`, dữ liệu sạch được tổ chức như sau:
```
data/processed/
├── images/
│   ├── train/   (560 ảnh - 70%)
│   ├── val/     (120 ảnh - 15%)
│   └── test/    (120 ảnh - 15%)
└── labels/
    ├── train/   (560 file nhãn YOLO)
    ├── val/     (120 file nhãn YOLO)
    └── test/    (120 file nhãn YOLO)
```

## 5. Bảng Thống Kê & Truy Vết
* **`docs/data_summary.csv`**: Bảng thống kê chi tiết số lượng ảnh duy nhất và số đối tượng từng lớp trên từng tập (*Train/Val/Test*).
* **`docs/split_manifest.csv`**: Danh sách truy vết toàn bộ ảnh gồm nguồn gốc, mã băm MD5, group sequence chuỗi video và tập được phân bổ.

## 6. Lưu Ý Kỹ Thuật
1. **Chống rò rỉ dữ liệu (Data Leakage):** Các khung hình video liên tiếp thuộc cùng một chuỗi quay được gom vào cụm (`group_id`) và phân bổ toàn bộ vào cùng một tập, tuyệt đối không chia cắt cùng một chuỗi sang cả Train và Test.
2. **Khử trùng lặp:** Kiểm tra mã băm MD5 trước khi phân tập, loại bỏ hoàn toàn các ảnh trùng lặp bit-by-bit.
3. **Chuẩn hóa tọa độ:** Bounding box được kiểm tra nằm hoàn toàn trong giới hạn $[0.0, 1.0]$.
