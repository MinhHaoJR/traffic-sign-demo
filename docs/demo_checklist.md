# Kiểm thử ứng dụng Tkinter của Hảo

Dùng checklist này để xác nhận luồng chọn ảnh, nhận diện, hiển thị và lưu ảnh trước buổi demo. Ghi lại lỗi kèm bước tái hiện và kết quả sau khi sửa.

## Kiểm tra chức năng

| Tình huống | Cách kiểm tra | Kết quả mong đợi | Đạt |
| --- | --- | --- | :---: |
| Chưa chọn ảnh | Mở ứng dụng và quan sát các nút | Nhận diện và Lưu kết quả bị tắt; hai khung ảnh hiện hướng dẫn | [x] |
| Một biển báo | Chọn ảnh kiểm thử tổng hợp `ảnh kiểm thử.jpg` rồi nhận diện | Ảnh có khung; bảng hiển thị tên lớp, confidence và bbox | [x] |
| Nhiều biển báo | Chọn ảnh có nhiều biển rồi nhận diện | Mỗi detection có một hàng và một khung trên ảnh | [x] |
| Không có biển thuộc phạm vi | Chọn ảnh trắng `ảnh trắng.png` rồi nhận diện | Ảnh kết quả vẫn hiện; bảng rỗng; trạng thái báo không phát hiện biển | [x] |
| Ảnh không đọc được | Chọn một file không phải JPG/PNG hoặc file hỏng | Báo lỗi, ảnh đang xem không bị thay thế | [x] |
| Đường dẫn/tên ảnh có dấu | Mở ảnh có tên tiếng Việt và nhận diện | Ảnh mở và nhận diện được | [x] |
| Thiếu model | Chạy bản sao dự án không có `models/best.pt` | Giao diện xem ảnh vẫn mở; Nhận diện không khả dụng và có thông báo rõ | [x] |
| Thiếu thư viện suy luận | Mở ứng dụng trong môi trường chưa cài thư viện | Giao diện vẫn mở; nhận diện không khả dụng và có thông báo rõ | [x] |
| Đổi ảnh khi đang nhận diện | Bắt đầu nhận diện ảnh lớn, sau đó chọn ảnh khác | Kết quả của ảnh trước bị bỏ; không gắn vào ảnh mới | [x] |
| Đổi ngưỡng khi đang nhận diện | Bắt đầu nhận diện rồi kéo thanh confidence | Kết quả dùng ngưỡng cũ bị bỏ; cần chạy lại | [x] |
| Lưu PNG trong giao diện | Nhận diện rồi lưu kết quả dạng PNG | File mở được, đúng ảnh hiện tại và giao diện báo nơi lưu | [x] |
| Lưu JPG và PNG trong giao diện | Nhận diện rồi lưu từng định dạng vào đường dẫn có dấu | File mở được, đúng ảnh hiện tại và báo nơi lưu | [x] |
| Lỗi khi lưu | Chọn vị trí không có quyền ghi | Báo lỗi; ứng dụng tiếp tục hoạt động | [x] |
| Giao diện không bị đơ | Nhận diện ảnh lớn và kéo/đổi ảnh trong lúc chạy | Cửa sổ vẫn phản hồi; kết quả cũ không ghi đè kết quả hiện tại | [x] |
| Đổi kích thước cửa sổ | Chọn ảnh ngang và ảnh dọc, co giãn cửa sổ | Ảnh không méo hoặc bị cắt khỏi khung xem | [x] |

## Kiểm tra bàn giao

- [x] Cài được thư viện bằng `python -m pip install -r requirements.txt` trong `.venv` của workspace.
- [x] Chạy được `python app.py` khi `models/best.pt` đã có.
- [x] Xác nhận model chỉ nạp một lần mỗi lần mở ứng dụng.
- [x] Xác nhận tên lớp và ID khớp `configs/classes.json`.
- [ ] Gửi Đạt ảnh giao diện và video luồng chọn ảnh → nhận diện → lưu kết quả.

## Kết quả đã chạy tại workspace

- `python -m py_compile app.py` và `git diff --check` hoàn tất thành công.
- Đã mở ứng dụng với model thật, chọn ảnh có tên tiếng Việt, nhận diện một đối tượng và lưu kết quả PNG từ giao diện.
- Đã chạy ảnh trắng trong giao diện: bảng kết quả rỗng và trạng thái báo không phát hiện biển.
- Đã mở ứng dụng trước khi cài thư viện để kiểm tra trạng thái thiếu dependency; giao diện vẫn mở và báo lỗi rõ.
- Ảnh một đối tượng là ảnh tổng hợp để kiểm tra luồng tích hợp; model gán nhãn `Cấm rẽ trái` cho ảnh này. Đây không phải đánh giá độ chính xác của mô hình.
- Chưa kiểm tra cài đặt từ bản clone sạch.

### Đợt kiểm thử 2026-11-06 (có dataset)

Dataset gốc `vietnam_traffic_signs_dataset-main` được giải nén vào `data/raw/`. Chạy lại `src.prepare_data --max_samples 800 --seed 42` cho `docs/split_manifest.csv` và `docs/data_summary.csv` giống từng byte bản cũ; MD5 của cả 800 ảnh trong `data/processed` khớp manifest. `src.evaluate --examples` cho lại đúng số liệu trong `docs/test_metrics.csv` (P 0.9907, R 0.9944, mAP50 0.995, mAP50–95 0.877; ví dụ 119 đúng / 1 sai / 0 bỏ sót).

Các ca UI ở trên được chạy bằng script điều khiển `TrafficSignApp` thật (model, luồng nền, queue, widget Tk đều thật); chỉ hộp thoại chọn/lưu file và messagebox được giả lập để ghi lại nội dung. Kết quả 18/18 đạt:

- Nhiều biển báo: ảnh ghép 2×2 từ 4 ảnh test thật (mỗi ảnh một lớp) → 4 hàng, đủ 4 lớp; confidence 0.92 / 0.84 / 0.83 / 0.25. Biển `Cấm dừng và đỗ xe` chỉ vừa chạm ngưỡng 0.25 sau khi ảnh bị thu nhỏ. Bộ gốc chỉ có 2 ảnh chứa ≥2 biển thuộc 4 lớp (`6442`, `6498`), cả hai là biển rất nhỏ nên không dùng cho ca này.
- Ảnh không đọc được: JPG hỏng, file text đặt đuôi `.png`, và GIF đều báo lỗi; ảnh đang xem giữ nguyên.
- Thiếu model: kiểm bằng cách trỏ `MODEL_PATH` tới file không tồn tại (không phải clone thật); ảnh vẫn mở, nút Nhận diện tắt, trạng thái nêu rõ đường dẫn thiếu.
- Đổi ảnh / đổi ngưỡng khi đang suy luận: kết quả cũ bị bỏ, nút Lưu tắt; nhận diện lại chạy bình thường.
- Lưu JPG và PNG vào thư mục `thư mục lưu có dấu/kết quả.{jpg,png}`: file mở được, đúng kích thước ảnh kết quả.
- Lỗi khi lưu: thư mục cha là file (WinError 183) và `C:\Windows\System32` (Errno 13) đều báo lỗi; sau đó app vẫn nhận diện được.
- Không bị đơ: suy luận ảnh 6000×4500 khoảng 0.7 s trên CPU; khoảng lặng event loop lớn nhất 72 ms.
- Đổi kích thước 940×650 → 1600×1000 → 1180×820 với ảnh dọc (1:2) và ngang (3.2:1): ảnh nằm trọn trong khung, sai lệch tỷ lệ < 0.1%.
- `load_model` được gọi đúng 1 lần trong toàn bộ phiên; `model.names` khớp `configs/classes.json`.
- Ghi chú môi trường: lần suy luận đầu, Ultralytics tự cài gói `pi-heif` vào `.venv` (cần internet). Gói này không có trong `requirements.txt`.

### Đợt kiểm thử 2026-11-06 (bản 11 lớp + video)

Model `run_11lop_50ep`. Kiểm định dữ liệu gộp: 3.855 ảnh, 5.328 khung, 0 nhãn hỏng, 0 cặp ảnh/nhãn lệch.

Luồng video trong app được kiểm bằng script điều khiển `TrafficSignApp` thật với model giả (trả 1 khung ở khung chẵn); hộp thoại được giả lập. Kết quả 13/13 đạt:

- Nút Chọn video chỉ bật khi model sẵn sàng; hủy hộp thoại chọn/lưu thì không chạy gì; chặn ghi đè chính video gốc.
- Khi đang chạy: khóa Chọn ảnh, Nhận diện, Lưu, Xóa, Chọn video và thanh ngưỡng; chỉ bật Dừng video. Khung và bảng kết quả cập nhật trực tiếp.
- Video 100 khung 960×540 lưu ra đường dẫn có dấu `kết quả có dấu/video kết quả.mp4`: đủ 100 khung, cùng kích thước, mở lại được. Trạng thái báo đúng số khung và lớp xuất hiện.
- Khoảng lặng event loop lớn nhất 159 ms (không đơ).
- Dừng giữa chừng: lưu 15/100 khung, file mở được. Video hỏng báo lỗi, nút mở lại. Luồng ảnh vẫn chạy sau video. Đóng app khi đang chạy đặt cờ dừng.

Với model thật, `python -m src.predict_video` trên video thử 100 khung (ghép từ khung test của dataset 3): 9,7 s (10,3 khung/s, GPU), 69 khung có biển, lớp xuất hiện Cấm rẽ trái, Cấm rẽ trái và quay đầu, Cấm đỗ xe, Cấm dừng và đỗ xe. Luồng video trong app với model thật (chỉ giả lập hộp thoại): 100/100 khung, 19,1 khung/s, cùng số khung mỗi lớp như dòng lệnh, khoảng lặng event loop lớn nhất 99 ms, không có hộp thoại lỗi. Chưa thử bằng tay trên màn hình.

## Nhật ký lỗi

| Ngày | Bước tái hiện | Lỗi quan sát được | Cách sửa | Kết quả kiểm tra lại |
| --- | --- | --- | --- | --- |
