# Kiểm thử ứng dụng Tkinter của Hảo

Dùng checklist này để xác nhận luồng chọn ảnh, nhận diện, hiển thị và lưu ảnh trước buổi demo. Ghi lại lỗi kèm bước tái hiện và kết quả sau khi sửa.

## Kiểm tra chức năng

| Tình huống | Cách kiểm tra | Kết quả mong đợi | Đạt |
| --- | --- | --- | :---: |
| Chưa chọn ảnh | Mở ứng dụng và quan sát các nút | Nhận diện và Lưu kết quả bị tắt; hai khung ảnh hiện hướng dẫn | [x] |
| Một biển báo | Chọn ảnh kiểm thử tổng hợp `ảnh kiểm thử.jpg` rồi nhận diện | Ảnh có khung; bảng hiển thị tên lớp, confidence và bbox | [x] |
| Nhiều biển báo | Chọn ảnh có nhiều biển rồi nhận diện | Mỗi detection có một hàng và một khung trên ảnh | [ ] |
| Không có biển thuộc phạm vi | Chọn ảnh trắng `ảnh trắng.png` rồi nhận diện | Ảnh kết quả vẫn hiện; bảng rỗng; trạng thái báo không phát hiện biển | [x] |
| Ảnh không đọc được | Chọn một file không phải JPG/PNG hoặc file hỏng | Báo lỗi, ảnh đang xem không bị thay thế | [ ] |
| Đường dẫn/tên ảnh có dấu | Mở ảnh có tên tiếng Việt và nhận diện | Ảnh mở và nhận diện được | [x] |
| Thiếu model | Chạy bản sao dự án không có `models/best.pt` | Giao diện xem ảnh vẫn mở; Nhận diện không khả dụng và có thông báo rõ | [ ] |
| Thiếu thư viện suy luận | Mở ứng dụng trong môi trường chưa cài thư viện | Giao diện vẫn mở; nhận diện không khả dụng và có thông báo rõ | [x] |
| Đổi ảnh khi đang nhận diện | Bắt đầu nhận diện ảnh lớn, sau đó chọn ảnh khác | Kết quả của ảnh trước bị bỏ; không gắn vào ảnh mới | [ ] |
| Đổi ngưỡng khi đang nhận diện | Bắt đầu nhận diện rồi kéo thanh confidence | Kết quả dùng ngưỡng cũ bị bỏ; cần chạy lại | [ ] |
| Lưu PNG trong giao diện | Nhận diện rồi lưu kết quả dạng PNG | File mở được, đúng ảnh hiện tại và giao diện báo nơi lưu | [x] |
| Lưu JPG và PNG trong giao diện | Nhận diện rồi lưu từng định dạng vào đường dẫn có dấu | File mở được, đúng ảnh hiện tại và báo nơi lưu | [ ] |
| Lỗi khi lưu | Chọn vị trí không có quyền ghi | Báo lỗi; ứng dụng tiếp tục hoạt động | [ ] |
| Giao diện không bị đơ | Nhận diện ảnh lớn và kéo/đổi ảnh trong lúc chạy | Cửa sổ vẫn phản hồi; kết quả cũ không ghi đè kết quả hiện tại | [ ] |
| Đổi kích thước cửa sổ | Chọn ảnh ngang và ảnh dọc, co giãn cửa sổ | Ảnh không méo hoặc bị cắt khỏi khung xem | [ ] |

## Kiểm tra bàn giao

- [x] Cài được thư viện bằng `python -m pip install -r requirements.txt` trong `.venv` của workspace.
- [x] Chạy được `python app.py` khi `models/best.pt` đã có.
- [ ] Xác nhận model chỉ nạp một lần mỗi lần mở ứng dụng.
- [ ] Xác nhận tên lớp và ID khớp `configs/classes.json`.
- [ ] Gửi Đạt ảnh giao diện và video luồng chọn ảnh → nhận diện → lưu kết quả.

## Kết quả đã chạy tại workspace

- `python -m py_compile app.py` và `git diff --check` hoàn tất thành công.
- Đã mở ứng dụng với model thật, chọn ảnh có tên tiếng Việt, nhận diện một đối tượng và lưu kết quả PNG từ giao diện.
- Đã chạy ảnh trắng trong giao diện: bảng kết quả rỗng và trạng thái báo không phát hiện biển.
- Đã mở ứng dụng trước khi cài thư viện để kiểm tra trạng thái thiếu dependency; giao diện vẫn mở và báo lỗi rõ.
- Ảnh một đối tượng là ảnh tổng hợp để kiểm tra luồng tích hợp; model gán nhãn `Cấm rẽ trái` cho ảnh này. Đây không phải đánh giá độ chính xác của mô hình.
- Chưa kiểm tra ảnh nhiều đối tượng, file hỏng, thiếu riêng file model, thao tác đổi ảnh/ngưỡng trong lúc suy luận, lỗi lưu, resize thủ công hoặc cài đặt từ bản clone sạch.

## Nhật ký lỗi

| Ngày | Bước tái hiện | Lỗi quan sát được | Cách sửa | Kết quả kiểm tra lại |
| --- | --- | --- | --- | --- |
