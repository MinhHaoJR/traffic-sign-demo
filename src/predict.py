"""
[Hào] Hàm nhận diện cho giao diện Tkinter (Hảo) gọi.

    model = load_model()                         # nạp 1 lần khi mở app
    annotated_bgr, detections = predict_image(model, "anh.jpg", 0.25)

detections: list dict {class_id, class_name, confidence, bbox=[x1, y1, x2, y2]}; rỗng nếu không có biển.
Lỗi file/model ném FileNotFoundError / ValueError để giao diện bắt và hiển thị.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.paths import CLASSES_PATH, MODEL_PATH

# BGR cho khung; mỗi lớp một màu để dễ phân biệt
COLORS_BGR = {0: (255, 144, 30), 1: (60, 20, 220), 2: (0, 140, 255), 3: (87, 139, 46)}
_FONT_CANDIDATES = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]  # cần font có dấu tiếng Việt


def _class_names() -> dict:
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _font(size: int):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()  # không có dấu nhưng không làm app sập


def load_model(model_path=MODEL_PATH):
    """Nạp trọng số YOLO (chỉ gọi 1 lần khi mở app)."""
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {model_path}. Tải best.pt theo models/README.md."
        )
    return YOLO(str(model_path))


def _read_bgr(image_path) -> np.ndarray:
    # np.fromfile + imdecode để đọc được đường dẫn có dấu tiếng Việt trên Windows
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh (file hỏng hoặc sai định dạng): {path}")
    return img


def _draw(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Vẽ khung + 'Tên 0.89' bằng Pillow để hiển thị đúng dấu tiếng Việt."""
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(max(14, img.width // 45))
    for d in detections:
        b, g, r = COLORS_BGR.get(d["class_id"], (0, 255, 0))
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=max(2, img.width // 300))
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        th = ty2 - ty1 + 6
        top = y1 - th if y1 - th >= 0 else y1  # nhãn nằm trên khung, hết chỗ thì xuống trong khung
        draw.rectangle([x1, top, x1 + (tx2 - tx1) + 8, top + th], fill=(r, g, b))
        draw.text((x1 + 4, top + 2 - ty1), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def predict_image(model, image_path, confidence: float = 0.25):
    """Trả về (annotated_bgr, detections). Không mở cửa sổ cv2.imshow."""
    img = _read_bgr(image_path)
    names = _class_names()
    result = model.predict(source=img, conf=confidence, verbose=False)[0]

    detections = []
    for box in result.boxes:
        cid = int(box.cls[0].item())
        detections.append({
            "class_id": cid,
            "class_name": names.get(cid, f"Class {cid}"),
            "confidence": float(box.conf[0].item()),
            "bbox": [int(round(v)) for v in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return _draw(img, detections), detections


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Dùng: python -m src.predict <duong_dan_anh> [confidence]")
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    out, dets = predict_image(load_model(), sys.argv[1], conf)
    for d in dets:
        print(d)
    print(f"Tổng: {len(dets)} biển")
    out_path = Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_pred.jpg")
    cv2.imencode(".jpg", out)[1].tofile(str(out_path))
    print("Đã lưu:", out_path)
