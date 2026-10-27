"""
[Hào] Đánh giá model cuối cùng trên tập TEST (chỉ chạy sau khi đã chốt model bằng val).

    python -m src.evaluate                  # ghi docs/test_metrics.csv
    python -m src.evaluate --examples       # thêm ảnh ví dụ đúng / sai / bỏ sót vào outputs/predictions/examples
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.paths import (
    CLASSES_PATH,
    DATA_YAML_PATH,
    MODEL_PATH,
    PREDICTIONS_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    TEST_METRICS_PATH,
    TRAINING_OUTPUT_DIR,
)
from src.predict import load_model, predict_image
from src.train import ensure_data_yaml

IOU_MATCH = 0.5
EXAMPLES_PER_KIND = 3


def read_gt(label_path: Path, w: int, h: int) -> list:
    """Nhãn YOLO -> [(class_id, [x1, y1, x2, y2] pixel)]."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = line.split()[:5]
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        boxes.append((int(c), [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]))
    return boxes


def iou(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify_image(gt: list, dets: list) -> str:
    """correct: khớp hết; missed: bỏ sót biển thật; wrong: sai lớp hoặc dự đoán thừa."""
    unmatched_gt = list(gt)
    wrong = False
    for d in dets:
        hit = next((g for g in unmatched_gt if g[0] == d["class_id"] and iou(g[1], d["bbox"]) >= IOU_MATCH), None)
        if hit:
            unmatched_gt.remove(hit)
        else:
            wrong = True
    if wrong:
        return "wrong"
    return "missed" if unmatched_gt else "correct"


def save_examples(model, conf: float) -> dict:
    img_dir = PROCESSED_DATA_DIR / "images" / "test"
    lbl_dir = PROCESSED_DATA_DIR / "labels" / "test"
    out_root = PREDICTIONS_OUTPUT_DIR / "examples"
    counts = {"correct": 0, "wrong": 0, "missed": 0}
    for img_path in sorted(img_dir.glob("*.jpg")):
        h, w = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR).shape[:2]
        annotated, dets = predict_image(model, img_path, conf)
        kind = classify_image(read_gt(lbl_dir / f"{img_path.stem}.txt", w, h), dets)
        counts[kind] += 1
        if counts[kind] <= EXAMPLES_PER_KIND:
            out = out_root / kind / img_path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", annotated)[1].tofile(str(out))
    return counts


def count_test_objects() -> tuple:
    per_class, n_images = {}, 0
    for f in (PROCESSED_DATA_DIR / "labels" / "test").glob("*.txt"):
        n_images += 1
        for line in f.read_text().splitlines():
            if line.strip():
                cid = int(line.split()[0])
                per_class[cid] = per_class.get(cid, 0) + 1
    return n_images, per_class


def main():
    p = argparse.ArgumentParser(description="Đánh giá model trên tập test.")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--imgsz", type=int, default=416, help="Phải trùng imgsz lúc huấn luyện")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25, help="Ngưỡng cho ảnh ví dụ (không ảnh hưởng mAP)")
    p.add_argument("--examples", action="store_true")
    a = p.parse_args()

    model = load_model(a.model)
    data_yaml = ensure_data_yaml()
    device = int(a.device) if a.device.isdigit() else a.device

    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=a.imgsz, batch=a.batch, device=device,
        project=str(TRAINING_OUTPUT_DIR), name="test_eval", exist_ok=True,
    )
    box = metrics.box
    names = {int(k): v for k, v in json.loads(CLASSES_PATH.read_text(encoding="utf-8")).items()}
    n_images, objects = count_test_objects()
    meta = {
        "model": Path(a.model).name, "imgsz": a.imgsz, "split": "test", "n_images": n_images,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    rows = [{**meta, "scope": "all", "class_id": "", "class_name": "Tất cả",
             "num_objects": sum(objects.values()), "precision": box.mp, "recall": box.mr,
             "map50": box.map50, "map50_95": box.map}]
    for i, cid in enumerate(box.ap_class_index):
        p_, r_, ap50, ap = box.class_result(i)
        rows.append({**meta, "scope": "class", "class_id": int(cid), "class_name": names[int(cid)],
                     "num_objects": objects.get(int(cid), 0), "precision": p_, "recall": r_,
                     "map50": ap50, "map50_95": ap})

    with open(TEST_METRICS_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in r.items()})
    print(f"Đã ghi {TEST_METRICS_PATH}")

    if a.examples:
        print("Ví dụ (số ảnh theo loại):", save_examples(model, a.conf), "->", PREDICTIONS_OUTPUT_DIR / "examples")


if __name__ == "__main__":
    main()
