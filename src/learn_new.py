"""
Train lại YOLO với các biển người dùng đã dạy trong app (data/feedback) cùng bộ dữ liệu gộp.

    python -m src.learn_new [--epochs 20] [--min-per-class 20] [--max-drop 0.02]

1. Kiểm tra mỗi lớp mới có đủ ảnh trong kho phản hồi.
2. Ghi configs/data_learn.yaml: train/val = data/processed + data/feedback, test = data/processed/test.
3. Fine-tune từ models/best.pt (lớp cũ giữ ID, lớp mới nối tiếp).
4. Đánh giá model cũ và mới trên cùng tập test. Chỉ thay models/best.pt khi mAP@0.5 trung bình các lớp
   cũ không giảm quá --max-drop; model cũ được giữ ở models/best_prev.pt.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from src.few_shot import feedback_counts, load_classes
from src.paths import CONFIGS_DIR, FEEDBACK_DIR, MODEL_PATH, MODELS_DIR, PROCESSED_DATA_DIR, ROOT_DIR, TRAINING_OUTPUT_DIR

DATA_LEARN_PATH = CONFIGS_DIR / "data_learn.yaml"


def write_yaml(names: dict) -> Path:
    def dirs(split):
        out = [(PROCESSED_DATA_DIR / "images" / split).as_posix()]
        if (FEEDBACK_DIR / "images" / split).is_dir():
            out.append((FEEDBACK_DIR / "images" / split).as_posix())
        return json.dumps(out)

    nc = max(names) + 1
    body = "".join(f"  {i}: {json.dumps(names.get(i, f'class_{i}'), ensure_ascii=False)}\n" for i in range(nc))
    DATA_LEARN_PATH.write_text(
        "# Sinh bởi src.learn_new, không đưa lên Git\n"
        f"train: {dirs('train')}\nval: {dirs('val')}\ntest: {json.dumps([(PROCESSED_DATA_DIR / 'images' / 'test').as_posix()])}\n"
        f"nc: {nc}\nnames:\n{body}", encoding="utf-8")
    return DATA_LEARN_PATH


def old_class_map50(weights, data_yaml, old_ids: set, device) -> tuple[float, dict]:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=640, batch=16, device=device,
                               workers=0, verbose=False, plots=False)
    per = {int(c): float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50) if int(c) in old_ids}
    return (sum(per.values()) / len(per) if per else 0.0), per


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Train lại YOLO với biển đã dạy trong app.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-per-class", type=int, default=20, help="Số ảnh tối thiểu của mỗi lớp mới")
    p.add_argument("--max-drop", type=float, default=0.02, help="mAP@0.5 lớp cũ được phép giảm tối đa")
    p.add_argument("--force", action="store_true", help="Train dù lớp mới chưa đủ ảnh")
    a = p.parse_args()

    import torch
    from ultralytics import YOLO

    names = load_classes()
    counts = feedback_counts()
    if not counts:
        sys.exit("Kho phản hồi trống. Dạy biển trong app trước (nút Đặt tên biển).")
    old_names = YOLO(str(MODEL_PATH)).names
    new_ids = sorted(c for c in counts if c not in old_names)
    print("Kho phản hồi:", {names.get(c, c): n for c, n in counts.items()})
    short = {names[c]: counts[c] for c in new_ids if counts[c] < a.min_per_class}
    if short and not a.force:
        sys.exit(f"Lớp mới chưa đủ {a.min_per_class} ảnh: {short}. Dạy thêm hoặc dùng --force.")
    data_yaml = write_yaml(names)
    device = 0 if torch.cuda.is_available() else "cpu"

    name = f"learn_{datetime.now():%Y%m%d_%H%M}"
    start = time.time()
    YOLO(str(MODEL_PATH)).train(
        data=str(data_yaml), epochs=a.epochs, imgsz=640, batch=16, device=device, workers=0, cache="disk",
        fliplr=0.0, flipud=0.0, project=str(TRAINING_OUTPUT_DIR), name=name, exist_ok=True,
    )
    best = TRAINING_OUTPUT_DIR / name / "weights" / "best.pt"
    old_ids = set(old_names)
    old_score, _ = old_class_map50(MODEL_PATH, data_yaml, old_ids, device)
    new_score, _ = old_class_map50(best, data_yaml, old_ids, device)
    print(f"mAP@0.5 lớp cũ trên test: model hiện tại {old_score:.4f}, model mới {new_score:.4f} "
          f"({(time.time() - start) / 60:.1f} phút)")

    from src.train import append_log
    import ultralytics

    append_log({"run_name": name, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": MODEL_PATH.name,
                "imgsz": 640, "epochs": a.epochs, "batch": 16, "device": str(device), "seed": 0,
                "train_time_min": round((time.time() - start) / 60, 1), "val_precision": "", "val_recall": "",
                "val_map50": "", "val_map50_95": "",  # điểm so sánh là trên test, in ra màn hình ở trên "ultralytics": ultralytics.__version__,
                "torch": torch.__version__, "weights": best.resolve().relative_to(ROOT_DIR).as_posix()})
    if new_score < old_score - a.max_drop:
        print(f"Model mới làm lớp cũ giảm quá {a.max_drop}; giữ nguyên {MODEL_PATH}. Trọng số mới ở {best}")
        return
    prev = MODELS_DIR / "best_prev.pt"
    shutil.copy2(MODEL_PATH, prev)
    shutil.copy2(best, MODEL_PATH)
    print(f"Đã thay {MODEL_PATH} (model cũ lưu ở {prev}). Mở lại app để dùng model mới.")


if __name__ == "__main__":
    main()
