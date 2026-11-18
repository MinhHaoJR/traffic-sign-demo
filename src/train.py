"""
[Hào] Huấn luyện YOLO11 cho các lớp biển báo trong configs/classes.json.

Chạy thử nhanh (3 epoch):   python -m src.train
Huấn luyện đầy đủ:          python -m src.train --epochs 30 --name run_30ep --save-model
Dùng GPU NVIDIA:            python -m src.train --device 0 ...
"""
import argparse
import csv
import shutil
import time
from datetime import datetime
from pathlib import Path

from src.paths import (
    DATA_EXAMPLE_YAML_PATH,
    DATA_YAML_PATH,
    EXPERIMENT_LOG_PATH,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    TRAINING_OUTPUT_DIR,
)

LOG_FIELDS = [
    "run_name", "date", "model", "imgsz", "epochs", "batch", "device", "seed",
    "train_time_min", "val_precision", "val_recall", "val_map50", "val_map50_95",
    "ultralytics", "torch", "weights",
]


def ensure_data_yaml() -> Path:
    """data.yaml chứa đường dẫn tuyệt đối từng máy nên không commit; thiếu thì sinh từ file mẫu."""
    if DATA_YAML_PATH.exists():
        return DATA_YAML_PATH
    if not (PROCESSED_DATA_DIR / "images" / "train").exists():
        raise FileNotFoundError(
            f"Chưa có dữ liệu tại {PROCESSED_DATA_DIR}. Nhờ Tâm gửi data/processed "
            "(hoặc chạy python -m src.prepare_data nếu đã có data/raw)."
        )
    text = DATA_EXAMPLE_YAML_PATH.read_text(encoding="utf-8")
    text = text.replace("path: data/processed", f"path: {PROCESSED_DATA_DIR.as_posix()}")
    DATA_YAML_PATH.write_text(text, encoding="utf-8")
    return DATA_YAML_PATH


def append_log(row: dict) -> None:
    new_file = not EXPERIMENT_LOG_PATH.exists()
    with open(EXPERIMENT_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Huấn luyện YOLO cho biển báo.")
    p.add_argument("--model", default="yolo11n.pt", help="Trọng số khởi đầu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu", help="cpu hoặc 0 (GPU NVIDIA)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", choices=["ram", "disk"], default=None,
                   help="Giữ ảnh đã giải mã để nạp nhanh hơn (cần --workers 0 trên máy ít RAM)")
    p.add_argument("--name", default="traffic_sign_yolo11")
    p.add_argument("--save-model", action="store_true",
                   help="Chép best.pt của lần chạy này vào models/best.pt")
    p.add_argument("--resume", action="store_true",
                   help="Chạy tiếp lần train --name bị ngắt từ last.pt (giữ nguyên tham số cũ)")
    a = p.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    data_yaml = ensure_data_yaml()
    device = int(a.device) if a.device.isdigit() else a.device

    start = time.time()
    if a.resume:
        # train_time_min trong nhật ký chỉ tính phần chạy tiếp
        results = YOLO(TRAINING_OUTPUT_DIR / a.name / "weights" / "last.pt").train(resume=True)
    else:
        results = YOLO(a.model).train(
            data=str(data_yaml),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            device=device,
            workers=a.workers,
            seed=a.seed,
            cache=a.cache or False,
            fliplr=0.0,  # không lật ngang: sẽ đảo nghĩa biển rẽ trái/rẽ phải
            flipud=0.0,
            project=str(TRAINING_OUTPUT_DIR),
            name=a.name,
            exist_ok=True,
        )
    minutes = (time.time() - start) / 60
    if a.resume:  # lấy cấu hình thật của lần chạy, không dùng giá trị mặc định của dòng lệnh
        import yaml

        cfg = yaml.safe_load((Path(results.save_dir) / "args.yaml").read_text(encoding="utf-8"))
        a.imgsz, a.epochs, a.batch, a.seed = (cfg[k] for k in ("imgsz", "epochs", "batch", "seed"))
        a.model = "chạy tiếp từ last.pt"  # args.yaml đã bị ghi đè bằng last.pt, không còn trọng số khởi đầu

    best = Path(results.save_dir) / "weights" / "best.pt"
    m = results.results_dict  # chỉ số trên tập VAL của epoch tốt nhất
    append_log({
        "run_name": a.name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": a.model,
        "imgsz": a.imgsz,
        "epochs": a.epochs,
        "batch": a.batch,
        "device": a.device,
        "seed": a.seed,
        "train_time_min": round(minutes, 1),
        "val_precision": round(m.get("metrics/precision(B)", float("nan")), 4),
        "val_recall": round(m.get("metrics/recall(B)", float("nan")), 4),
        "val_map50": round(m.get("metrics/mAP50(B)", float("nan")), 4),
        "val_map50_95": round(m.get("metrics/mAP50-95(B)", float("nan")), 4),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "weights": best.resolve().relative_to(ROOT_DIR).as_posix(),
    })
    print(f"\nĐã ghi nhật ký: {EXPERIMENT_LOG_PATH}\nTrọng số tốt nhất: {best}")

    if a.save_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, MODEL_PATH)
        print(f"Đã chép vào {MODEL_PATH}")


if __name__ == "__main__":
    main()
