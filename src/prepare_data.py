"""
prepare_data.py - Chuẩn bị và làm sạch dữ liệu nhận diện biển báo giao thông.

Thực hiện các nhiệm vụ của Tâm:
- T2: Lọc mẫu từ dữ liệu gốc vietnam_traffic_signs_dataset-main (khoảng 400-800 ảnh)
- T3: Lọc, kiểm tra nhãn YOLO, remap ID (3->0, 2->1, 11->2, 12->3), khử trùng lặp qua hash,
      nhóm theo cụm chuỗi video để chống rò rỉ (leakage), chia tập train/val/test (70/15/15),
      xuất dữ liệu vào data/processed và tạo docs/data_summary.csv, docs/split_manifest.csv.
"""

import os
import sys
import shutil
import hashlib
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hỗ trợ chạy trực tiếp hoặc thông qua module (python -m src.prepare_data)
try:
    from src.paths import (
        ROOT_DIR,
        CONFIGS_DIR,
        CLASSES_PATH,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        IMAGES_DIR,
        LABELS_DIR,
        DOCS_DIR,
        DATA_SUMMARY_PATH,
        SPLIT_MANIFEST_PATH,
        DATA_YAML_PATH,
        DATA_EXAMPLE_YAML_PATH,
    )
except ImportError:
    # Nếu chạy script trực tiếp từ thư mục src hoặc từ root mà không có PYTHONPATH
    current_file = Path(__file__).resolve()
    sys.path.insert(0, str(current_file.parent.parent))
    from src.paths import (
        ROOT_DIR,
        CONFIGS_DIR,
        CLASSES_PATH,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        IMAGES_DIR,
        LABELS_DIR,
        DOCS_DIR,
        DATA_SUMMARY_PATH,
        SPLIT_MANIFEST_PATH,
        DATA_YAML_PATH,
        DATA_EXAMPLE_YAML_PATH,
    )

# Ánh xạ từ ID gốc của bộ dữ liệu 16 lớp sang ID dự án
# 3: no parking -> 0: Cấm đỗ xe
# 2: no stop or parking -> 1: Cấm dừng và đỗ xe
# 11: no turn left -> 2: Cấm rẽ trái
# 12: no turn right -> 3: Cấm rẽ phải
TARGET_CLASS_MAP = {
    3: 0,
    2: 1,
    11: 2,
    12: 3,
}

CLASS_NAMES = {
    0: "Cấm đỗ xe",
    1: "Cấm dừng và đỗ xe",
    2: "Cấm rẽ trái",
    3: "Cấm rẽ phải",
}


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Tính mã băm MD5 của file ảnh để phát hiện trùng lặp."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_labels(lines: list[list[str]]) -> list[tuple[int, float, float, float, float]]:
    """
    Kiểm tra tính hợp lệ của nhãn YOLO:
    Format: class_id, x_center, y_center, width, height
    Tọa độ và kích thước chuẩn hóa trong khoảng [0, 1].
    """
    valid_boxes = []
    for parts in lines:
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        # Kiểm tra tọa độ trong khoảng [0, 1]
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue

        # Giới hạn bounding box trong khung ảnh [0, 1] (chống tràn do làm tròn)
        x1 = max(0.0, xc - w / 2)
        y1 = max(0.0, yc - h / 2)
        x2 = min(1.0, xc + w / 2)
        y2 = min(1.0, yc + h / 2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Tính lại center và w, h đã chuẩn hóa an toàn
        new_w = x2 - x1
        new_h = y2 - y1
        new_xc = x1 + new_w / 2
        new_yc = y1 + new_h / 2

        valid_boxes.append((cls_id, new_xc, new_yc, new_w, new_h))
    return valid_boxes


def remap_classes(
    boxes: list[tuple[int, float, float, float, float]],
    target_map: dict[int, int]
) -> list[tuple[int, float, float, float, float]]:
    """
    Đổi ID nhãn theo bảng ánh xạ và loại bỏ các nhãn ngoài phạm vi 4 lớp.
    3 -> 0, 2 -> 1, 11 -> 2, 12 -> 3
    """
    remapped = []
    for cls_id, xc, yc, w, h in boxes:
        if cls_id in target_map:
            remapped.append((target_map[cls_id], xc, yc, w, h))
    return remapped


def collect_samples(
    raw_data_dir: Path,
    target_map: dict[int, int],
    subsets: tuple[str, ...] = ("data",)
) -> list[dict]:
    """
    Quét danh sách ảnh và file nhãn từ thư mục raw.
    Lọc các ảnh chứa ít nhất một đối tượng thuộc 4 lớp mục tiêu.
    """
    samples = []
    for subset in subsets:
        subset_dir = raw_data_dir / subset
        lbl_dir = subset_dir / "labels"
        img_dir = subset_dir / "images"

        if not lbl_dir.exists() or not img_dir.exists():
            continue

        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            base_name = lbl_file.stem
            # Thử các phần mở rộng ảnh phổ biến
            img_file = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_dir / f"{base_name}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [line.strip().split() for line in f if line.strip()]

            boxes = validate_labels(lines)
            remapped_boxes = remap_classes(boxes, target_map)

            # Chỉ lấy các ảnh có ít nhất một nhãn hợp lệ thuộc 4 lớp
            if not remapped_boxes:
                continue

            # Xác định frame sequence index từ tên file số (nếu có)
            frame_idx = None
            if base_name.isdigit():
                frame_idx = int(base_name)

            samples.append({
                "source_subset": subset,
                "base_name": base_name,
                "img_path": img_file,
                "lbl_path": lbl_file,
                "frame_idx": frame_idx,
                "boxes": remapped_boxes,
            })
    return samples


def remove_duplicates(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Tính mã hash MD5 từng ảnh và loại bỏ các ảnh trùng lặp hoàn toàn.
    Trả về (unique_samples, duplicate_samples).
    """
    seen_hashes = {}
    unique_samples = []
    duplicates = []

    for s in samples:
        h = compute_file_hash(s["img_path"])
        s["file_hash"] = h
        if h in seen_hashes:
            duplicates.append(s)
        else:
            seen_hashes[h] = s
            unique_samples.append(s)

    return unique_samples, duplicates


def assign_group_ids(samples: list[dict], chunk_size: int = 25) -> None:
    """
    Gán group_id cho các ảnh. Các ảnh trong cùng một khối chuỗi liên tiếp (chunk_size frames)
    được coi là cùng một đoạn video (burst sequence) để đảm bảo không bị chia cắt sang cả train và test.
    """
    for s in samples:
        if s["frame_idx"] is not None:
            chunk_id = s["frame_idx"] // chunk_size
            s["group_id"] = f"{s['source_subset']}_chunk_{chunk_id}"
        else:
            s["group_id"] = f"{s['source_subset']}_single_{s['base_name']}"


def select_balanced_subset(
    samples: list[dict],
    target_count: int = 800,
    random_seed: int = 42
) -> list[dict]:
    """
    Chọn tập mẫu kích thước ~target_count (khoảng 400-800 ảnh theo phân công, mặc định 800)
    đảm bảo phân bố cân bằng giữa 4 lớp.
    """
    if len(samples) <= target_count:
        return samples

    rng = random.Random(random_seed)

    # Đếm số ảnh cho từng lớp
    class_groups = defaultdict(list)
    for s in samples:
        classes_in_sample = {b[0] for b in s["boxes"]}
        for cid in classes_in_sample:
            class_groups[cid].append(s)

    # Lấy đều từ các lớp
    selected_set = set()
    per_class_quota = target_count // len(TARGET_CLASS_MAP)

    for cid in sorted(class_groups.keys()):
        imgs = class_groups[cid]
        rng.shuffle(imgs)
        count = 0
        for s in imgs:
            if s["base_name"] not in selected_set:
                selected_set.add(s["base_name"])
                count += 1
                if count >= per_class_quota:
                    break

    # Nếu còn thiếu thì bù thêm từ các mẫu còn lại
    remaining = [s for s in samples if s["base_name"] not in selected_set]
    rng.shuffle(remaining)
    for s in remaining:
        if len(selected_set) >= target_count:
            break
        selected_set.add(s["base_name"])

    return [s for s in samples if s["base_name"] in selected_set]


def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> dict[str, list[dict]]:
    """
    Chia dữ liệu theo nhóm (group-based split) để đảm bảo các khung hình cùng chuỗi video
    không bị rò rỉ sang cả tập train và test.
    Tỷ lệ mục tiêu: 70% train, 15% val, 15% test.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s["group_id"]].append(s)

    group_keys = list(groups.keys())
    rng = random.Random(random_seed)
    rng.shuffle(group_keys)

    total_samples = len(samples)
    train_target = int(total_samples * train_ratio)
    val_target = int(total_samples * val_ratio)

    splits = {"train": [], "val": [], "test": []}
    for gid in group_keys:
        items = groups[gid]
        g_len = len(items)

        if len(splits["train"]) + g_len <= train_target:
            splits["train"].extend(items)
            for item in items:
                item["target_split"] = "train"
        elif len(splits["val"]) + g_len <= val_target:
            splits["val"].extend(items)
            for item in items:
                item["target_split"] = "val"
        else:
            splits["test"].extend(items)
            for item in items:
                item["target_split"] = "test"

    return splits


def export_dataset(
    splits: dict[str, list[dict]],
    processed_dir: Path,
    docs_dir: Path
) -> None:
    """
    Xuất ảnh và nhãn vào thư mục data/processed:
      - images/{train,val,test}
      - labels/{train,val,test}
    Xuất tài liệu:
      - docs/data_summary.csv
      - docs/split_manifest.csv
    """
    # Chuẩn bị thư mục processed (làm sạch trước khi xuất mới)
    for split_name in ["train", "val", "test"]:
        img_out = processed_dir / "images" / split_name
        lbl_out = processed_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for f in list(img_out.glob("*")):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        for f in list(lbl_out.glob("*")):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    manifest_rows = []
    summary_counts = {
        split: {cid: 0 for cid in CLASS_NAMES.keys()}
        for split in ["train", "val", "test"]
    }
    split_unique_images = {split: 0 for split in ["train", "val", "test"]}

    for split_name, items in splits.items():
        split_unique_images[split_name] = len(items)
        for s in items:
            out_img = processed_dir / "images" / split_name / s["img_path"].name
            out_lbl = processed_dir / "labels" / split_name / f"{s['base_name']}.txt"

            # Sao chép file ảnh
            shutil.copy2(s["img_path"], out_img)

            # Ghi file nhãn remap
            with open(out_lbl, "w", encoding="utf-8") as lf:
                for cid, xc, yc, w, h in s["boxes"]:
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    summary_counts[split_name][cid] += 1

            rel_img_path = f"images/{split_name}/{out_img.name}"
            manifest_rows.append({
                "source_subset": s["source_subset"],
                "relative_path": rel_img_path,
                "file_hash": s["file_hash"],
                "group_id": s["group_id"],
                "target_split": split_name,
                "num_boxes": len(s["boxes"]),
            })

    # Xuất docs/split_manifest.csv
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = docs_dir / "split_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_subset", "relative_path", "file_hash", "group_id", "target_split", "num_boxes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Xuất docs/data_summary.csv
    summary_csv = docs_dir / "data_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "class_name", "num_objects"])
        for split_name in ["train", "val", "test"]:
            for cid in sorted(CLASS_NAMES.keys()):
                writer.writerow([split_name, cid, CLASS_NAMES[cid], summary_counts[split_name][cid]])
        writer.writerow([])
        writer.writerow(["split", "unique_images", "total_objects"])
        for split_name in ["train", "val", "test"]:
            total_objs = sum(summary_counts[split_name].values())
            writer.writerow([split_name, split_unique_images[split_name], total_objs])


def generate_yaml_configs(
    configs_dir: Path,
    processed_dir: Path
) -> None:
    """
    Sinh configs/data.yaml (đường dẫn tuyệt đối cho máy hiện tại)
    và configs/data.example.yaml (mẫu đưa lên Git).
    """
    configs_dir.mkdir(parents=True, exist_ok=True)

    # data.example.yaml
    example_content = """# Cấu hình dữ liệu YOLO (mẫu đưa lên Git)
path: data/processed
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: Cấm đỗ xe
  1: Cấm dừng và đỗ xe
  2: Cấm rẽ trái
  3: Cấm rẽ phải
"""
    with open(configs_dir / "data.example.yaml", "w", encoding="utf-8") as f:
        f.write(example_content)

    # data.yaml
    processed_path_str = processed_dir.resolve().as_posix()
    local_content = f"""# Cấu hình dữ liệu YOLO (sinh tự động trên máy hiện tại, không đưa lên Git)
path: {processed_path_str}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: Cấm đỗ xe
  1: Cấm dừng và đỗ xe
  2: Cấm rẽ trái
  3: Cấm rẽ phải
"""
    with open(configs_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(local_content)


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị và làm sạch dữ liệu biển báo giao thông.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Thư mục chứa dữ liệu gốc (mặc định: data/raw)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Thư mục xuất dữ liệu đã xử lý (mặc định: data/processed)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=800,
        help="Số lượng ảnh mục tiêu cần chọn (mặc định: 800, trong khoảng 400-800)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để tái lập kết quả (mặc định: 42)"
    )
    args = parser.parse_args()

    print(f"=== Bắt đầu chuẩn bị dữ liệu (Tâm - T2 & T3) ===")
    print(f"Dữ liệu gốc: {args.raw_dir}")
    print(f"Thư mục xuất: {args.output_dir}")
    print(f"Số lượng ảnh mục tiêu: {args.max_samples} (Seed: {args.seed})")

    if not args.raw_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục raw: {args.raw_dir}")
        sys.exit(1)

    # 1. Thu thập và kiểm tra nhãn
    print("\n1. Đang quét và kiểm tra nhãn ảnh từ data/raw...")
    samples = collect_samples(args.raw_dir, TARGET_CLASS_MAP, subsets=("data",))
    print(f"   -> Tìm thấy {len(samples)} ảnh hợp lệ chứa ít nhất 1 trong 4 lớp.")

    if not samples:
        print("[LỖI] Không tìm thấy ảnh phù hợp trong thư mục raw.")
        sys.exit(1)

    # 2. Khử trùng lặp qua MD5 hash
    print("\n2. Đang kiểm tra khử trùng lặp qua MD5 hash...")
    unique_samples, duplicates = remove_duplicates(samples)
    print(f"   -> Số ảnh duy nhất: {len(unique_samples)}, số ảnh trùng lặp: {len(duplicates)}")

    # 3. Gán cụm chuỗi video
    print("\n3. Đang gom nhóm các chuỗi video để chống rò rỉ (leakage)...")
    assign_group_ids(unique_samples, chunk_size=25)

    # 4. Lọc chọn tập cân bằng khoảng 400-800 ảnh
    print(f"\n4. Đang chọn tập mẫu cân bằng 4 lớp (~{args.max_samples} ảnh)...")
    selected_samples = select_balanced_subset(unique_samples, target_count=args.max_samples, random_seed=args.seed)
    print(f"   -> Đã chọn {len(selected_samples)} ảnh.")

    # 5. Chia tập train / val / test theo group
    print("\n5. Đang chia tập theo nhóm (Train 70%, Val 15%, Test 15%)...")
    splits = split_dataset(selected_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=args.seed)
    for sp_name, sp_list in splits.items():
        print(f"   - Tập {sp_name.upper():<5}: {len(sp_list)} ảnh")

    # 6. Xuất dữ liệu ra data/processed và tài liệu CSV
    print("\n6. Đang sao chép ảnh, xuất nhãn remap và tài liệu báo cáo...")
    export_dataset(splits, args.output_dir, DOCS_DIR)
    print(f"   -> Dữ liệu đã xuất vào: {args.output_dir}")
    print(f"   -> Bảng thống kê: {DATA_SUMMARY_PATH}")
    print(f"   -> Danh sách chia tập: {SPLIT_MANIFEST_PATH}")

    # 7. Sinh file cấu hình YAML (T4 liên kết)
    print("\n7. Đang sinh cấu hình configs/data.yaml và configs/data.example.yaml...")
    generate_yaml_configs(CONFIGS_DIR, args.output_dir)
    print(f"   -> Đã tạo {DATA_YAML_PATH} và {DATA_EXAMPLE_YAML_PATH}")

    print("\n=== Hoàn thành xuất sắc T2 & T3 ===")


if __name__ == "__main__":
    main()
