"""
prepare_merged.py - Gộp 3 bộ dữ liệu biển báo Việt Nam theo configs/classes.json (28 lớp: biển cấm, tốc độ, nguy hiểm, hiệu lệnh, chỉ dẫn).

Nguồn (đặt trong data/, không đưa lên Git):
  ds1 data/raw/data/{images,labels}           vietnam_traffic_signs_dataset-main (16 lớp)
  ds2 data/raw_ds2/train_data/{images,labels}/{train,val,test}
                                              Vietnamese traffic signs detection and recognition (29 lớp)
  ds3 data/raw_ds3/archive/{images,labels}    Vietnam Traffic Signs (52 lớp)

Các bước: remap nhãn về các lớp dự án, bỏ ảnh có nhãn lẫn lộn, khử trùng MD5, giới hạn số đối tượng mỗi lớp,
gom khung hình liên tiếp thành nhóm và chia nhóm sang train/val/test (70/15/15) theo từng lớp.
Kết quả ghi vào data/processed, docs/data_summary.csv, docs/split_manifest.csv và configs/*.yaml,
cùng đường dẫn mà src.train, src.evaluate và src.check_and_visualize đang dùng.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.paths import CLASSES_PATH, CONFIGS_DIR, DATA_DIR, DOCS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.prepare_data import compute_file_hash, remap_classes, validate_labels

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CLASS_NAMES = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}

# ID nguồn -> ID dự án. ID 0-3 giữ nguyên như bản 4 lớp.
SOURCES = {
    "ds1": {
        "root": RAW_DATA_DIR / "data",
        "map": {3: 0, 2: 1, 11: 2, 12: 3, 0: 4, 1: 10, 4: 22, 6: 23, 8: 24, 9: 25, 14: 26, 15: 27},
        # 5/10 lẫn nhiều kiểu biển (có cả biển giống lớp 23-25), 13 là biển trẻ em kiểu châu Âu.
        "skip": {5, 10, 13},
    },
    "ds2": {
        "root": DATA_DIR / "raw_ds2" / "train_data",
        "map": {1: 0, 2: 1, 3: 2, 4: 3, 0: 4, 5: 5, 6: 6, 8: 9, 22: 12, 24: 17},
        # Nhãn gộp nhiều loại biển, có thể chứa lớp của dự án: 9/10 cấm ô tô/xe tải, 11 cấm khác,
        # 12 hiệu lệnh, 14 tốc độ, 18 giao nhau, 19 nguy hiểm, 20 người đi bộ, 23 cảnh báo khác,
        # 25 làn + tốc độ, 28 khác. Ảnh có các nhãn này bị bỏ để biển không thành nền chưa gán nhãn.
        "skip": {9, 10, 11, 12, 14, 18, 19, 20, 23, 25, 28},
    },
    "ds3": {
        "root": DATA_DIR / "raw_ds3" / "archive",
        "map": {46: 0, 10: 1, 7: 2, 32: 3, 2: 4, 17: 5, 49: 6, 18: 7, 23: 8, 29: 9,
                # ID 11-21: biển có >= 13 nhóm 25 khung riêng biệt trong ds3
                3: 11, 13: 12, 39: 13, 0: 14, 41: 15, 5: 16, 12: 17, 8: 18, 40: 19, 45: 20, 6: 21,
                20: 22, 27: 26},  # ID 22-27 chủ yếu từ ds1, ds3 góp thêm Cấm ô tô và Trẻ em
        "skip": set(),
    },
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect(source: str, cfg: dict) -> list[dict]:
    samples = []
    for lbl in sorted(cfg["root"].glob("labels/**/*.txt")):
        rel = lbl.relative_to(cfg["root"] / "labels").with_suffix("")
        img = next((p for ext in (".jpg", ".png", ".jpeg")
                    if (p := cfg["root"] / "images" / rel.with_suffix(ext)).exists()), None)
        if img is None:
            continue
        boxes = validate_labels([l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()])
        if any(b[0] in cfg["skip"] for b in boxes):
            continue
        boxes = remap_classes(boxes, cfg["map"])
        if not boxes:
            continue
        digits = re.findall(r"\d+", img.stem)
        # Ảnh đặt tên theo số khung hình: 25 khung liên tiếp là một nhóm để không rò rỉ giữa các tập.
        group = f"{source}_{rel.parent.as_posix()}_{int(digits[-1]) // 25}" if digits else f"{source}_{img.stem}"
        samples.append({"source": source, "img": img, "boxes": boxes, "group": group,
                        "out_name": f"{source}_{img.name}"})
    return samples


def cap_per_class(samples: list[dict], cap: int, seed: int) -> list[dict]:
    """Giữ ảnh nếu có ít nhất một lớp chưa đủ `cap` đối tượng; ds2/ds3 xét trước vì có lớp hiếm."""
    rng = random.Random(seed)
    order = sorted(samples, key=lambda s: (s["source"] == "ds1", rng.random()))
    counts, kept = Counter(), []
    for s in order:
        if any(counts[b[0]] < cap for b in s["boxes"]):
            kept.append(s)
            counts.update(b[0] for b in s["boxes"])
    return kept


def split_groups(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    """Chia theo nhóm; nhóm được xếp vào tập đang thiếu nhiều nhất cho lớp hiếm nhất của nhóm."""
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)
    total = Counter(b[0] for s in samples for b in s["boxes"])
    have = {sp: Counter() for sp in SPLIT_RATIOS}
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda g: min(total[b[0]] for s in groups[g] for b in s["boxes"]))
    splits = {sp: [] for sp in SPLIT_RATIOS}
    for g in keys:
        gcount = Counter(b[0] for s in groups[g] for b in s["boxes"])
        rare = min(gcount, key=lambda c: total[c])
        sp = max(SPLIT_RATIOS, key=lambda sp: SPLIT_RATIOS[sp] - have[sp][rare] / total[rare])
        splits[sp].extend(groups[g])
        have[sp].update(gcount)
    return splits


def export(splits: dict[str, list[dict]], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rows, counts = [], {sp: Counter() for sp in splits}
    for sp, items in splits.items():
        (out_dir / "images" / sp).mkdir(parents=True)
        (out_dir / "labels" / sp).mkdir(parents=True)
        for s in items:
            shutil.copy2(s["img"], out_dir / "images" / sp / s["out_name"])
            (out_dir / "labels" / sp / (Path(s["out_name"]).stem + ".txt")).write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in s["boxes"]),
                encoding="utf-8")
            counts[sp].update(b[0] for b in s["boxes"])
            rows.append({"source_subset": s["source"], "relative_path": f"images/{sp}/{s['out_name']}",
                         "file_hash": s["hash"], "group_id": s["group"], "target_split": sp,
                         "num_boxes": len(s["boxes"])})

    with open(DOCS_DIR / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(DOCS_DIR / "data_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class_id", "class_name", "num_objects"])
        for sp in splits:
            for cid, name in CLASS_NAMES.items():
                w.writerow([sp, cid, name, counts[sp][cid]])
        w.writerow([])
        w.writerow(["split", "unique_images", "total_objects"])
        for sp, items in splits.items():
            w.writerow([sp, len(items), sum(counts[sp].values())])
    print_table(splits, counts)


def print_table(splits, counts):
    print(f"{'Lớp':<30}" + "".join(f"{sp:>7}" for sp in splits) + f"{'tổng':>7}")
    for cid, name in CLASS_NAMES.items():
        vals = [counts[sp][cid] for sp in splits]
        print(f"{cid:>2} {name:<27}" + "".join(f"{v:>7}" for v in vals) + f"{sum(vals):>7}")
    print(f"{'Số ảnh':<30}" + "".join(f"{len(splits[sp]):>7}" for sp in splits))


def write_yaml(out_dir: Path) -> None:
    # json.dumps cho chuỗi trong ngoặc kép hợp lệ YAML (tên có dấu ":")
    names = "".join(f"  {cid}: {json.dumps(name, ensure_ascii=False)}\n" for cid, name in CLASS_NAMES.items())
    body = f"train: images/train\nval: images/val\ntest: images/test\n\nnc: {len(CLASS_NAMES)}\nnames:\n{names}"
    (CONFIGS_DIR / "data.example.yaml").write_text(
        f"# Cấu hình dữ liệu YOLO (mẫu đưa lên Git)\npath: data/processed\n{body}", encoding="utf-8")
    (CONFIGS_DIR / "data.yaml").write_text(
        f"# Sinh tự động trên máy hiện tại, không đưa lên Git\npath: {out_dir.resolve().as_posix()}\n{body}",
        encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Gộp 3 bộ dữ liệu theo configs/classes.json.")
    p.add_argument("--cap", type=int, default=700, help="Số đối tượng tối đa mỗi lớp (cắt bớt lớp nhiều ảnh của ds1)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    samples = []
    for name, cfg in SOURCES.items():
        if not cfg["root"].exists():
            sys.exit(f"[LỖI] Thiếu bộ dữ liệu {name}: {cfg['root']}")
        found = collect(name, cfg)
        print(f"{name}: {len(found)} ảnh có biển thuộc {len(CLASS_NAMES)} lớp")
        samples += found

    seen, unique = set(), []
    for s in samples:
        s["hash"] = compute_file_hash(s["img"])
        if s["hash"] not in seen:
            seen.add(s["hash"])
            unique.append(s)
    print(f"Sau khử trùng MD5: {len(unique)} ảnh (bỏ {len(samples) - len(unique)})")

    kept = cap_per_class(unique, a.cap, a.seed)
    print(f"Sau giới hạn {a.cap} đối tượng/lớp: {len(kept)} ảnh\n")
    splits = split_groups(kept, a.seed)
    export(splits, PROCESSED_DATA_DIR)
    write_yaml(PROCESSED_DATA_DIR)
    print(f"\nĐã xuất {PROCESSED_DATA_DIR}, docs/data_summary.csv, docs/split_manifest.csv, configs/data.yaml")


if __name__ == "__main__":
    main()
