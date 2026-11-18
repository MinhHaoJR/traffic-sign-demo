"""
Ảnh biển cận cảnh: cắt phóng to quanh biển có sẵn để model nhận được biển chụp gần, chiếm phần lớn khung
và có thể bị cắt ở mép (dữ liệu gốc gần như toàn biển nhỏ ở xa).

    python -m src.closeups build                 # chạy sau src.prepare_merged
    python -m src.closeups eval [--model models/best.pt]

build:
  * data/processed/images/train/cu_*.jpg: tối đa --train-cap ảnh mỗi lớp, chỉ cắt từ ảnh train.
  * data/processed/images/closeup_test/: tối đa --test-cap ảnh mỗi lớp, chỉ cắt từ ảnh test, để đo.
  Val và test gốc không đổi. Biển chỉ lộ một phần nhỏ trong khung cắt được tô xám để không thành
  biển chưa gán nhãn.
eval: mAP trên bộ test cận cảnh.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.paths import IMAGES_DIR, LABELS_DIR, MODEL_PATH, OUTPUT_DIR

MIN_SIDE = 32           # cạnh biển tối thiểu (px) để phóng to mà vẫn đủ chi tiết
RATIO = (0.3, 0.75)     # cạnh biển / cạnh ngắn của khung cắt
ASPECT = (1.0, 1.8)     # rộng / cao của khung cắt
MIN_VISIBLE = 0.6       # biển được chọn lộ ít nhất 60% trong khung
KEEP_VISIBLE = 0.5      # biển khác lộ >= 50% thì giữ nhãn, lộ ít hơn thì tô xám
EXTS = (".jpg", ".jpeg", ".png")


def _boxes(label: Path, w: int, h: int) -> list:
    out = []
    for line in label.read_text().splitlines():
        if line.strip():
            c, xc, yc, bw, bh = line.split()[:5]
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            out.append((int(c), xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return out


def _crop(img: np.ndarray, boxes: list, k: int, rng: random.Random):
    """Khung cắt quanh boxes[k]; trả (ảnh, dòng nhãn YOLO) hoặc None."""
    H, W = img.shape[:2]
    _, x1, y1, x2, y2 = boxes[k]
    bw, bh = x2 - x1, y2 - y1
    ch = max(bw, bh) / rng.uniform(*RATIO)
    cw = ch * rng.uniform(*ASPECT)
    cw, ch = min(cw, W), min(ch, H)
    if min(cw, ch) < 2 * MIN_SIDE:
        return None
    # vị trí khung: biển có thể lệch hẳn về một mép, miễn lộ ít nhất MIN_VISIBLE
    lo_x, hi_x = max(0, x1 + MIN_VISIBLE * bw - cw), min(W - cw, x2 - MIN_VISIBLE * bw)
    lo_y, hi_y = max(0, y1 + MIN_VISIBLE * bh - ch), min(H - ch, y2 - MIN_VISIBLE * bh)
    if lo_x > hi_x or lo_y > hi_y:
        return None
    cx, cy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    X1, Y1, X2, Y2 = int(cx), int(cy), int(cx + cw), int(cy + ch)
    out = img[Y1:Y2, X1:X2].copy()
    oh, ow = out.shape[:2]
    lines = []
    for c, a1, b1, a2, b2 in boxes:
        i1, j1, i2, j2 = max(a1, X1), max(b1, Y1), min(a2, X2), min(b2, Y2)
        if i2 <= i1 or j2 <= j1:
            continue
        vis = (i2 - i1) * (j2 - j1) / ((a2 - a1) * (b2 - b1))
        i1, j1, i2, j2 = i1 - X1, j1 - Y1, i2 - X1, j2 - Y1
        if vis >= KEEP_VISIBLE:
            lines.append(f"{c} {(i1 + i2) / 2 / ow:.6f} {(j1 + j2) / 2 / oh:.6f} {(i2 - i1) / ow:.6f} {(j2 - j1) / oh:.6f}")
        else:
            out[int(j1):int(np.ceil(j2)), int(i1):int(np.ceil(i2))] = 114
    return out, lines


def build_split(src: str, dst: str, prefix: str, cap: int, seed: int) -> dict:
    rng = random.Random(seed)
    img_dir, lbl_dir = IMAGES_DIR / src, LABELS_DIR / src
    out_img, out_lbl = IMAGES_DIR / dst, LABELS_DIR / dst
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for old in list(out_img.glob(f"{prefix}*")) + list(out_lbl.glob(f"{prefix}*")):
        old.unlink()  # chạy lại thì thay bộ cũ

    cands = defaultdict(list)  # lớp -> [(ảnh, chỉ số biển)]
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS and not p.stem.startswith("cu_")):
        w, h = Image.open(ip).size
        for k, (c, x1, y1, x2, y2) in enumerate(_boxes(lbl_dir / f"{ip.stem}.txt", w, h)):
            if max(x2 - x1, y2 - y1) >= MIN_SIDE:
                cands[c].append((ip, k))

    counts = {}
    for c in sorted(cands):
        pool = cands[c][:]
        rng.shuffle(pool)
        n = 0
        for ip, k in pool:
            if n == cap:
                break
            img = cv2.imdecode(np.fromfile(str(ip), np.uint8), cv2.IMREAD_COLOR)
            res = _crop(img, _boxes(lbl_dir / f"{ip.stem}.txt", img.shape[1], img.shape[0]), k, rng)
            if res is None or not res[1]:
                continue
            name = f"{prefix}{ip.stem}_{k}"
            cv2.imencode(".jpg", res[0], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(out_img / f"{name}.jpg"))
            (out_lbl / f"{name}.txt").write_text("\n".join(res[1]) + "\n")
            n += 1
        counts[c] = n
    return counts


def evaluate(model_path, imgsz=640, batch=16, device=None):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    yaml = OUTPUT_DIR / "closeup_eval.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    names = "".join(f"  {i}: \"{n}\"\n" for i, n in model.names.items())
    yaml.write_text(f"path: {IMAGES_DIR.parent.as_posix()}\ntrain: images/train\nval: images/closeup_test\n"
                    f"nc: {len(model.names)}\nnames:\n{names}", encoding="utf-8")
    m = model.val(data=str(yaml), split="val", imgsz=imgsz, batch=batch, device=device, workers=0,
                  plots=False, verbose=False)
    return m.box.mp, m.box.mr, m.box.map50, m.box.map


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Ảnh biển cận cảnh cho train và để đo.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train-cap", type=int, default=80, help="Số ảnh cận cảnh tối đa mỗi lớp cho train")
    b.add_argument("--test-cap", type=int, default=40, help="Số ảnh cận cảnh tối đa mỗi lớp để đo")
    b.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("eval")
    e.add_argument("--model", default=str(MODEL_PATH))
    e.add_argument("--device", default=None)
    a = p.parse_args()
    if a.cmd == "build":
        for src, dst, prefix, cap, seed in [("train", "train", "cu_", a.train_cap, a.seed),
                                            ("test", "closeup_test", "cut_", a.test_cap, a.seed + 1)]:
            counts = build_split(src, dst, prefix, cap, seed)
            print(f"{dst}: {sum(counts.values())} ảnh cận cảnh từ {src} | mỗi lớp: {counts}")
    else:
        P, R, m50, m = evaluate(a.model, device=a.device)
        print(f"Test cận cảnh ({a.model}): Precision {P:.4f} · Recall {R:.4f} · mAP@0.5 {m50:.4f} · mAP@0.5:0.95 {m:.4f}")


if __name__ == "__main__":
    main()
