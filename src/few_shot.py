"""
Học nhanh biển mới từ vài ảnh mẫu, và kho phản hồi để train lại YOLO.

Luồng nhận diện: YOLO khoanh vùng biển (biển lạ rơi vào lớp "Biển báo khác") -> bộ mã hóa ảnh cắt
so với ảnh mẫu trong models/few_shot.npz -> đủ giống thì đặt tên, không thì báo "Biển chưa biết".
Khi người dùng đặt tên một biển trong app (teach), ảnh cắt được thêm vào kho mẫu nên nhận ra ngay,
còn ảnh + nhãn được lưu vào data/feedback/ để `python -m src.learn_new` train lại YOLO.

    python -m src.few_shot train-encoder   # train bộ mã hóa (~1 phút GPU, cần 3 dataset gốc trong data/)
    python -m src.few_shot build           # ảnh mẫu cho các lớp đã học, lấy từ data/processed/train
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.paths import CLASSES_PATH, ENCODER_PATH, FEEDBACK_DIR, FEW_SHOT_PATH, PROCESSED_DATA_DIR

CROP = 96
MATCH_SIM = 0.88  # cosine tối thiểu để gán tên cho biển "khác"; đo bằng biển giữ riêng (xem README mục 7)
OTHER_NAME = "Biển báo khác"
UNKNOWN_NAME = "Biển chưa biết"
SHOTS_PER_CLASS = 30
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def read_bgr(path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def crop_box(img: np.ndarray, x1, y1, x2, y2, pad: float = 0.15):
    """Cắt vùng biển (nới 15%) và đưa về CROP×CROP; None nếu biển quá nhỏ."""
    h, w = img.shape[:2]
    dx, dy = (x2 - x1) * pad / 2, (y2 - y1) * pad / 2
    c = img[max(0, int(y1 - dy)):int(y2 + dy), max(0, int(x1 - dx)):int(x2 + dx)]
    if c.size == 0 or min(c.shape[:2]) < 8:
        return None
    return cv2.resize(c, (CROP, CROP))


def load_classes() -> dict:
    return {int(k): v for k, v in json.loads(CLASSES_PATH.read_text(encoding="utf-8")).items()}


def other_id(names: dict):
    return next((i for i, n in names.items() if n == OTHER_NAME), None)


def add_class(name: str) -> int:
    """Trả về ID của tên lớp; tên mới được thêm vào cuối configs/classes.json."""
    names = load_classes()
    name = name.strip()
    for i, n in names.items():
        if n.casefold() == name.casefold():
            return i
    new_id = max(names) + 1
    names[new_id] = name
    CLASSES_PATH.write_text(json.dumps({str(k): v for k, v in names.items()}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    from src.predict import _class_names
    _class_names.cache_clear()
    return new_id


def _to_tensor(batch_bgr: np.ndarray, device, aug: bool = False):
    """uint8 BGR (N,H,W,3) -> tensor chuẩn hóa ImageNet; aug: đổi sáng/màu, dịch, phóng nhẹ (không lật)."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(batch_bgr[..., ::-1].copy()).to(device).permute(0, 3, 1, 2).float() / 255
    if aug:
        b = x.shape[0]
        x = x * (0.6 + 0.8 * torch.rand(b, 1, 1, 1, device=device)) + 0.1 * (torch.rand(b, 3, 1, 1, device=device) - 0.5)
        theta = torch.zeros(b, 2, 3, device=device)
        s = 0.85 + 0.3 * torch.rand(b, device=device)
        theta[:, 0, 0], theta[:, 1, 1] = s, s
        theta[:, :, 2] = 0.15 * (torch.rand(b, 2, device=device) - 0.5)
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False,
                          padding_mode="border").clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


class Encoder:
    """ResNet18 đã train trên ảnh cắt biển báo; trả vector 512 chiều đã chuẩn hóa."""

    def __init__(self, path=ENCODER_PATH):
        import torch
        import torchvision

        if not Path(path).is_file():
            raise FileNotFoundError(f"Chưa có bộ mã hóa {path}. Chạy: python -m src.few_shot train-encoder")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = torchvision.models.resnet18()
        net.fc = torch.nn.Identity()
        net.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
        self.net = net.eval().to(self.device)

    def __call__(self, crops: list) -> np.ndarray:
        import torch

        if not crops:
            return np.zeros((0, 512), np.float32)
        x = np.stack(crops)
        with torch.no_grad():
            e = np.concatenate([self.net(_to_tensor(x[i:i + 256], self.device)).cpu().numpy()
                                for i in range(0, len(x), 256)])
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class FewShot:
    """Kho ảnh mẫu đã mã hóa + so khớp láng giềng gần nhất. An toàn khi app dạy trong lúc đang nhận diện."""

    def __init__(self, encoder: Encoder | None = None, path=FEW_SHOT_PATH):
        self.encoder = encoder or Encoder()
        self.path = Path(path)
        self.lock = threading.Lock()
        if self.path.is_file():
            d = np.load(self.path)
            self.emb, self.cls, self.taught = d["emb"].astype(np.float32), d["cls"], d["taught"]
        else:
            self.emb, self.cls, self.taught = np.zeros((0, 512), np.float32), np.zeros(0, int), np.zeros(0, bool)

    def save(self):
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, emb=self.emb.astype(np.float16), cls=self.cls, taught=self.taught)
        tmp.replace(self.path)

    def match(self, emb: np.ndarray):
        """Trả về [(class_id, cosine)] của mẫu gần nhất cho mỗi vector."""
        with self.lock:
            if len(self.emb) == 0:
                return [(None, 0.0)] * len(emb)
            sim = emb @ self.emb.T
            j = sim.argmax(1)
            return [(int(self.cls[k]), float(sim[i, k])) for i, k in enumerate(j)]

    def refine(self, img: np.ndarray, detections: list, names: dict) -> list:
        """Khung "Biển báo khác": đặt tên theo mẫu gần nhất, không đủ giống thì là "Biển chưa biết".
        Khung YOLO đã gán lớp: chỉ đổi khi giống một mẫu người dùng đã dạy hơn mọi mẫu của lớp YOLO chọn
        (ví dụ biển ngoặt phải mới dạy mà YOLO nhầm thành ngoặt trái)."""
        oid = other_id(names)
        for d in detections:
            if d["class_id"] == oid:
                d["class_name"] = UNKNOWN_NAME
        crops = [crop_box(img, *d["bbox"]) for d in detections]
        idx = [i for i, c in enumerate(crops) if c is not None]
        if not idx:
            return detections
        emb = self.encoder([crops[i] for i in idx])
        with self.lock:
            if len(self.emb) == 0:
                return detections
            sim, cls, taught = emb @ self.emb.T, self.cls, self.taught
        for row, i in zip(sim, idx):
            d, j = detections[i], int(row.argmax())
            cid, s = int(cls[j]), float(row[j])
            if s < MATCH_SIM or cid == d["class_id"] or cid == oid:
                continue
            if d["class_id"] == oid or (taught[j] and s > row[cls == d["class_id"]].max(initial=0.0)):
                d.update(class_id=cid, class_name=names.get(cid, f"Class {cid}"), confidence=s, few_shot=True)
        return detections

    def teach(self, img: np.ndarray, bbox, class_id: int, detections: list) -> Path:
        """Thêm biển trong `bbox` làm mẫu của `class_id` và lưu ảnh + nhãn vào kho phản hồi."""
        crop = crop_box(img, *bbox)
        if crop is None:
            raise ValueError("Biển quá nhỏ để làm mẫu (cần ít nhất 8×8 pixel).")
        e = self.encoder([crop])
        with self.lock:
            self.emb = np.vstack([self.emb, e])
            self.cls = np.append(self.cls, class_id)
            self.taught = np.append(self.taught, True)
            self.save()
        return save_feedback(img, bbox, class_id, detections)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / union if union > 0 else 0.0


def save_feedback(img: np.ndarray, bbox, class_id: int, detections: list) -> Path:
    """Ảnh lưu theo hash nên dạy nhiều biển trên cùng ảnh sẽ cập nhật chung một file nhãn.
    Các biển khác trên ảnh lấy nhãn từ kết quả hiện tại để không thành nền chưa gán nhãn."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Không mã hóa được ảnh để lưu phản hồi.")
    digest = hashlib.md5(buf.tobytes()).hexdigest()[:12]
    split = "val" if int(digest, 16) % 5 == 0 else "train"
    img_path = FEEDBACK_DIR / "images" / split / f"fb_{digest}.jpg"
    lbl_path = FEEDBACK_DIR / "labels" / split / f"fb_{digest}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(img_path))
    h, w = img.shape[:2]
    boxes = [(class_id, list(bbox))] + [(d["class_id"], d["bbox"]) for d in detections if _iou(d["bbox"], bbox) < 0.5]
    lbl_path.write_text("".join(
        f"{c} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n"
        for c, b in boxes), encoding="utf-8")
    return img_path


def feedback_counts() -> dict:
    """Số biển đã dạy theo lớp (đếm trong kho phản hồi)."""
    counts = defaultdict(int)
    for f in FEEDBACK_DIR.glob("labels/*/*.txt"):
        # shortcut: chỉ đếm biển dạy gần nhất của mỗi ảnh (dòng đầu), đủ để biết lớp nào còn thiếu mẫu
        counts[int(f.read_text().split()[0])] += 1
    return dict(counts)


# ---------------------------------------------------------------- train-encoder / build

def _encoder_samples(seed: int, cap: int):
    """Ảnh cắt từ cả 3 dataset, nhãn hợp nhất: lớp dự án -> P<id>, biển ngoài dự án giữ nhãn gốc của dataset."""
    from src.prepare_merged import SOURCES, OTHER_ID

    items = defaultdict(list)
    for src, cfg in SOURCES.items():
        root = cfg["root"]
        for lf in root.glob("labels/**/*.txt"):
            rel = lf.relative_to(root / "labels").with_suffix("")
            img = next((p for e in IMAGE_EXTS if (p := root / "images" / rel.with_suffix(e)).exists()), None)
            if img is None:
                continue
            d = re.findall(r"\d+", img.stem)
            group = f"{src}_{rel.parent}_{int(d[-1]) // 25 if d else img.stem}"
            for line in lf.read_text().splitlines():
                p = line.split()
                if not p or int(p[0]) in cfg["skip"]:
                    continue
                c = int(p[0])
                mapped = cfg["map"].get(c)
                if mapped is not None and mapped != OTHER_ID:
                    label = f"P{mapped}"
                elif src == "ds2" and mapped is None:
                    continue  # ds2 chưa map phần lớn là nhãn gộp nhiều loại biển
                else:
                    label = f"{src}_{c}"
                items[label].append((group, img, tuple(map(float, p[1:5]))))
    rng = random.Random(seed)
    labels = sorted(k for k, v in items.items() if len(v) >= 20)
    picked = {k: rng.sample(items[k], min(cap, len(items[k]))) for k in labels}
    return labels, picked


def _load_crops(sel):
    by_img = defaultdict(list)
    for i, (_, img, b) in enumerate(sel):
        by_img[img].append((i, b))
    out = [None] * len(sel)
    for img, lst in by_img.items():
        im = read_bgr(img)
        h, w = im.shape[:2]
        for i, (xc, yc, bw, bh) in lst:
            out[i] = crop_box(im, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
    return out


def train_encoder(epochs: int = 8, seed: int = 42, cap: int = 500):
    import torch
    import torch.nn as nn
    import torchvision

    torch.manual_seed(seed)
    np.random.seed(seed)
    labels, picked = _encoder_samples(seed, cap)
    X, Y, Xv, Yv = [], [], [], []
    rng = random.Random(seed)
    for li, k in enumerate(labels):
        groups = sorted({g for g, _, _ in picked[k]})
        rng.shuffle(groups)
        val_groups = set(groups[: max(1, len(groups) // 6)])
        for (g, _, _), c in zip(picked[k], _load_crops(picked[k])):
            if c is not None:
                (Xv if g in val_groups else X).append(c)
                (Yv if g in val_groups else Y).append(li)
    X, Y, Xv, Yv = np.stack(X), np.array(Y), np.stack(Xv), np.array(Yv)
    print(f"{len(labels)} nhãn, {len(X)} ảnh cắt train, {len(Xv)} val")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(512, len(labels))
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (len(X) // 128 + 1))
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(_to_tensor(X[idx], dev, aug=True)),
                                               torch.from_numpy(Y[idx]).to(dev), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            correct = sum((net(_to_tensor(Xv[i:i + 256], dev)).argmax(1).cpu().numpy() == Yv[i:i + 256]).sum()
                          for i in range(0, len(Xv), 256))
        print(f"epoch {ep + 1}/{epochs}  loss {loss.item():.3f}  val acc {correct / len(Yv):.3f}", flush=True)
    net.fc = nn.Identity()
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() for k, v in net.state_dict().items()}, ENCODER_PATH)
    print("Đã lưu", ENCODER_PATH)


def _split_crops(split: str, per_class: int, seed: int, skip: set):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    files = sorted((PROCESSED_DATA_DIR / "labels" / split).glob("*.txt"))
    rng.shuffle(files)
    for lf in files:
        for line in lf.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) not in skip and len(by_cls[int(p[0])]) < per_class:
                by_cls[int(p[0])].append((None, next(f for e in IMAGE_EXTS
                                                     if (f := PROCESSED_DATA_DIR / "images" / split / (lf.stem + e)).exists()),
                                          tuple(map(float, p[1:5]))))
    sel = [(c, it) for c, lst in by_cls.items() for it in lst]
    crops = _load_crops([it for _, it in sel])
    keep = [(c, x) for (c, _), x in zip(sel, crops) if x is not None]
    return [x for _, x in keep], np.array([c for c, _ in keep])


def build(per_class: int = SHOTS_PER_CLASS, seed: int = 42):
    """Tạo lại mẫu của các lớp đã học từ data/processed/train; giữ nguyên các mẫu người dùng đã dạy."""
    names = load_classes()
    oid = other_id(names)
    fs = FewShot()
    crops, cls = _split_crops("train", per_class, seed, {oid})
    emb = fs.encoder(crops)
    keep = fs.taught.astype(bool)
    fs.emb = np.vstack([emb, fs.emb[keep]])
    fs.cls = np.concatenate([cls, fs.cls[keep]])
    fs.taught = np.concatenate([np.zeros(len(cls), bool), fs.taught[keep]])
    fs.save()
    print(f"Đã lưu {FEW_SHOT_PATH}: {len(cls)} mẫu cho {len(set(cls))} lớp + {keep.sum()} mẫu đã dạy")
    # tự kiểm tra: ảnh cắt tập val phải khớp đúng lớp với mẫu gần nhất
    vcrops, vcls = _split_crops("val", 20, seed, {oid})
    pred = np.array([c for c, _ in fs.match(fs.encoder(vcrops))])
    acc = (pred == vcls).mean()
    print(f"Kiểm tra trên {len(vcls)} ảnh cắt tập val: so khớp đúng {acc:.3f}")
    assert acc > 0.8, "Bộ mã hóa so khớp kém bất thường; kiểm tra lại models/sign_encoder.pt"


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Học nhanh biển mới từ vài ảnh mẫu.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-encoder", help="Train bộ mã hóa ảnh cắt biển báo")
    t.add_argument("--epochs", type=int, default=8)
    b = sub.add_parser("build", help="Tạo ảnh mẫu cho các lớp đã học")
    b.add_argument("--per-class", type=int, default=SHOTS_PER_CLASS)
    sub.add_parser("status", help="Số biển đã dạy trong kho phản hồi")
    a = p.parse_args()
    if a.cmd == "train-encoder":
        train_encoder(a.epochs)
    elif a.cmd == "build":
        build(a.per_class)
    else:
        names, counts = load_classes(), feedback_counts()
        print(f"Kho phản hồi {FEEDBACK_DIR}: {'trống' if not counts else f'{sum(counts.values())} biển'}")
        for cid, n in sorted(counts.items()):
            print(f"{cid:>3} {names.get(cid, '?')}: {n} biển")


if __name__ == "__main__":
    main()
