"""
check_and_visualize.py - Kiểm tra toàn diện và vẽ trực quan hóa dữ liệu nhãn (Tâm - T4).

Thực hiện:
1. Kiểm tra tính hợp lệ của file cấu hình configs/data.yaml và configs/data.example.yaml.
2. Kiểm tra toàn vẹn định lượng 100% tập dữ liệu (800 ảnh và 800 nhãn):
   - Đọc kích thước ảnh thực tế (W, H).
   - Xác thực bounding box YOLO (class_id, x_center, y_center, width, height).
   - Thống kê kích thước bounding box và phân bố theo lớp.
3. Vẽ trực quan hóa bounding box kèm tên tiếng Việt cho các ảnh đại diện từ cả 3 tập (Train, Val, Test):
   - Mỗi lớp một màu sắc nhận diện riêng biệt.
   - Nhãn chữ tiếng Việt sắc nét bằng Pillow và font Unicode Windows.
   - Xuất từng ảnh mẫu vào outputs/label_verification/{train,val,test}/.
4. Tạo ảnh ghép dạng lưới (Collage Grid) tổng quan 4 lớp x 3 tập để đối chiếu nhanh.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import yaml
import json
import random
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module từ thư mục src
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.paths import (
    CLASSES_PATH,
    DATA_YAML_PATH,
    DATA_EXAMPLE_YAML_PATH,
    PROCESSED_DATA_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    OUTPUT_DIR,
    LABEL_VERIFICATION_DIR,
)

# Bảng màu đại diện cho 4 lớp biển báo (RGB)
CLASS_COLORS = {
    0: (30, 144, 255),    # Cấm đỗ xe: Xanh dương Dodger Blue
    1: (220, 20, 60),     # Cấm dừng và đỗ xe: Đỏ Crimson
    2: (255, 140, 0),     # Cấm rẽ trái: Cam Dark Orange
    3: (46, 139, 87),     # Cấm rẽ phải: Xanh lá Sea Green
}


def load_font(size: int = 18) -> ImageFont.ImageFont:
    """Tải font Unicode hỗ trợ tiếng Việt trên Windows."""
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\times.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def verify_yaml_configs() -> bool:
    """Kiểm tra cú pháp và nội dung 2 file cấu hình YAML."""
    print("\n--- 1. Kiểm tra cấu hình YAML (T4) ---")
    all_ok = True

    # 1. Kiểm tra data.example.yaml
    if not DATA_EXAMPLE_YAML_PATH.exists():
        print(f"  [X] Không tìm thấy file: {DATA_EXAMPLE_YAML_PATH}")
        all_ok = False
    else:
        with open(DATA_EXAMPLE_YAML_PATH, "r", encoding="utf-8") as f:
            ex_data = yaml.safe_load(f)
        if ex_data.get("nc") == 4 and "names" in ex_data:
            print(f"  [OK] data.example.yaml hợp lệ (nc={ex_data.get('nc')}, names={len(ex_data.get('names', {}))} lớp)")
        else:
            print("  [!] data.example.yaml thiếu trường nc hoặc names")
            all_ok = False

    # 2. Kiểm tra data.yaml
    if not DATA_YAML_PATH.exists():
        print(f"  [X] Không tìm thấy file: {DATA_YAML_PATH}")
        all_ok = False
    else:
        with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
            local_data = yaml.safe_load(f)
        root_path = Path(local_data.get("path", ""))
        if root_path.exists():
            print(f"  [OK] data.yaml hợp lệ và đường dẫn dữ liệu tồn tại: {root_path}")
        else:
            print(f"  [!] data.yaml đường dẫn path không tồn tại: {root_path}")
            all_ok = False

    return all_ok


def audit_dataset_integrity(classes: dict[int, str]) -> dict:
    """
    Quét và kiểm tra tính toàn vẹn 100% dữ liệu đã xử lý trong data/processed:
    - Ảnh đọc được, không bị hỏng.
    - File nhãn khớp chính xác với ảnh.
    - Tọa độ bbox hợp lệ, không có bounding box ngoài biên [0, 1].
    - Tất cả class_id đều nằm trong {0, 1, 2, 3}.
    """
    print("\n--- 2. Kiểm tra toàn vẹn định lượng 100% dữ liệu ---")
    splits = ["train", "val", "test"]
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0,
        "splits": {},
        "class_counts": Counter(),
        "invalid_boxes": 0,
        "box_sizes": {"small": 0, "medium": 0, "large": 0},
    }

    for split in splits:
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split

        img_files = sorted(img_dir.glob("*.jpg"))
        lbl_files = sorted(lbl_dir.glob("*.txt"))

        img_names = {f.stem for f in img_files}
        lbl_names = {f.stem for f in lbl_files}

        mismatched_imgs = img_names - lbl_names
        mismatched_lbls = lbl_names - img_names

        split_boxes = 0
        split_class_counts = Counter()

        for img_path in img_files:
            base_name = img_path.stem
            lbl_path = lbl_dir / f"{base_name}.txt"

            # Xác nhận ảnh mở được và đọc kích thước W, H
            try:
                with Image.open(img_path) as img:
                    w_img, h_img = img.size
            except Exception as e:
                print(f"  [LỖI] Không thể đọc ảnh: {img_path} ({e})")
                continue

            # Đọc file nhãn
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]

            for parts in lines:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])

                # Kiểm tra tính hợp lệ
                if cid not in classes:
                    stats["invalid_boxes"] += 1
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    stats["invalid_boxes"] += 1

                # Phân loại kích thước hộp (pixel thực tế)
                box_area_px = (bw * w_img) * (bh * h_img)
                if box_area_px < 32 * 32:
                    stats["box_sizes"]["small"] += 1
                elif box_area_px < 96 * 96:
                    stats["box_sizes"]["medium"] += 1
                else:
                    stats["box_sizes"]["large"] += 1

                split_boxes += 1
                split_class_counts[cid] += 1
                stats["class_counts"][cid] += 1

        stats["total_images"] += len(img_files)
        stats["total_labels"] += len(lbl_files)
        stats["total_boxes"] += split_boxes
        stats["splits"][split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "boxes": split_boxes,
            "classes": dict(split_class_counts),
            "mismatch": len(mismatched_imgs) + len(mismatched_lbls),
        }

        print(f"  * Tập {split.upper():<5}: {len(img_files)} ảnh | {split_boxes} bounding boxes | Lệch file: {len(mismatched_imgs) + len(mismatched_lbls)}")

    print(f"\n  => Tổng cộng: {stats['total_images']} ảnh, {stats['total_boxes']} đối tượng.")
    print(f"  => Số hộp vi phạm giới hạn: {stats['invalid_boxes']}")
    print(f"  => Phân bố kích thước biển báo: Nhỏ (<32px): {stats['box_sizes']['small']} | Vừa (32-96px): {stats['box_sizes']['medium']} | Lớn (>96px): {stats['box_sizes']['large']}")

    return stats


def draw_bounding_boxes(
    img_path: Path,
    lbl_path: Path,
    classes: dict[int, str],
    font: ImageFont.ImageFont,
    line_width: int = 3
) -> Image.Image:
    """
    Vẽ bounding box và nhãn tên tiếng Việt sắc nét lên ảnh bằng Pillow.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size

    if not lbl_path.exists():
        return img

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split() for l in f if l.strip()]

    for parts in lines:
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        # Tọa độ pixel góc trên trái và góc dưới phải
        x1 = max(0, int((xc - bw / 2) * w_img))
        y1 = max(0, int((yc - bh / 2) * h_img))
        x2 = min(w_img - 1, int((xc + bw / 2) * w_img))
        y2 = min(h_img - 1, int((yc + bh / 2) * h_img))

        color = CLASS_COLORS.get(cid, (255, 0, 0))
        class_name = classes.get(cid, f"Lớp {cid}")
        label_text = f"[{cid}] {class_name}"

        # 1. Vẽ khung chữ nhật nổi bật
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 2. Đo kích thước chữ để tạo nền thẻ nhãn (tag background)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Vị trí tag nhãn: Ưu tiên đặt phía trên bounding box; nếu tràn mép thì đặt phía dưới
        tag_y2 = y1
        tag_y1 = max(0, y1 - text_h - 6)
        if tag_y1 == 0 and y1 < text_h + 6:
            tag_y1 = y1
            tag_y2 = y1 + text_h + 6

        tag_x1 = x1
        tag_x2 = min(w_img - 1, x1 + text_w + 10)

        # Vẽ nền thẻ nhãn màu tương ứng với lớp
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=color)

        # Vẽ chữ tiếng Việt màu trắng tương phản
        draw.text((tag_x1 + 4, tag_y1 + 1), label_text, fill=(255, 255, 255), font=font)

        # 3. Vẽ dấu tâm điểm (+) nhỏ hỗ trợ kiểm tra trực quan độ chuẩn xác
        cx_px = int(xc * w_img)
        cy_px = int(yc * h_img)
        cross_size = 4
        draw.line([cx_px - cross_size, cy_px, cx_px + cross_size, cy_px], fill=(255, 255, 255), width=1)
        draw.line([cx_px, cy_px - cross_size, cx_px, cy_px + cross_size], fill=(255, 255, 255), width=1)

    return img


def generate_visual_samples(
    classes: dict[int, str],
    output_dir: Path,
    samples_per_class: int = 2
) -> list[tuple[str, int, Path, Image.Image]]:
    """
    Chọn mẫu tiêu biểu của từng lớp ở cả 3 tập Train, Val, Test và xuất ảnh đã vẽ khung.
    """
    print(f"\n--- 3. Vẽ trực quan hóa bounding box vào {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(size=16)

    selected_samples = []  # Lưu (split, cid, path, img_annotated) để làm collage

    for split in ["train", "val", "test"]:
        split_out_dir = output_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split

        # Phân loại ảnh theo các lớp chứa bên trong
        class_to_files = defaultdict(list)
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_file, "r", encoding="utf-8") as f:
                cids = {int(line.split()[0]) for line in f if line.strip()}
            for cid in cids:
                class_to_files[cid].append(lbl_file.stem)

        print(f"  * Đang xuất ảnh trực quan tập {split.upper()}...")
        for cid in sorted(classes.keys()):
            available = class_to_files.get(cid, [])
            if not available:
                continue

            # Chọn ngẫu nhiên có cố định seed để tái lập
            rng = random.Random(42 + cid * 10)
            chosen_stems = rng.sample(available, min(samples_per_class, len(available)))

            for idx, stem in enumerate(chosen_stems):
                img_path = img_dir / f"{stem}.jpg"
                lbl_path = lbl_dir / f"{stem}.txt"

                if not img_path.exists():
                    continue

                annotated_img = draw_bounding_boxes(img_path, lbl_path, classes, font)
                out_filename = f"class_{cid}_{classes[cid].replace(' ', '_')}_{stem}.jpg"
                save_path = split_out_dir / out_filename
                annotated_img.save(save_path, quality=95)

                if idx == 0:
                    selected_samples.append((split, cid, save_path, annotated_img))

        print(f"    -> Đã lưu các ảnh kiểm tra vào: {split_out_dir}")

    return selected_samples


def build_collage_overview(
    selected_samples: list[tuple[str, int, Path, Image.Image]],
    output_path: Path,
    classes: dict[int, str]
) -> None:
    """
    Tạo ảnh ghép dạng lưới (Collage Grid) gồm đại diện 4 lớp x 3 tập (12 ô).
    """
    print(f"\n--- 4. Tạo ảnh lưới tổng quan đối chiếu (Collage Grid) ---")
    if not selected_samples:
        return

    # Sắp xếp các mẫu theo (cid, split)
    split_order = {"train": 0, "val": 1, "test": 2}
    grid_items = {}
    for split, cid, path, img in selected_samples:
        grid_items[(cid, split)] = img

    tile_w, tile_h = 360, 270
    cols = 3  # Train, Val, Test
    rows = 4  # 4 lớp biển báo

    header_h = 60
    margin = 10
    total_w = cols * tile_w + (cols + 1) * margin
    total_h = header_h + rows * tile_h + (rows + 1) * margin

    collage = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(collage)
    title_font = load_font(size=22)
    header_font = load_font(size=18)
    cell_font = load_font(size=14)

    # Tiêu đề chính
    title_text = "BẢNG KIỂM TRA TRỰC QUAN NHÃN BIỂN BÁO (TÂM - BƯỚC T4)"
    draw.text((margin + 10, 15), title_text, fill=(20, 20, 20), font=title_font)

    # Header từng cột (Train, Val, Test)
    split_names = ["TRAIN (560 ảnh)", "VAL (120 ảnh)", "TEST (120 ảnh)"]
    for c_idx, sp_name in enumerate(split_names):
        col_x = margin + c_idx * (tile_w + margin)
        draw.text((col_x + 10, header_h - 25), sp_name, fill=(80, 80, 80), font=header_font)

    for r_idx, cid in enumerate(sorted(classes.keys())):
        for c_idx, sp in enumerate(["train", "val", "test"]):
            x = margin + c_idx * (tile_w + margin)
            y = header_h + margin + r_idx * (tile_h + margin)

            img = grid_items.get((cid, sp))
            if img:
                thumb = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                collage.paste(thumb, (x, y))
                # Viền nhẹ quanh ô
                draw.rectangle([x, y, x + tile_w, y + tile_h], outline=(200, 200, 200), width=1)

                # Tag tên góc nhỏ
                info_text = f"Lớp {cid}: {classes[cid]} [{sp.upper()}]"
                draw.rectangle([x, y + tile_h - 22, x + tile_w, y + tile_h], fill=(0, 0, 0))
                draw.text((x + 6, y + tile_h - 20), info_text, fill=(255, 255, 255), font=cell_font)

    collage.save(output_path, quality=95)
    print(f"  [OK] Đã tạo thành công ảnh lưới tổng quan: {output_path}")


def main():
    print("=================================================================")
    print("   BƯỚC T4: KIỂM TRA TOÀN DIỆN CẤU HÌNH & TRỰC QUAN HÓA NHÃN    ")
    print("=================================================================")

    # Đọc bảng classes.json
    if not CLASSES_PATH.exists():
        print(f"[LỖI] Không tìm thấy file {CLASSES_PATH}")
        sys.exit(1)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_raw = json.load(f)
    classes = {int(k): v for k, v in classes_raw.items()}

    # 1. Kiểm tra cấu hình YAML
    yaml_ok = verify_yaml_configs()
    if not yaml_ok:
        print("[CẢNH BÁO] Cấu hình YAML có điểm chưa hoàn hảo, tiếp tục kiểm tra...")

    # 2. Kiểm tra toàn vẹn định lượng dữ liệu
    stats = audit_dataset_integrity(classes)

    # 3. Xuất ảnh trực quan hóa
    selected_samples = generate_visual_samples(
        classes=classes,
        output_dir=LABEL_VERIFICATION_DIR,
        samples_per_class=2
    )

    # 4. Tạo ảnh lưới tổng quan Collage Grid
    collage_path = LABEL_VERIFICATION_DIR / "verification_overview_grid.jpg"
    build_collage_overview(selected_samples, collage_path, classes)

    # 5. Lưu báo cáo kiểm định T4 thành file
    report_file = LABEL_VERIFICATION_DIR / "t4_verification_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=================================================================")
    print(f"[HOÀN THÀNH T4] Kết quả kiểm tra trực quan đã lưu tại:")
    print(f"  -> Thư mục ảnh kiểm tra: {LABEL_VERIFICATION_DIR}")
    print(f"  -> Ảnh lưới tổng hợp   : {collage_path}")
    print(f"  -> File báo cáo JSON   : {report_file}")
    print("=================================================================")


if __name__ == "__main__":
    main()
