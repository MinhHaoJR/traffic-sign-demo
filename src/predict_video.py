"""
Nhận diện biển báo trên video: đọc từng khung, vẽ khung nhận diện, ghi ra MP4.

    python -m src.predict_video video.mp4 [confidence] [--out ket_qua.mp4]

Giao diện Tkinter gọi predict_video(...) với on_frame để xem trực tiếp và stop để dừng giữa chừng.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from src.paths import PREDICTIONS_OUTPUT_DIR
from src.predict import load_model, predict_frame

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def default_output(video_path) -> Path:
    return PREDICTIONS_OUTPUT_DIR / "videos" / f"{Path(video_path).stem}_ket_qua.mp4"


def predict_video(model, video_path, out_path, confidence: float = 0.25, on_frame=None, stop=None) -> dict:
    """
    on_frame(frame_bgr, annotated_bgr, detections, index, total) được gọi sau mỗi khung;
    stop() trả True để dừng sớm. Trả về thống kê; phần video đã xử lý vẫn được lưu khi dừng.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video (file hỏng hoặc sai định dạng): {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer, index, frames_with_signs = None, 0, 0
    per_class = Counter()  # số khung có mặt mỗi lớp
    start = time.time()
    try:
        while not (stop and stop()):
            ok, frame = cap.read()
            if not ok:
                break
            annotated, dets = predict_frame(model, frame, confidence)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise OSError(f"Không ghi được video ra: {out_path}")
            writer.write(annotated)
            index += 1
            frames_with_signs += bool(dets)
            per_class.update({d["class_name"] for d in dets})
            if on_frame:
                on_frame(frame, annotated, dets, index, total)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if index == 0:
        raise ValueError(f"Video không có khung hình nào đọc được: {video_path}")
    return {
        "frames": index, "total": total, "stopped": bool(stop and stop()), "fps": fps,
        "seconds": time.time() - start, "frames_with_signs": frames_with_signs,
        "per_class": dict(per_class.most_common()), "out_path": out_path,
    }


def main():
    p = argparse.ArgumentParser(description="Nhận diện biển báo trên video và lưu MP4 có khung.")
    p.add_argument("video")
    p.add_argument("confidence", nargs="?", type=float, default=0.25)
    p.add_argument("--out", type=Path, help="Mặc định: outputs/predictions/videos/<tên>_ket_qua.mp4")
    a = p.parse_args()

    def progress(_f, _a, _d, i, total):
        if i % 25 == 0:
            print(f"\r  {i}/{total or '?'} khung", end="", flush=True)

    s = predict_video(load_model(), a.video, a.out or default_output(a.video), a.confidence, on_frame=progress)
    print(f"\nĐã xử lý {s['frames']} khung trong {s['seconds']:.1f}s ({s['frames'] / s['seconds']:.1f} khung/s)")
    print(f"Khung có biển: {s['frames_with_signs']}")
    for name, n in s["per_class"].items():
        print(f"  {name}: {n} khung")
    print("Đã lưu:", s["out_path"])


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
