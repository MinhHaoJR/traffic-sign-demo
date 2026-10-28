"""Tkinter demo for detecting four Vietnamese traffic sign classes."""

from __future__ import annotations

import queue
import threading
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

from src.paths import MODEL_PATH, PREDICTIONS_OUTPUT_DIR, ROOT_DIR


SUPPORTED_FORMATS = {"JPEG", "PNG"}


def read_image(path: str | Path) -> Image.Image:
    """Read a JPG/PNG, apply EXIF orientation, and flatten transparency on white."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise ValueError("Vui lòng chọn ảnh JPG, JPEG hoặc PNG.")
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            return background.convert("RGB")


def bgr_to_rgb_image(image_bgr) -> Image.Image:
    """Convert the Hào predictor's HxWx3 BGR array to a Pillow RGB image."""
    shape = getattr(image_bgr, "shape", None)
    if shape is None or len(shape) != 3 or shape[2] != 3:
        raise ValueError("Model trả về ảnh kết quả không đúng định dạng H×W×3.")
    height, width = int(shape[0]), int(shape[1])
    # Pillow's raw decoder performs BGR -> RGB and avoids an extra OpenCV import.
    return Image.frombytes("RGB", (width, height), image_bgr.tobytes(), "raw", "BGR")


class ImagePanel(ttk.Frame):
    """Resizable image preview that keeps the source image's aspect ratio."""

    def __init__(self, parent, title: str, placeholder: str):
        super().__init__(parent, style="Card.TFrame", padding=12)
        self.source: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.placeholder = placeholder
        self._resize_job = None

        ttk.Label(self, text=title, style="CardTitle.TLabel").pack(anchor="w")
        self.canvas = tk.Canvas(
            self,
            background="#f1f5f9",
            highlightthickness=0,
            width=400,
            height=280,
        )
        self.canvas.pack(fill="both", expand=True, pady=(10, 0))
        self.canvas.bind("<Configure>", self._schedule_render)

    def set_image(self, image: Image.Image | None, placeholder: str | None = None):
        self.source = image
        if placeholder is not None:
            self.placeholder = placeholder
        self._render()

    def _schedule_render(self, _event=None):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self._render)

    def _render(self):
        self._resize_job = None
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        if self.source is None:
            self.photo = None
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=self.placeholder,
                fill="#64748b",
                font=("Segoe UI", 11),
                width=max(1, width - 40),
                justify="center",
            )
            return

        preview = self.source.copy()
        preview.thumbnail(
            (max(1, width - 20), max(1, height - 20)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(preview, master=self.canvas)
        self.canvas.create_image(width / 2, height / 2, image=self.photo)


class TrafficSignApp(tk.Tk):
    """Desktop UI; model loading and inference run outside Tk's main thread."""

    def __init__(self):
        super().__init__()
        self.title("Nhận diện biển báo giao thông")
        self.geometry("1180x820")
        self.minsize(940, 650)
        self.configure(background="#e8edf3")
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.selected_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.annotated_image: Image.Image | None = None
        self._model = None
        self._model_state = "loading"
        self._model_error = ""
        self._predicting = False
        self._input_revision = 0
        self._result_revision: int | None = None
        self._closed = False
        self._worker_events: queue.Queue = queue.Queue()

        self.confidence = tk.DoubleVar(value=0.25)
        self.confidence_text = tk.StringVar(value="0.25")
        self.status = tk.StringVar(value="Đang khởi động ứng dụng...")
        self.file_info = tk.StringVar(value="Chưa chọn ảnh")

        self._configure_styles()
        self._build_ui()
        self._load_model_once()
        self.after(80, self._process_worker_events)

    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#e8edf3")
        style.configure("TLabel", background="#e8edf3", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#0f172a")
        style.configure("Muted.TLabel", foreground="#475569")
        style.configure("Card.TFrame", background="white")
        style.configure(
            "CardTitle.TLabel",
            background="white",
            foreground="#0f172a",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=27)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        body = ttk.Frame(self, padding=18)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)

        ttk.Label(
            body,
            text="Nhận diện biển báo giao thông",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            body,
            text="Demo nhận diện bốn loại biển báo cấm bằng YOLO11n.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 12))

        controls = ttk.Frame(body)
        controls.grid(row=2, column=0, sticky="ew")
        self.choose_button = ttk.Button(
            controls, text="Chọn ảnh", command=self.choose_image
        )
        self.choose_button.pack(side="left")
        self.detect_button = ttk.Button(
            controls,
            text="Nhận diện",
            command=self.detect_image,
            state="disabled",
        )
        self.detect_button.pack(side="left", padx=(8, 0))
        self.save_button = ttk.Button(
            controls,
            text="Lưu kết quả",
            command=self.save_result,
            state="disabled",
        )
        self.save_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(
            controls,
            text="Xóa ảnh",
            command=self.clear_image,
            state="disabled",
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        ttk.Label(controls, text="Ngưỡng tin cậy").pack(side="left", padx=(18, 8))
        self.confidence_scale = ttk.Scale(
            controls,
            from_=0.05,
            to=0.95,
            variable=self.confidence,
            command=self._confidence_changed,
            length=150,
        )
        self.confidence_scale.pack(side="left")
        ttk.Label(
            controls,
            textvariable=self.confidence_text,
            width=5,
            anchor="w",
        ).pack(side="left", padx=(7, 0))

        ttk.Label(
            body,
            textvariable=self.file_info,
            style="Muted.TLabel",
            wraplength=1080,
        ).grid(row=3, column=0, sticky="w", pady=(10, 8))

        images = ttk.Frame(body)
        images.grid(row=4, column=0, sticky="nsew")
        images.columnconfigure(0, weight=1, uniform="image_panels")
        images.columnconfigure(1, weight=1, uniform="image_panels")
        images.rowconfigure(0, weight=1)
        self.original_panel = ImagePanel(
            images,
            "Ảnh gốc",
            "Chưa có ảnh\nBấm Chọn ảnh để mở ảnh từ máy.",
        )
        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.result_panel = ImagePanel(
            images,
            "Ảnh kết quả",
            "Kết quả sẽ xuất hiện sau khi nhận diện.",
        )
        self.result_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ttk.Label(
            body,
            text="Danh sách biển báo",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=5, column=0, sticky="w", pady=(12, 6))
        self.results = ttk.Treeview(
            body,
            columns=("class_name", "confidence", "bbox"),
            show="headings",
            height=4,
        )
        self.results.heading("class_name", text="Loại biển báo")
        self.results.heading("confidence", text="Độ tin cậy")
        self.results.heading("bbox", text="Vị trí [x1, y1, x2, y2]")
        self.results.column("class_name", width=330, minwidth=180)
        self.results.column("confidence", width=130, minwidth=100, anchor="center")
        self.results.column("bbox", width=330, minwidth=200, anchor="center")
        self.results.grid(row=6, column=0, sticky="ew")

        ttk.Label(
            body,
            textvariable=self.status,
            style="Muted.TLabel",
            wraplength=1080,
        ).grid(row=7, column=0, sticky="w", pady=(9, 0))

    def _load_model_once(self):
        if not Path(MODEL_PATH).is_file():
            self._model_state = "error"
            self._model_error = f"Không tìm thấy trọng số: {MODEL_PATH}"
            self.status.set(f"{self._model_error}. Xem models/README.md để biết cách chuẩn bị model.")
            return

        self.status.set("Đang nạp model một lần; bạn vẫn có thể chọn ảnh.")
        threading.Thread(target=self._model_worker, daemon=True).start()

    def _model_worker(self):
        try:
            # Import here so the image viewer still opens if model dependencies are absent.
            from src.predict import load_model

            model = load_model(MODEL_PATH)
            self._worker_events.put(("model_ready", model))
        except Exception as error:
            self._worker_events.put(("model_error", f"{type(error).__name__}: {error}"))

    def _process_worker_events(self):
        if self._closed:
            return
        while True:
            try:
                event = self._worker_events.get_nowait()
            except queue.Empty:
                break
            kind, *payload = event
            if kind == "model_ready":
                self._model = payload[0]
                self._model_state = "ready"
                if self.selected_path is None:
                    self.status.set("Model đã sẵn sàng. Chọn ảnh để bắt đầu.")
                else:
                    self.status.set("Model đã sẵn sàng. Bấm Nhận diện để xử lý ảnh hiện tại.")
                self._update_controls()
            elif kind == "model_error":
                self._model_state = "error"
                self._model_error = payload[0]
                self.status.set(f"Không nạp được model: {self._model_error}")
                self._update_controls()
            elif kind == "prediction_done":
                self._finish_prediction(*payload)
            elif kind == "prediction_error":
                self._fail_prediction(*payload)
        self.after(80, self._process_worker_events)

    def _update_controls(self):
        has_image = self.selected_path is not None
        model_ready = self._model_state == "ready" and self._model is not None
        self.detect_button.configure(
            state="normal" if has_image and model_ready and not self._predicting else "disabled"
        )
        has_current_result = (
            self.annotated_image is not None
            and self._result_revision == self._input_revision
        )
        self.save_button.configure(state="normal" if has_current_result else "disabled")
        self.clear_button.configure(state="normal" if has_image else "disabled")

    def choose_image(self):
        filename = filedialog.askopenfilename(
            parent=self,
            title="Chọn ảnh biển báo",
            filetypes=[
                ("Ảnh JPG và PNG", "*.jpg *.jpeg *.png"),
                ("Tất cả file", "*.*"),
            ],
        )
        if not filename:
            return
        try:
            image = read_image(filename)
        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            messagebox.showerror("Không mở được ảnh", str(error), parent=self)
            return

        self._input_revision += 1
        self.selected_path = Path(filename)
        self.original_image = image
        self.original_panel.set_image(image)
        self._clear_result_view()
        self.file_info.set(
            f"{self.selected_path.name}  |  {image.width} × {image.height} pixel"
        )
        if self._predicting:
            self.status.set("Đã đổi ảnh. Kết quả của lần nhận diện trước sẽ bị bỏ qua.")
        elif self._model_state == "ready":
            self.status.set("Đã mở ảnh. Bấm Nhận diện để xử lý.")
        elif self._model_state == "loading":
            self.status.set("Đã mở ảnh. Model đang được nạp.")
        else:
            self.status.set(f"Đã mở ảnh, nhưng model chưa sẵn sàng: {self._model_error}")
        self._update_controls()

    def clear_image(self):
        self._input_revision += 1
        self.selected_path = None
        self.original_image = None
        self.original_panel.set_image(None, "Chưa có ảnh\nBấm Chọn ảnh để mở ảnh từ máy.")
        self._clear_result_view()
        self.file_info.set("Chưa chọn ảnh")
        self.status.set("Đã xóa ảnh. Chọn một ảnh khác để tiếp tục.")
        self._update_controls()

    def _clear_result_view(self):
        self.annotated_image = None
        self._result_revision = None
        self.result_panel.set_image(None, "Kết quả sẽ xuất hiện sau khi nhận diện.")
        self.results.delete(*self.results.get_children())
        self._update_controls()

    def _confidence_changed(self, value):
        self.confidence_text.set(f"{float(value):.2f}")
        self._input_revision += 1
        had_result = self.annotated_image is not None
        self._clear_result_view()
        if self._predicting:
            self.status.set("Đã đổi ngưỡng. Kết quả đang chạy với ngưỡng cũ sẽ bị bỏ qua.")
        elif had_result:
            self.status.set("Đã đổi ngưỡng tin cậy. Hãy nhận diện lại để cập nhật kết quả.")
        self._update_controls()

    def detect_image(self):
        if self._model_state != "ready" or self._model is None:
            self.status.set("Model chưa sẵn sàng. Kiểm tra models/best.pt và thư viện cài đặt.")
            return
        if self.selected_path is None:
            self.status.set("Hãy chọn ảnh trước khi nhận diện.")
            return
        if self._predicting:
            return

        path = self.selected_path
        revision = self._input_revision
        confidence = float(self.confidence.get())
        model = self._model
        self._predicting = True
        self._clear_result_view()
        self.status.set("Đang nhận diện. Cửa sổ vẫn hoạt động; có thể chọn ảnh khác.")
        self._update_controls()
        threading.Thread(
            target=self._prediction_worker,
            args=(model, path, confidence, revision),
            daemon=True,
        ).start()

    def _prediction_worker(self, model, path: Path, confidence: float, revision: int):
        try:
            from src.predict import predict_image

            annotated_bgr, detections = predict_image(model, str(path), confidence)
            self._worker_events.put(
                ("prediction_done", revision, annotated_bgr, detections)
            )
        except Exception as error:
            self._worker_events.put(
                ("prediction_error", revision, f"{type(error).__name__}: {error}")
            )

    def _finish_prediction(self, revision: int, annotated_bgr, detections: list):
        self._predicting = False
        if revision != self._input_revision or self.selected_path is None:
            self._update_controls()
            return

        try:
            image = bgr_to_rgb_image(annotated_bgr)
        except (AttributeError, TypeError, ValueError) as error:
            self._fail_prediction(revision, f"Kết quả model không hợp lệ: {error}")
            return

        self.annotated_image = image
        self._result_revision = revision
        self.result_panel.set_image(image, "Ảnh đã được xử lý.")
        self.results.delete(*self.results.get_children())
        for detection in detections:
            bbox = detection.get("bbox", [])
            bbox_text = ", ".join(str(int(value)) for value in bbox)
            self.results.insert(
                "",
                "end",
                values=(
                    detection.get("class_name", "Không rõ"),
                    f"{float(detection.get('confidence', 0.0)):.2f}",
                    f"[{bbox_text}]",
                ),
            )

        if detections:
            self.status.set(f"Đã phát hiện {len(detections)} biển báo.")
        else:
            self.status.set("Không phát hiện biển báo thuộc bốn lớp hỗ trợ trong ảnh này.")
        self._update_controls()

    def _fail_prediction(self, revision: int, error_message: str):
        self._predicting = False
        if revision != self._input_revision or self.selected_path is None:
            self._update_controls()
            return
        self._clear_result_view()
        self.status.set(f"Không nhận diện được ảnh: {error_message}")
        self._update_controls()
        messagebox.showerror(
            "Không nhận diện được ảnh",
            error_message,
            parent=self,
        )

    def save_result(self):
        if (
            self.annotated_image is None
            or self._result_revision != self._input_revision
            or self.selected_path is None
        ):
            self.status.set("Chưa có kết quả hiện tại để lưu. Hãy nhận diện ảnh trước.")
            self._update_controls()
            return

        suggested_name = f"{self.selected_path.stem}_ket_qua.jpg"
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="Lưu ảnh kết quả",
            initialdir=str(PREDICTIONS_OUTPUT_DIR if PREDICTIONS_OUTPUT_DIR.exists() else ROOT_DIR),
            initialfile=suggested_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")],
        )
        if not filename:
            return

        target = Path(filename)
        if target.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            messagebox.showerror(
                "Định dạng chưa hỗ trợ",
                "Hãy lưu ảnh với phần mở rộng .jpg, .jpeg hoặc .png.",
                parent=self,
            )
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            image_format = "PNG" if target.suffix.lower() == ".png" else "JPEG"
            self.annotated_image.save(target, format=image_format, quality=95)
        except (OSError, ValueError) as error:
            messagebox.showerror("Không lưu được ảnh", str(error), parent=self)
            self.status.set(f"Không lưu được ảnh: {error}")
            return

        self.status.set(f"Đã lưu ảnh kết quả: {target}")
        messagebox.showinfo("Đã lưu kết quả", f"Ảnh đã được lưu tại:\n{target}", parent=self)

    def close(self):
        self._closed = True
        self.destroy()


if __name__ == "__main__":
    TrafficSignApp().mainloop()
