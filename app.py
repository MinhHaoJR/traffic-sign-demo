"""Tkinter demo nhận diện biển báo Việt Nam trên ảnh/video, có nút dạy biển mới (src.few_shot)."""

from __future__ import annotations

import queue
import threading
import time
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
        self.box: list[float] | None = None  # khung người dùng vẽ, theo tọa độ ảnh gốc
        self._scale, self._offset = 1.0, (0.0, 0.0)
        self._drag_start = None
        self._on_box = None

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
        self.box = None
        if placeholder is not None:
            self.placeholder = placeholder
        self._render()

    def enable_box_drawing(self, on_box):
        """Kéo chuột trên ảnh để vẽ khung; on_box(box | None) nhận tọa độ [x1, y1, x2, y2] của ảnh gốc."""
        self._on_box = on_box
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)

    def clear_box(self):
        self.box = None
        self.canvas.delete("drawn")

    def _drag_begin(self, event):
        if self.source is not None:
            self._drag_start = (event.x, event.y)

    def _drag_move(self, event):
        if self._drag_start is None:
            return
        self.canvas.delete("drawn")
        self.canvas.create_rectangle(*self._drag_start, event.x, event.y, outline="#f97316", width=2, tags="drawn")

    def _drag_end(self, event):
        if self._drag_start is None:
            return
        (x0, y0), self._drag_start = self._drag_start, None
        ox, oy = self._offset
        w, h = self.source.size
        xs = sorted(min(max((x - ox) / self._scale, 0), w) for x in (x0, event.x))
        ys = sorted(min(max((y - oy) / self._scale, 0), h) for y in (y0, event.y))
        self.box = [xs[0], ys[0], xs[1], ys[1]] if xs[1] - xs[0] >= 8 and ys[1] - ys[0] >= 8 else None
        self._render()
        if self._on_box:
            self._on_box(self.box)

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
        self._scale = preview.width / self.source.width
        self._offset = (width / 2 - preview.width / 2, height / 2 - preview.height / 2)
        if self.box is not None:
            ox, oy = self._offset
            x1, y1, x2, y2 = (v * self._scale for v in self.box)
            self.canvas.create_rectangle(x1 + ox, y1 + oy, x2 + ox, y2 + oy, outline="#f97316", width=2, tags="drawn")


def ask_sign_name(parent, choices: list[str], initial: str = "") -> str | None:
    """Hộp thoại chọn tên có sẵn hoặc gõ tên biển mới; trả None nếu hủy."""
    dialog = tk.Toplevel(parent)
    dialog.title("Đặt tên biển báo")
    dialog.transient(parent)
    dialog.resizable(False, False)
    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Chọn tên có sẵn hoặc gõ tên biển mới:").pack(anchor="w")
    value = tk.StringVar(value=initial)
    box = ttk.Combobox(frame, textvariable=value, values=choices, width=46)
    box.pack(fill="x", pady=(6, 12))
    result: list[str | None] = [None]

    def accept(_event=None):
        if value.get().strip():
            result[0] = value.get().strip()
            dialog.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(anchor="e")
    ttk.Button(buttons, text="Hủy", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Học biển này", command=accept).pack(side="right", padx=(0, 8))
    dialog.bind("<Return>", accept)
    dialog.bind("<Escape>", lambda _e: dialog.destroy())
    box.focus_set()
    dialog.grab_set()
    parent.wait_window(dialog)
    return result[0]


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
        self._teaching = False
        self._teach_note = ""
        self._detections: list = []  # kết quả ảnh hiện tại, theo thứ tự hàng trong bảng
        self._video_running = False
        self._video_stop = threading.Event()
        self._video_resume = threading.Event()  # set = chạy, clear = đang xin tạm dừng (luồng video chặn chờ)
        self._video_resume.set()
        self._paused: tuple[Image.Image, str] | None = None  # (khung đang dừng, tên gợi ý) khi đã tạm dừng thật
        self._video_stem = ""
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
            text="Nhận diện 28 loại biển báo trên ảnh và video bằng YOLO11n; biển lạ có thể dạy cho hệ thống học ngay.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 12))

        controls = ttk.Frame(body)
        controls.grid(row=2, column=0, sticky="ew")
        self.choose_button = ttk.Button(
            controls, text="Chọn ảnh", command=self.choose_image
        )
        self.choose_button.pack(side="left")
        self.video_button = ttk.Button(
            controls, text="Chọn video", command=self.choose_video, state="disabled"
        )
        self.video_button.pack(side="left", padx=(8, 0))
        self.pause_button = ttk.Button(
            controls, text="Tạm dừng", command=self.toggle_pause, state="disabled"
        )
        self.pause_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(
            controls, text="Dừng video", command=self.stop_video, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
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
        # vẽ trên ảnh kết quả: cùng hệ tọa độ với ảnh model xử lý (ảnh gốc hiển thị đã xoay theo EXIF)
        self.result_panel.enable_box_drawing(lambda _box: self._update_controls())

        list_header = ttk.Frame(body)
        list_header.grid(row=5, column=0, sticky="ew", pady=(12, 6))
        ttk.Label(
            list_header,
            text="Danh sách biển báo",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        self.teach_button = ttk.Button(
            list_header, text="Đặt tên biển…", command=self.teach_selected, state="disabled"
        )
        self.teach_button.pack(side="right")
        ttk.Label(
            list_header,
            text="Chọn một hàng, hoặc kéo chuột khoanh biển bị bỏ sót trên Ảnh kết quả, rồi đặt tên để hệ thống học.",
            style="Muted.TLabel",
        ).pack(side="right", padx=(0, 10))
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
        self.results.bind("<<TreeviewSelect>>", lambda _e: self._update_controls())

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
        latest_frame = None
        while True:
            try:
                event = self._worker_events.get_nowait()
            except queue.Empty:
                break
            kind, *payload = event
            if kind == "video_frame":
                latest_frame = payload  # chỉ vẽ khung mới nhất để giao diện không bị dồn hàng
                continue
            if kind == "video_done":
                self._finish_video(*payload)
                continue
            if kind == "video_error":
                self._fail_video(*payload)
                continue
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
            elif kind == "teach_done":
                self._finish_teach(*payload)
        if latest_frame is not None:
            self._show_video_frame(*latest_frame)
        self.after(80, self._process_worker_events)

    def _update_controls(self):
        has_image = self.selected_path is not None
        model_ready = self._model_state == "ready" and self._model is not None
        video = self._video_running
        self.detect_button.configure(
            state="normal" if has_image and model_ready and not self._predicting and not video else "disabled"
        )
        has_current_result = (
            self.annotated_image is not None
            and self._result_revision == self._input_revision
        )
        paused = self._paused is not None
        self.save_button.configure(state="normal" if (has_current_result and not video) or paused else "disabled")
        self.clear_button.configure(state="normal" if has_image and not video else "disabled")
        self.choose_button.configure(state="disabled" if video else "normal")
        self.video_button.configure(
            state="normal" if model_ready and not self._predicting and not video else "disabled"
        )
        self.stop_button.configure(state="normal" if video else "disabled")
        pausing = not paused and not self._video_resume.is_set()  # đã bấm Tạm dừng, luồng chưa dừng hẳn
        self.pause_button.configure(
            text="Chạy tiếp" if paused else "Tạm dừng",
            state="normal" if video and not pausing and not self._video_stop.is_set() else "disabled",
        )
        self.confidence_scale.state(["disabled"] if video else ["!disabled"])
        can_teach = (has_current_result and not video and not self._predicting and not self._teaching
                     and (bool(self.results.selection()) or self.result_panel.box is not None))
        self.teach_button.configure(state="normal" if can_teach else "disabled")

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
        self._detections = []
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
        self._fill_results(detections)

        if detections:
            unknown = sum(d.get("class_name") == "Biển chưa biết" for d in detections)
            hint = f" Có {unknown} biển chưa biết: chọn hàng và bấm Đặt tên biển để dạy." if unknown else ""
            self.status.set(f"Đã phát hiện {len(detections)} biển báo.{hint}{self._teach_note}")
        else:
            self.status.set(f"Không phát hiện biển báo nào trong ảnh này.{self._teach_note}")
        self._teach_note = ""
        self._update_controls()

    def _fill_results(self, detections: list):
        self.results.delete(*self.results.get_children())
        self._detections = list(detections)
        for detection in detections:
            bbox = detection.get("bbox", [])
            bbox_text = ", ".join(str(int(value)) for value in bbox)
            self.results.insert(
                "",
                "end",
                values=(
                    detection.get("class_name", "Không rõ") + (" (học nhanh)" if detection.get("few_shot") else ""),
                    f"{float(detection.get('confidence', 0.0)):.2f}",
                    f"[{bbox_text}]",
                ),
            )

    def teach_selected(self):
        """Đặt tên cho biển đang chọn: học nhanh ngay + lưu vào kho phản hồi để train lại sau."""
        if self.selected_path is None or self._teaching:
            return
        from src.few_shot import OTHER_NAME, UNKNOWN_NAME, load_classes

        drawn = self.result_panel.box
        selection = self.results.selection()
        if drawn is not None:  # khung tự vẽ được ưu tiên hơn hàng đang chọn
            bbox, current, others = [int(round(v)) for v in drawn], "", list(self._detections)
        elif selection and self.results.index(selection[0]) < len(self._detections):
            index = self.results.index(selection[0])
            bbox, current = self._detections[index]["bbox"], self._detections[index].get("class_name", "")
            others = [d for i, d in enumerate(self._detections) if i != index]
        else:
            return
        choices = [n for n in load_classes().values() if n != OTHER_NAME]
        name = ask_sign_name(self, choices, "" if current == UNKNOWN_NAME else current)
        if not name:
            return
        self._teaching = True
        self.status.set(f"Đang học biển \"{name}\"...")
        self._update_controls()
        threading.Thread(
            target=self._teach_worker,
            args=(self.selected_path, bbox, name, others, self._input_revision),
            daemon=True,
        ).start()

    def _teach_worker(self, path: Path, bbox, name: str, others: list, revision: int):
        try:
            from src.few_shot import add_class, feedback_counts
            from src.predict import _read_bgr, few_shot

            store = few_shot()
            if store is None:
                raise FileNotFoundError(
                    "Chưa có bộ học nhanh (models/sign_encoder.pt, models/few_shot.npz). Xem README mục 7."
                )
            class_id = add_class(name)
            saved = store.teach(_read_bgr(path), bbox, class_id, others)
            count = feedback_counts().get(class_id, 0)
            self._worker_events.put(("teach_done", revision, name, count, str(saved), None))
        except Exception as error:
            self._worker_events.put(("teach_done", revision, name, 0, "", f"{type(error).__name__}: {error}"))

    def _finish_teach(self, revision: int, name: str, count: int, saved: str, error: str | None):
        self._teaching = False
        self._update_controls()
        if error:
            self.status.set(f"Không học được biển: {error}")
            messagebox.showerror("Không học được biển", error, parent=self)
            return
        self._teach_note = (
            f" Đã học biển \"{name}\" ({count} ảnh trong kho phản hồi). Dạy thêm ảnh khác của biển này để "
            "nhận chắc hơn; đủ khoảng 30 ảnh thì chạy python -m src.learn_new để train lại model."
        )
        self.status.set(self._teach_note.strip())
        if revision == self._input_revision and self.selected_path is not None:
            self.detect_image()  # nhận diện lại để thấy tên vừa dạy; ghi chú được nối vào trạng thái kết quả

    def choose_video(self):
        if self._model_state != "ready" or self._predicting or self._video_running:
            return
        source = filedialog.askopenfilename(
            parent=self,
            title="Chọn video biển báo",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả file", "*.*")],
        )
        if not source:
            return
        out_dir = PREDICTIONS_OUTPUT_DIR / "videos"
        target = filedialog.asksaveasfilename(
            parent=self,
            title="Lưu video kết quả",
            initialdir=str(out_dir if out_dir.exists() else ROOT_DIR),
            initialfile=f"{Path(source).stem}_ket_qua.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if not target:
            return
        if Path(target).resolve() == Path(source).resolve():
            messagebox.showerror("Không lưu được video", "Hãy chọn file kết quả khác file video gốc.", parent=self)
            return

        self.clear_image()
        revision = self._input_revision
        self._video_running = True
        self._video_stop.clear()
        self._reset_pause()
        self._video_stem = Path(source).stem
        self.file_info.set(f"Video: {Path(source).name}  →  {Path(target).name}")
        self.status.set("Đang mở video...")
        self._update_controls()
        threading.Thread(
            target=self._video_worker,
            args=(self._model, source, target, float(self.confidence.get()), revision),
            daemon=True,
        ).start()

    def _video_worker(self, model, source: str, target: str, confidence: float, revision: int):
        last_sent = [0.0]

        def on_frame(frame, annotated, detections, index, total):
            now = time.monotonic()
            if now - last_sent[0] >= 0.08 or index == total:
                last_sent[0] = now
                self._worker_events.put(("video_frame", revision, frame, annotated, detections, index, total))

        def on_pause(frame, annotated, detections, index, total):  # luôn gửi đúng khung mà luồng dừng lại
            self._worker_events.put(("video_frame", revision, frame, annotated, detections, index, total, True))

        try:
            from src.predict_video import predict_video

            summary = predict_video(
                model, source, target, confidence, on_frame=on_frame, stop=self._video_stop.is_set,
                resume=self._video_resume, on_pause=on_pause,
            )
            self._worker_events.put(("video_done", revision, summary))
        except Exception as error:
            self._worker_events.put(("video_error", revision, f"{type(error).__name__}: {error}"))

    def _show_video_frame(self, revision, frame, annotated, detections, index, total, paused=False):
        if revision != self._input_revision:
            return
        image = bgr_to_rgb_image(annotated)
        self.original_panel.set_image(bgr_to_rgb_image(frame))
        self.result_panel.set_image(image)
        self._fill_results(detections)
        if paused and not self._video_resume.is_set():  # bỏ qua tin cũ nếu đã bấm Chạy tiếp hoặc Dừng
            self._paused = (image, f"{self._video_stem}_khung_{index}.jpg")
            self.status.set(
                f"Đã tạm dừng ở khung {index}/{total or '?'}. Bấm Lưu kết quả để lưu khung này, "
                "Chạy tiếp để xử lý tiếp hoặc Dừng video để dừng hẳn."
            )
            self._update_controls()
        elif self._video_running and self._video_resume.is_set() and not self._video_stop.is_set():
            self.status.set(
                f"Đang nhận diện video: khung {index}/{total or '?'}. Bấm Tạm dừng để xem hoặc lưu khung, "
                "Dừng video để dừng hẳn."
            )

    def toggle_pause(self):
        if not self._video_running or self._video_stop.is_set():
            return
        if self._paused is not None:
            self._reset_pause()
            self.status.set("Chạy tiếp video từ khung đang dừng...")
        else:
            self._video_resume.clear()
            self.status.set("Đang tạm dừng, chờ xử lý xong khung hiện tại...")
        self._update_controls()

    def _reset_pause(self):
        """Xóa trạng thái tạm dừng và mở khóa luồng video (chạy tiếp, dừng hẳn, xong, lỗi hoặc đóng app)."""
        self._paused = None
        self._video_resume.set()

    def stop_video(self):
        if self._video_running:
            self._video_stop.set()
            self._reset_pause()
            self.status.set("Đang dừng video, phần đã xử lý vẫn được lưu...")
            self._update_controls()

    def _finish_video(self, revision: int, summary: dict):
        self._video_running = False
        self._reset_pause()
        self._update_controls()
        if revision != self._input_revision:
            return
        seen = ", ".join(f"{name} ({n} khung)" for name, n in summary["per_class"].items()) or "không có"
        state = "Đã dừng" if summary["stopped"] else "Xong"
        self.status.set(
            f"{state}: {summary['frames']} khung, {summary['frames'] / max(summary['seconds'], 1e-6):.1f} khung/s. "
            f"Biển xuất hiện: {seen}. Đã lưu: {summary['out_path']}"
        )

    def _fail_video(self, revision: int, error_message: str):
        self._video_running = False
        self._reset_pause()
        self._update_controls()
        if revision != self._input_revision:
            return
        self.status.set(f"Không nhận diện được video: {error_message}")
        messagebox.showerror("Không nhận diện được video", error_message, parent=self)

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
        if self._paused is not None:  # đang tạm dừng video: chỉ lưu khung đang dừng
            image, suggested_name = self._paused
        elif (
            self.annotated_image is None
            or self._result_revision != self._input_revision
            or self.selected_path is None
        ):
            self.status.set("Chưa có kết quả hiện tại để lưu. Hãy nhận diện ảnh trước.")
            self._update_controls()
            return
        else:
            image, suggested_name = self.annotated_image, f"{self.selected_path.stem}_ket_qua.jpg"
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
            image.save(target, format=image_format, quality=95)
        except (OSError, ValueError) as error:
            messagebox.showerror("Không lưu được ảnh", str(error), parent=self)
            self.status.set(f"Không lưu được ảnh: {error}")
            return

        self.status.set(f"Đã lưu {'khung hình' if self._paused is not None else 'ảnh kết quả'}: {target}")
        messagebox.showinfo("Đã lưu kết quả", f"Ảnh đã được lưu tại:\n{target}", parent=self)

    def close(self):
        self._video_stop.set()
        self._reset_pause()  # luồng video đang chờ ở trạng thái tạm dừng phải được mở khóa
        self._closed = True
        self.destroy()


if __name__ == "__main__":
    TrafficSignApp().mainloop()
