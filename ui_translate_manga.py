from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "漫画翻译工具"
SETTINGS_FILE = "ui_settings.json"
AI_PROFILE_FILE = "ai_profile.json"


def detect_default_font_path() -> str:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return ""


class MangaTranslatorUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("920x620")
        self.root.minsize(780, 520)

        self.base_dir = Path(__file__).resolve().parent
        self.python_exe = self.base_dir / ".venv" / "Scripts" / "python.exe"
        self.translate_script = self.base_dir / "translate_manga.py"
        self.settings_path = self.base_dir / SETTINGS_FILE
        self.ai_profile_path = self.base_dir / AI_PROFILE_FILE

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = False
        self.temp_input_dir: Path | None = None

        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.translator_backend = tk.StringVar(value="local")
        self.ai_endpoint = tk.StringVar()
        self.ai_api_key = tk.StringVar()
        self.ai_model = tk.StringVar()
        self.force_rebuild = tk.BooleanVar(value=True)
        self.text_detector = tk.StringVar(value="comic-text-detector")
        self.inpaint_backend = tk.StringVar(value="lama")
        self.conf_threshold = tk.StringVar(value="0.25")
        self.margin = tk.StringVar(value="8")
        self.font_path = tk.StringVar(value=detect_default_font_path())
        self.status_text = tk.StringVar(value="就绪")
        self.mode_hint_text = tk.StringVar(value="当前模式：本地翻译，适合离线处理。")
        self.output_summary_text = tk.StringVar(value="输出：未选择")
        self.ai_saved_text = tk.StringVar(value="AI 参数状态：未保存")
        self.selected_files: list[Path] = []

        self._configure_style()
        self._build_ui()
        self._load_settings()
        self._load_ai_profile()
        self._update_backend_state()
        self._update_ai_saved_status()
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f3f0e8")
        style.configure("Hero.TFrame", background="#183a37")
        style.configure("HeroTitle.TLabel", background="#183a37", foreground="#f8f3e8", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("HeroBody.TLabel", background="#183a37", foreground="#d8e6df", font=("Microsoft YaHei UI", 10))
        style.configure("Section.TLabelframe", padding=10)
        style.configure("Section.TLabelframe.Label", foreground="#183a37", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#355c55", font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", foreground="#183a37", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self) -> None:
        wrapper = ttk.Frame(self.root, style="App.TFrame")
        wrapper.pack(fill=tk.BOTH, expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)

        canvas = tk.Canvas(wrapper, highlightthickness=0, bg="#f3f0e8")
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)

        container = ttk.Frame(canvas, padding=12, style="App.TFrame")
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")

        def _on_frame_configure(_: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        def _on_mousewheel(event: tk.Event) -> None:
            delta = -1 * int(event.delta / 120) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        container.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(5, weight=1)

        hero = ttk.Frame(container, padding=(18, 16), style="Hero.TFrame")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="漫画翻译工具", style="HeroTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hero,
            text="支持本地模型和 OpenAI 兼容 AI 接口。可直接选择单张图片或整个文件夹，并自由指定输出目录。",
            style="HeroBody.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        path_frame = ttk.LabelFrame(container, text="路径", style="Section.TLabelframe")
        path_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="源图片 / 文件夹").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(path_frame, textvariable=self.source_path).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(path_frame, text="选文件(可多选)", command=self._choose_source_file).grid(row=0, column=2, padx=4, pady=8)
        ttk.Button(path_frame, text="选文件夹", command=self._choose_source_dir).grid(row=0, column=3, padx=(0, 8), pady=8)

        ttk.Label(path_frame, text="输出目录").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(path_frame, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(path_frame, text="选择", command=self._choose_output_dir).grid(row=1, column=2, columnspan=2, padx=(4, 8), pady=8, sticky="ew")
        ttk.Label(path_frame, textvariable=self.output_summary_text, style="Hint.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6)
        )

        backend_frame = ttk.LabelFrame(container, text="翻译模式", style="Section.TLabelframe")
        backend_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        backend_frame.columnconfigure(0, weight=1)
        backend_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            backend_frame,
            text="本地翻译",
            value="local",
            variable=self.translator_backend,
            command=self._update_backend_state,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Radiobutton(
            backend_frame,
            text="AI 翻译",
            value="openai-compatible",
            variable=self.translator_backend,
            command=self._update_backend_state,
        ).grid(row=0, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(backend_frame, textvariable=self.mode_hint_text, style="Hint.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6)
        )

        ai_frame = ttk.LabelFrame(container, text="AI 参数", style="Section.TLabelframe")
        ai_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        ai_frame.columnconfigure(1, weight=1)
        self.ai_widgets: list[tk.Widget] = []

        self._add_labeled_entry(ai_frame, 0, "Endpoint", self.ai_endpoint, is_ai=True)
        self._add_labeled_entry(ai_frame, 1, "API Key", self.ai_api_key, show="*", is_ai=True)
        self._add_labeled_entry(ai_frame, 2, "Model", self.ai_model, is_ai=True)
        ai_action_frame = ttk.Frame(ai_frame)
        ai_action_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 4))
        ai_action_frame.columnconfigure(2, weight=1)
        save_ai_button = ttk.Button(ai_action_frame, text="保存 AI 参数", command=self._save_ai_profile)
        save_ai_button.grid(row=0, column=0, padx=(0, 8))
        delete_ai_button = ttk.Button(ai_action_frame, text="删除 AI 参数", command=self._delete_ai_profile)
        delete_ai_button.grid(row=0, column=1)
        ttk.Label(ai_action_frame, textvariable=self.ai_saved_text, style="Hint.TLabel").grid(row=0, column=2, sticky="e")
        self.ai_widgets.extend([save_ai_button, delete_ai_button])
        ttk.Label(
            ai_frame,
            text="如果使用 OpenAI 兼容接口，这里填写基地址、密钥和模型名；本地翻译模式下会自动禁用。",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        options_frame = ttk.LabelFrame(container, text="参数", style="Section.TLabelframe")
        options_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)

        ttk.Label(options_frame, text="文字检测").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        detector_combo = ttk.Combobox(
            options_frame,
            textvariable=self.text_detector,
            values=["comic-text-detector", "none"],
            state="readonly",
        )
        detector_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(options_frame, text="去字后端").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        inpaint_combo = ttk.Combobox(
            options_frame,
            textvariable=self.inpaint_backend,
            values=["lama", "opencv"],
            state="readonly",
        )
        inpaint_combo.grid(row=0, column=3, sticky="ew", padx=8, pady=8)

        ttk.Label(options_frame, text="置信度").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(options_frame, textvariable=self.conf_threshold).grid(row=1, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(options_frame, text="气泡边距").grid(row=1, column=2, sticky="w", padx=8, pady=8)
        ttk.Entry(options_frame, textvariable=self.margin).grid(row=1, column=3, sticky="ew", padx=8, pady=8)

        ttk.Label(options_frame, text="字体").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(options_frame, textvariable=self.font_path).grid(row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Button(options_frame, text="选择字体", command=self._choose_font).grid(row=2, column=3, sticky="ew", padx=8, pady=8)

        ttk.Checkbutton(options_frame, text="强制重跑（忽略缓存）", variable=self.force_rebuild).grid(
            row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 8)
        )

        quick_frame = ttk.Frame(container, style="App.TFrame")
        quick_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        for index in range(4):
            quick_frame.columnconfigure(index, weight=1)
        ttk.Button(quick_frame, text="打开输出目录", command=self._open_output_dir).grid(row=0, column=0, padx=(0, 8), sticky="ew")
        ttk.Button(quick_frame, text="打开成品图目录", command=self._open_translated_dir).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(quick_frame, text="清空日志", command=self._clear_log).grid(row=0, column=2, padx=8, sticky="ew")
        ttk.Button(quick_frame, text="填入 AI 示例", command=self._fill_ai_example).grid(row=0, column=3, padx=(8, 0), sticky="ew")

        log_frame = ttk.LabelFrame(container, text="运行日志", style="Section.TLabelframe")
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            bg="#fffdf8",
            fg="#21302d",
            relief="flat",
            font=("Consolas", 10),
            insertbackground="#21302d",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        action_frame = ttk.Frame(container, style="App.TFrame")
        action_frame.grid(row=7, column=0, sticky="ew")
        action_frame.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(action_frame, text="开始翻译", command=self._start_translation, style="Primary.TButton")
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(action_frame, text="停止", command=self._stop_translation, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="w")

        ttk.Label(action_frame, textvariable=self.status_text, style="Status.TLabel").grid(row=0, column=2, sticky="e")

    def _add_labeled_entry(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        show: str | None = None,
        is_ai: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=8)
        entry = ttk.Entry(parent, textvariable=variable, show=show or "")
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=8)
        if is_ai:
            self.ai_widgets.extend([entry])

    def _load_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self.source_path.set(str(payload.get("source_path", "")))
        self.output_path.set(str(payload.get("output_path", "")))
        self.translator_backend.set(str(payload.get("translator_backend", "local")))
        self.force_rebuild.set(bool(payload.get("force_rebuild", True)))
        self.text_detector.set(str(payload.get("text_detector", "comic-text-detector")))
        self.inpaint_backend.set(str(payload.get("inpaint_backend", "lama")))
        self.conf_threshold.set(str(payload.get("conf_threshold", "0.25")))
        self.margin.set(str(payload.get("margin", "8")))
        self.font_path.set(str(payload.get("font_path", detect_default_font_path())))
        selected_files_raw = payload.get("selected_files", [])
        if isinstance(selected_files_raw, list):
            self.selected_files = [Path(p) for p in selected_files_raw if isinstance(p, str)]
        self._refresh_output_summary()

    def _save_settings(self) -> None:
        payload = {
            "source_path": self.source_path.get().strip(),
            "output_path": self.output_path.get().strip(),
            "translator_backend": self.translator_backend.get(),
            "force_rebuild": self.force_rebuild.get(),
            "text_detector": self.text_detector.get(),
            "inpaint_backend": self.inpaint_backend.get(),
            "conf_threshold": self.conf_threshold.get().strip(),
            "margin": self.margin.get().strip(),
            "font_path": self.font_path.get().strip(),
            "selected_files": [str(p) for p in self.selected_files],
        }
        try:
            self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_ai_profile(self) -> None:
        if self.ai_profile_path.exists():
            try:
                payload = json.loads(self.ai_profile_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            self.ai_endpoint.set(str(payload.get("ai_endpoint", "")))
            self.ai_api_key.set(str(payload.get("ai_api_key", "")))
            self.ai_model.set(str(payload.get("ai_model", "")))
            return

        self.ai_endpoint.set(os.environ.get("OPENAI_COMPAT_ENDPOINT", "").strip())
        self.ai_api_key.set(os.environ.get("OPENAI_COMPAT_API_KEY", "").strip())
        self.ai_model.set(os.environ.get("OPENAI_COMPAT_MODEL", "").strip())

    def _save_ai_profile(self) -> None:
        endpoint = self.ai_endpoint.get().strip()
        api_key = self.ai_api_key.get().strip()
        model = self.ai_model.get().strip()
        if not endpoint or not api_key or not model:
            messagebox.showerror(APP_TITLE, "保存 AI 参数前，请先填写 Endpoint、API Key 和 Model。")
            return
        payload = {
            "ai_endpoint": endpoint,
            "ai_api_key": api_key,
            "ai_model": model,
        }
        try:
            self.ai_profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._update_ai_saved_status()
            messagebox.showinfo(APP_TITLE, "AI 参数已保存。")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"保存 AI 参数失败：\n{exc}")

    def _delete_ai_profile(self) -> None:
        confirm = messagebox.askyesno(APP_TITLE, "确定要删除已保存的 AI 参数吗？")
        if not confirm:
            return
        self.ai_endpoint.set("")
        self.ai_api_key.set("")
        self.ai_model.set("")
        try:
            if self.ai_profile_path.exists():
                self.ai_profile_path.unlink()
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"删除 AI 参数失败：\n{exc}")
            return
        self._update_ai_saved_status()
        messagebox.showinfo(APP_TITLE, "AI 参数已删除。")

    def _update_ai_saved_status(self) -> None:
        if self.ai_profile_path.exists():
            self.ai_saved_text.set(f"AI 参数状态：已保存（{self.ai_profile_path.name}）")
        elif self.ai_endpoint.get().strip() and self.ai_api_key.get().strip() and self.ai_model.get().strip():
            self.ai_saved_text.set("AI 参数状态：来自环境变量")
        else:
            self.ai_saved_text.set("AI 参数状态：未保存")

    def _update_backend_state(self) -> None:
        ai_enabled = self.translator_backend.get() == "openai-compatible"
        state = "normal" if ai_enabled else "disabled"
        for widget in self.ai_widgets:
            widget.configure(state=state)
        if ai_enabled:
            self.mode_hint_text.set("当前模式：AI 翻译，适合追求更自然的漫画对白。")
        else:
            self.mode_hint_text.set("当前模式：本地翻译，适合离线处理。")

    def _choose_source_file(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择源图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All Files", "*.*")],
        )
        if paths:
            self.selected_files = [Path(p) for p in paths]
            if len(self.selected_files) == 1:
                self.source_path.set(str(self.selected_files[0]))
            else:
                self.source_path.set(f"已选择 {len(self.selected_files)} 个文件")
            self._auto_fill_output(str(self.selected_files[0]))
            self._refresh_output_summary()

    def _choose_source_dir(self) -> None:
        path = filedialog.askdirectory(title="选择源图片文件夹")
        if path:
            self.selected_files = []
            self.source_path.set(path)
            self._auto_fill_output(path)
            self._refresh_output_summary()

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_path.set(path)
            self._refresh_output_summary()

    def _choose_font(self) -> None:
        path = filedialog.askopenfilename(
            title="选择字体文件",
            filetypes=[("Font Files", "*.ttf *.ttc *.otf"), ("All Files", "*.*")],
        )
        if path:
            self.font_path.set(path)

    def _auto_fill_output(self, source: str) -> None:
        if self.output_path.get().strip():
            return
        source_path = Path(source)
        if source_path.is_dir():
            target = source_path / "outputs_ui"
        else:
            target = source_path.parent / "outputs_ui"
        self.output_path.set(str(target))
        self._refresh_output_summary()

    def _refresh_output_summary(self) -> None:
        output = self.output_path.get().strip()
        if not output:
            self.output_summary_text.set("输出：未选择")
            return
        translated_dir = Path(output) / "translated"
        source_hint = f"已选文件：{len(self.selected_files)} 个" if self.selected_files else "输入：单路径模式"
        self.output_summary_text.set(f"{source_hint}    输出：{output}    成品图目录：{translated_dir}")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _fill_ai_example(self) -> None:
        if not self.ai_endpoint.get().strip():
            self.ai_endpoint.set("https://your-openai-compatible-endpoint/")
        if not self.ai_model.get().strip():
            self.ai_model.set("your-model-name")
        self.translator_backend.set("openai-compatible")
        self._update_backend_state()

    def _open_path(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"无法打开目录：\n{path}\n\n{exc}")

    def _open_output_dir(self) -> None:
        output = self.output_path.get().strip()
        if not output:
            messagebox.showinfo(APP_TITLE, "请先选择输出目录。")
            return
        self._open_path(Path(output))

    def _open_translated_dir(self) -> None:
        output = self.output_path.get().strip()
        if not output:
            messagebox.showinfo(APP_TITLE, "请先选择输出目录。")
            return
        self._open_path(Path(output) / "translated")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                self._append_log(message)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def _set_running_state(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.status_text.set("运行中" if running else "就绪")

    def _validate_inputs(self) -> list[str]:
        errors: list[str] = []
        source = self.source_path.get().strip()
        output = self.output_path.get().strip()
        if not source and not self.selected_files:
            errors.append("请先选择源图片或源文件夹。")
        elif not self.selected_files and not Path(source).exists():
            errors.append("源路径不存在。")
        if self.selected_files and not all(path.exists() for path in self.selected_files):
            errors.append("部分已选文件不存在，请重新选择。")
        if not output:
            errors.append("请先选择输出目录。")

        try:
            float(self.conf_threshold.get().strip())
        except ValueError:
            errors.append("置信度必须是数字。")

        try:
            int(self.margin.get().strip())
        except ValueError:
            errors.append("气泡边距必须是整数。")

        if self.translator_backend.get() == "openai-compatible":
            if not self.ai_endpoint.get().strip():
                errors.append("AI 模式下必须填写 Endpoint。")
            if not self.ai_api_key.get().strip():
                errors.append("AI 模式下必须填写 API Key。")
            if not self.ai_model.get().strip():
                errors.append("AI 模式下必须填写 Model。")
        return errors

    def _start_translation(self) -> None:
        if self.process is not None:
            return
        if self.selected_files and not self.source_path.get().strip().startswith("已选择 "):
            self.selected_files = []
        if not self.python_exe.exists():
            messagebox.showerror(APP_TITLE, f"未找到虚拟环境 Python：\n{self.python_exe}")
            return
        if not self.translate_script.exists():
            messagebox.showerror(APP_TITLE, f"未找到翻译脚本：\n{self.translate_script}")
            return

        errors = self._validate_inputs()
        if errors:
            messagebox.showerror(APP_TITLE, "\n".join(errors))
            return

        self._save_settings()
        self._clear_log()
        self._append_log("开始翻译...\n")
        commands = self._build_commands()

        self.stop_requested = False
        self._set_running_state(True)
        self.status_text.set("运行中：准备启动")
        self.worker_thread = threading.Thread(target=self._run_batch, args=(commands,), daemon=True)
        self.worker_thread.start()

    def _build_commands(self) -> list[list[str]]:
        output = self.output_path.get().strip()
        input_item = self._resolve_input_for_command()
        command = [
            str(self.python_exe),
            str(self.translate_script),
            "--input",
            input_item,
            "--output",
            output,
            "--translator-backend",
            self.translator_backend.get(),
            "--text-detector",
            self.text_detector.get(),
            "--inpaint-backend",
            self.inpaint_backend.get(),
            "--conf",
            self.conf_threshold.get().strip(),
            "--margin",
            self.margin.get().strip(),
            "--font",
            self.font_path.get().strip(),
        ]
        if self.force_rebuild.get():
            command.append("--force")
        if self.translator_backend.get() == "openai-compatible":
            command.extend(
                [
                    "--ai-endpoint",
                    self.ai_endpoint.get().strip(),
                    "--ai-api-key",
                    self.ai_api_key.get().strip(),
                    "--ai-model",
                    self.ai_model.get().strip(),
                ]
            )
        return [command]

    def _resolve_input_for_command(self) -> str:
        if not self.selected_files:
            return self.source_path.get().strip()
        if len(self.selected_files) == 1:
            return str(self.selected_files[0])
        self._cleanup_temp_input_dir()
        temp_dir = Path(tempfile.mkdtemp(prefix="manga_ui_batch_", dir=str(self.base_dir)))
        self.temp_input_dir = temp_dir
        for idx, src in enumerate(sorted(self.selected_files, key=lambda p: p.name), start=1):
            safe_name = src.name
            target = temp_dir / safe_name
            if target.exists():
                target = temp_dir / f"{idx:03d}_{safe_name}"
            try:
                os.link(src, target)
            except OSError:
                shutil.copy2(src, target)
        return str(temp_dir)

    def _cleanup_temp_input_dir(self) -> None:
        if self.temp_input_dir is None:
            return
        try:
            if self.temp_input_dir.exists():
                shutil.rmtree(self.temp_input_dir, ignore_errors=True)
        finally:
            self.temp_input_dir = None

    def _run_batch(self, commands: list[list[str]]) -> None:
        failed = False
        total = len(commands)
        try:
            for idx, command in enumerate(commands, start=1):
                if self.stop_requested:
                    break
                self.root.after(0, lambda i=idx, t=total: self.status_text.set(f"运行中：{i}/{t}"))
                ok = self._run_single_process(command, idx, total)
                if not ok:
                    failed = True
                    break
            if self.stop_requested:
                self.log_queue.put("\n任务已停止。\n")
                self.root.after(0, lambda: messagebox.showinfo(APP_TITLE, "任务已停止。"))
            elif failed:
                self.root.after(0, lambda: messagebox.showerror(APP_TITLE, "翻译过程中出现错误，请查看日志。"))
            else:
                self.log_queue.put("\n翻译完成。\n")
                self.root.after(0, self._refresh_output_summary)
                self.root.after(0, lambda: messagebox.showinfo(APP_TITLE, "翻译完成。"))
        finally:
            self._cleanup_temp_input_dir()
            self.process = None
            self.root.after(0, lambda: self._set_running_state(False))

    def _run_single_process(self, command: list[str], index: int, total: int) -> bool:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self.process.stdout is not None
            self.log_queue.put(f"\n[{index}/{total}] 命令：{' '.join(command)}\n\n")
            for line in self.process.stdout:
                self.log_queue.put(line)
            return_code = self.process.wait()
            if return_code == 0:
                self.log_queue.put(f"\n[{index}/{total}] 完成。\n")
                return True
            else:
                self.log_queue.put(f"\n[{index}/{total}] 翻译失败，退出码：{return_code}\n")
                return False
        except Exception as exc:
            self.log_queue.put(f"\n[{index}/{total}] 运行失败：{exc}\n")
            return False

    def _stop_translation(self) -> None:
        self.stop_requested = True
        if self.process is None:
            return
        self._append_log("\n正在停止...\n")
        try:
            self.process.terminate()
        except OSError:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        if self.process is not None:
            if not messagebox.askyesno(APP_TITLE, "翻译任务还在运行，确定要退出吗？"):
                return
            self._stop_translation()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        root.iconname(APP_TITLE)
    except tk.TclError:
        pass
    app = MangaTranslatorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
