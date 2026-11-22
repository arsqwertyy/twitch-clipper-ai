import os
import subprocess
import sys
from pathlib import Path

# Если gui.py открыли через системный Python («Открыть с помощью -> Python»),
# автоматически запускаемся из виртуального окружения .venv, где стоят все нейросети и библиотеки.
_BASE_DIR = Path(__file__).resolve().parent
_VENV_PYW = _BASE_DIR / ".venv" / "Scripts" / "pythonw.exe"
if _VENV_PYW.exists() and Path(sys.executable).resolve().parent != _VENV_PYW.resolve().parent:
    subprocess.Popen([str(_VENV_PYW), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(_BASE_DIR))
    sys.exit(0)

import asyncio
import queue
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import yaml
from PIL import Image

from chat_monitor import ClipEvent, TwitchChatMonitor, WindowStats, parse_channels_list
from emote_manager import ChannelEmoteManager
from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path, setup_nvidia_dlls
from stream_recorder import StreamRecorder
from subtitle_burner import SubtitleGenerator
from trainer import AutoTrainer
from video_clipper import VideoClipper
from webcam_editor import WebcamEditorDialog
from youtube_uploader import (
    DEFAULT_YOUTUBE_CFG,
    SETUP_HELP_TEXT,
    YouTubeShortsUploader,
    build_shorts_metadata,
    compute_clip_hype_rank,
    evaluate_transcript_hype,
    parse_channel_from_filename,
)

setup_nvidia_dlls()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Современная палитра (Twitch + TikTok Dark Studio)
COLOR_BG = "#0b0b0e"
COLOR_SIDEBAR = "#121217"
COLOR_CARD = "#181820"
COLOR_CARD_HOVER = "#20202b"
COLOR_INPUT = "#23232e"
COLOR_BORDER = "#2a2a38"
COLOR_PURPLE = "#8b44f7"
COLOR_PURPLE_HOVER = "#732ce0"
COLOR_TEAL = "#00b894"
COLOR_TEAL_HOVER = "#00997a"
COLOR_RED = "#e84118"
COLOR_RED_HOVER = "#c23616"
COLOR_GREEN = "#00e676"
COLOR_CYAN = "#00f2fe"
COLOR_YELLOW = "#fbc531"
COLOR_TEXT = "#f5f6fa"
COLOR_MUTED = "#9c9cab"

WEBCAM_LABELS_TO_CODE = {
    "🤖 Авто-поиск ИИ (лицо + рамка)": "auto",
    "↖ Слева сверху": "top_left",
    "⬅ Слева по центру": "middle_left",
    "↙ Слева снизу": "bottom_left",
    "↗ Справа сверху": "top_right",
    "➡ Справа по центру": "middle_right",
    "↘ Справа снизу": "bottom_right",
    "🚫 Без отдельной вебки": "none",
}
WEBCAM_CODE_TO_LABEL = {v: k for k, v in WEBCAM_LABELS_TO_CODE.items()}

BANNER_TIME_LABELS_TO_CODE = {
    "🧠 Умная интрига (перед кульминацией, без обрыва слов)": "smart",
    "🎯 В середине клипа (по паузе в речи)": "middle",
    "▶ В начале клипа (на 3-й сек)": "start",
    "⏹ В конце клипа": "end",
}
BANNER_TIME_CODE_TO_LABEL = {v: k for k, v in BANNER_TIME_LABELS_TO_CODE.items()}

YT_VISIBILITY_LABELS_TO_CODE = {
    "🌍 Открытый доступ (Public)": "public",
    "🔗 Доступ по ссылке (Unlisted)": "unlisted",
    "🔒 Личное (Private)": "private",
}
YT_VISIBILITY_CODE_TO_LABEL = {v: k for k, v in YT_VISIBILITY_LABELS_TO_CODE.items()}


class TwitchClipperApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Twitch Clipper AI — Авто-нарезка в TikTok 9:16 (RTX + Faster-Whisper)")
        self.geometry("1240x820")
        self.minsize(1060, 700)
        self.configure(fg_color=COLOR_BG)

        self.base_dir = Path(__file__).resolve().parent
        self.config_path = self.base_dir / "config.yaml"
        self.cfg = self._load_config()

        initial_raw = self.cfg.get("channels") or self.cfg.get("channel") or []
        self.channels: list[str] = parse_channels_list(initial_raw)

        # Состояние фонового движка
        self.is_running = False
        self.is_training = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.recorders: dict[str, StreamRecorder] = {}
        self.chat_monitors: dict[str, TwitchChatMonitor] = {}
        self.clipper: Optional[VideoClipper] = None
        self.sub_generator: Optional[SubtitleGenerator] = None
        self.clip_queue: Optional[asyncio.Queue[tuple[StreamRecorder, ClipEvent]]] = None

        # История графика и живые метрики по каналам
        self.chart_history_by_channel: dict[str, list[tuple[float, float, bool]]] = {}
        self.selected_chart_channel: str = self.channels[0] if self.channels else ""
        self.channel_live_state: dict[str, str] = {}
        self.channel_chat_state: dict[str, str] = {}
        self.channel_hype_ratio: dict[str, float] = {}

        # Кэш превью-картинок для галереи клипов и виджетов карточек
        self._thumb_cache: dict[str, ctk.CTkImage] = {}
        self._rendered_cards_state: Optional[tuple] = None
        self._card_widgets: dict[str, dict[str, ctk.CTkLabel]] = {}

        # Менеджер эмоутов 7TV / BTTV для всех стримеров
        self.emote_mgr = ChannelEmoteManager(self.base_dir)

        # Авто-загрузчик в YouTube Shorts (собирает пул кандидатов, выбирает №1 лучший, публикует и очищает остальные)
        self.yt_uploader = YouTubeShortsUploader(self.base_dir, self.cfg.get("youtube"))
        self._yt_cond = threading.Condition()
        self._yt_manual_jobs: list[dict] = []
        self._yt_candidates_pool: list[dict] = []
        self._yt_round_deadline_ts: float = 0.0
        self._yt_busy = False
        self._last_global_clip_ts: float = 0.0
        threading.Thread(target=self._yt_worker_loop, daemon=True).start()
        threading.Thread(target=self._bg_sync_emotes, args=(list(self.channels),), daemon=True).start()

        self._setup_clipboard_bindings()
        self._build_layout()

        self.show_page("dashboard")
        self.refresh_streamers_cards()
        self.refresh_clips_gallery()
        self.after(2000, self._periodic_ui_tick)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config_dict(self, data: dict) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # =================== БУФЕР ОБМЕНА (РУССКАЯ/АНГЛ РАСКЛАДКА + ПКМ) ===================

    def _setup_clipboard_bindings(self) -> None:
        def _on_ctrl_key(event: tk.Event) -> Optional[str]:
            widget = event.widget
            if not isinstance(widget, (tk.Entry, tk.Text)):
                return None
            if event.keysym.lower() in ("v", "c", "x", "a"):
                return None
            if event.keycode == 86:
                self._paste_to_tk_widget(widget)
                return "break"
            if event.keycode == 67:
                widget.event_generate("<<Copy>>")
                return "break"
            if event.keycode == 88:
                widget.event_generate("<<Cut>>")
                return "break"
            if event.keycode == 65:
                if isinstance(widget, tk.Text):
                    widget.tag_add("sel", "1.0", "end")
                else:
                    widget.selection_range(0, "end")
                return "break"
            return None

        self.bind_all("<Control-KeyPress>", _on_ctrl_key, add="+")

        self._ctx_menu = tk.Menu(
            self,
            tearoff=0,
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            activebackground=COLOR_PURPLE,
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        )

        def _show_ctx_menu(event: tk.Event) -> None:
            widget = event.widget
            if not isinstance(widget, (tk.Entry, tk.Text)):
                return
            widget.focus_set()
            self._ctx_menu.delete(0, "end")
            self._ctx_menu.add_command(
                label="📋 Вставить", command=lambda: self._paste_to_tk_widget(widget)
            )
            self._ctx_menu.add_command(
                label="📄 Копировать", command=lambda: widget.event_generate("<<Copy>>")
            )
            if isinstance(widget, tk.Entry):
                self._ctx_menu.add_command(
                    label="✂️ Вырезать", command=lambda: widget.event_generate("<<Cut>>")
                )
                self._ctx_menu.add_separator()
                self._ctx_menu.add_command(
                    label="🧹 Очистить", command=lambda: widget.delete(0, "end")
                )
            self._ctx_menu.tk_popup(event.x_root, event.y_root)

        self.bind_all("<Button-3>", _show_ctx_menu, add="+")

    def _paste_to_tk_widget(self, widget: tk.Widget) -> None:
        try:
            text = self.clipboard_get().strip()
        except tk.TclError:
            return
        if not text:
            return
        if isinstance(widget, tk.Entry):
            try:
                if widget.selection_present():
                    widget.delete("sel.first", "sel.last")
            except tk.TclError:
                pass
            widget.insert("insert", text)
            if widget is self.channel_entry._entry:
                self.after(80, self.add_channels_from_entry)
        else:
            widget.event_generate("<<Paste>>")

    def paste_and_add_streamer(self) -> None:
        try:
            text = self.clipboard_get().strip()
        except tk.TclError:
            return
        if not text:
            return
        self.channel_entry.delete(0, "end")
        self.channel_entry.insert(0, text)
        self.add_channels_from_entry()

    # =================== ПОСТРОЕНИЕ СОВРЕМЕННОГО ИНТЕРФЕЙСА ===================

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- ЛЕВЫЙ САЙДБАР ----------------
        self.sidebar = ctk.CTkFrame(self, width=265, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Логотип
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(22, 18))
        ctk.CTkLabel(
            logo_frame,
            text="⚡ TWITCH CLIPPER",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            logo_frame,
            text="TikTok 9:16 • ИИ Вебка • Whisper RTX",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_CYAN,
        ).pack(anchor="w", pady=(2, 0))

        # Навигационные кнопки
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        nav_items = [
            ("dashboard", "📡  Стримеры и Эфир"),
            ("clips", "🎬  Галерея клипов"),
            ("settings", "⚙️  Настройки и ИИ"),
        ]
        for key, label in nav_items:
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                anchor="w",
                height=44,
                corner_radius=10,
                font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                fg_color="transparent",
                text_color=COLOR_TEXT,
                hover_color=COLOR_CARD_HOVER,
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill="x", padx=14, pady=4)
            self.nav_buttons[key] = btn

        # Разделитель
        ctk.CTkFrame(self.sidebar, height=1, fg_color=COLOR_BORDER).pack(fill="x", padx=18, pady=14)

        # Блок быстрых сводок в сайдбаре
        stats_card = ctk.CTkFrame(self.sidebar, corner_radius=12, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        stats_card.pack(fill="x", padx=14, pady=(0, 14))

        self.sb_streamers_lbl = ctk.CTkLabel(
            stats_card,
            text="Стримеров: 0 (Онлайн: 0)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        )
        self.sb_streamers_lbl.pack(anchor="w", padx=12, pady=(10, 2))

        self.sb_trained_lbl = ctk.CTkLabel(
            stats_card,
            text="Обучено профилей: 0",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_MUTED,
        )
        self.sb_trained_lbl.pack(anchor="w", padx=12, pady=(0, 4))

        self.sb_gpu_lbl = ctk.CTkLabel(
            stats_card,
            text="⚡ RTX 4050 CUDA + NVENC",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_CYAN,
        )
        self.sb_gpu_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Кнопки главных действий внизу сайдбара
        bottom_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_box.pack(side="bottom", fill="x", padx=14, pady=16)

        self.train_btn = ctk.CTkButton(
            bottom_box,
            text="🧠 Обучить и Тест (до стрима)",
            height=44,
            corner_radius=10,
            fg_color=COLOR_TEAL,
            hover_color=COLOR_TEAL_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.start_prestream_training,
        )
        self.train_btn.pack(fill="x", pady=(0, 8))

        self.start_btn = ctk.CTkButton(
            bottom_box,
            text="▶ ЗАПУСТИТЬ МОНИТОРИНГ",
            height=48,
            corner_radius=10,
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.toggle_monitoring,
        )
        self.start_btn.pack(fill="x", pady=(0, 8))

        self.manual_clip_btn = ctk.CTkButton(
            bottom_box,
            text="✂️ Сделать клип сейчас (30с)",
            height=38,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            font=ctk.CTkFont(size=13),
            state="disabled",
            command=self.trigger_manual_clip,
        )
        self.manual_clip_btn.pack(fill="x", pady=(0, 6))

        ctk.CTkButton(
            bottom_box,
            text="📂 Открыть папку с клипами",
            height=36,
            corner_radius=10,
            fg_color="transparent",
            border_width=1,
            border_color=COLOR_BORDER,
            hover_color=COLOR_CARD_HOVER,
            font=ctk.CTkFont(size=12),
            command=self.open_clips_folder,
        ).pack(fill="x")

        # ---------------- ПРАВАЯ ОСНОВНАЯ ОБЛАСТЬ ----------------
        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_BG)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=18, pady=16)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {
            "dashboard": ctk.CTkFrame(self.main_area, fg_color="transparent"),
            "clips": ctk.CTkFrame(self.main_area, fg_color="transparent"),
            "settings": ctk.CTkFrame(self.main_area, fg_color="transparent"),
        }

        self._build_dashboard_page(self.pages["dashboard"])
        self._build_clips_page(self.pages["clips"])
        self._build_settings_page(self.pages["settings"])

    def show_page(self, page_key: str) -> None:
        for k, frame in self.pages.items():
            if k == page_key:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        for k, btn in self.nav_buttons.items():
            if k == page_key:
                btn.configure(fg_color=COLOR_PURPLE, text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT)

        if page_key == "clips":
            self.refresh_clips_gallery()

    # =================== СТРАНИЦА 1: СТРИМЕРЫ И ЭФИР ===================

    def _build_dashboard_page(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_rowconfigure(2, weight=0)
        parent.grid_rowconfigure(3, weight=1)

        # 1. Панель быстрого добавления стримеров
        add_card = ctk.CTkFrame(parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        add_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        inner_add = ctk.CTkFrame(add_card, fg_color="transparent")
        inner_add.pack(fill="x", padx=16, pady=12)

        self.channel_entry = ctk.CTkEntry(
            inner_add,
            placeholder_text="Введите ник или ссылку на Twitch (можно несколько через запятую)...",
            height=40,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(size=14),
        )
        self.channel_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.channel_entry._entry.bind("<Return>", lambda _e: self.add_channels_from_entry())
        self.channel_entry._entry.bind("<<Paste>>", lambda _e: self.after(80, self.add_channels_from_entry))

        ctk.CTkButton(
            inner_add,
            text="📋 Вставить из буфера",
            height=40,
             width=165,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.paste_and_add_streamer,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            inner_add,
            text="+ Добавить",
            height=40,
            width=115,
            corner_radius=10,
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.add_channels_from_entry,
        ).pack(side="left", padx=(0, 14))

        ctk.CTkLabel(inner_add, text="Качество:", font=ctk.CTkFont(size=13), text_color=COLOR_MUTED).pack(
            side="left", padx=(0, 6)
        )
        self.quality_var = ctk.StringVar(value=str(self.cfg.get("stream", {}).get("quality", "best")))
        self.quality_menu = ctk.CTkOptionMenu(
            inner_add,
            variable=self.quality_var,
            values=["best", "1080p60", "720p60", "720p", "480p"],
            width=110,
            height=40,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
            command=lambda _v: self.save_settings_from_ui(silent=True),
        )
        self.quality_menu.pack(side="left")

        # 2. Секция карточек стримеров
        streamers_card = ctk.CTkFrame(parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        streamers_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        s_hdr = ctk.CTkFrame(streamers_card, fg_color="transparent")
        s_hdr.pack(fill="x", padx=16, pady=(10, 4))
        ctk.CTkLabel(
            s_hdr,
            text="🎯 Отслеживаемые стримеры и их ИИ-профили",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT,
        ).pack(side="left")

        self.make_test_clip_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            s_hdr,
            text="При обучении нарезать пробный TikTok-клип (VOD)",
            variable=self.make_test_clip_var,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=12),
        ).pack(side="right")

        self.streamers_scroll = ctk.CTkScrollableFrame(
            streamers_card, height=165, corner_radius=10, fg_color="#13131a"
        )
        self.streamers_scroll.pack(fill="x", padx=12, pady=(4, 12))

        # 3. Живой график хайпа выбранного стримера
        chart_card = ctk.CTkFrame(parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        chart_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        c_hdr = ctk.CTkFrame(chart_card, fg_color="transparent")
        c_hdr.pack(fill="x", padx=16, pady=(10, 4))
        self.chart_title_lbl = ctk.CTkLabel(
            c_hdr,
            text="📈 Живой график скорости чата (столбик = 10 сек, красный = ВЗРЫВ ЧАТА)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        )
        self.chart_title_lbl.pack(side="left")

        self.top_msg_lbl = ctk.CTkLabel(
            c_hdr,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_YELLOW,
        )
        self.top_msg_lbl.pack(side="right")

        self.chart_canvas = tk.Canvas(chart_card, bg="#121218", height=115, highlightthickness=0)
        self.chart_canvas.pack(fill="x", padx=14, pady=(2, 12))
        self.chart_canvas.bind("<Configure>", lambda _e: self._draw_chart())

        # 4. Журнал событий внизу
        log_card = ctk.CTkFrame(parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        log_card.grid(row=3, column=0, sticky="nsew")

        ctk.CTkLabel(
            log_card,
            text="📋 Журнал работы в реальном времени",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_MUTED,
        ).pack(anchor="w", padx=16, pady=(10, 4))

        self.log_text = tk.Text(
            log_card,
            bg="#121218",
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
            state="disabled",
            padx=10,
            pady=8,
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log_text.tag_config("info", foreground=COLOR_TEXT)
        self.log_text.tag_config("muted", foreground=COLOR_MUTED)
        self.log_text.tag_config("success", foreground=COLOR_GREEN)
        self.log_text.tag_config("warn", foreground=COLOR_YELLOW)
        self.log_text.tag_config("spike", foreground="#ff5252")
        self.log_text.tag_config("cyan", foreground=COLOR_CYAN)

        self.append_log("Готово к работе! Добавьте стримеров сверху или нажмите «🧠 Обучить и Тест (до стрима)».", "cyan")

    # =================== СТРАНИЦА 2: ВИЗУАЛЬНАЯ ГАЛЕРЕЯ КЛИПОВ ===================

    def _build_clips_page(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        t_in = ctk.CTkFrame(top_bar, fg_color="transparent")
        t_in.pack(fill="x", padx=16, pady=12)

        self.gallery_title_lbl = ctk.CTkLabel(
            t_in,
            text="🎬 Готовые вертикальные клипы TikTok 9:16",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLOR_TEXT,
        )
        self.gallery_title_lbl.pack(side="left")

        ctk.CTkButton(
            t_in,
            text="📂 Открыть папку clips",
            height=36,
            corner_radius=10,
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.open_clips_folder,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            t_in,
            text="🔄 Обновить галерею",
            height=36,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            font=ctk.CTkFont(size=13),
            command=self.refresh_clips_gallery,
        ).pack(side="right")

        self.clips_scroll = ctk.CTkScrollableFrame(
            parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER
        )
        self.clips_scroll.grid(row=1, column=0, sticky="nsew")

    # =================== СТРАНИЦА 3: НАСТРОЙКИ С ПОЛЗУНКАМИ ===================

    def _build_settings_page(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            parent, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        scroll.grid_columnconfigure(1, weight=1)

        det = self.cfg.get("detector", {})
        vid = self.cfg.get("video", {})
        sub = self.cfg.get("subtitles", {})

        # Левая карточка: Детектор чата и тайминги клипа
        left_box = ctk.CTkFrame(scroll, corner_radius=12, fg_color="#13131a", border_width=1, border_color=COLOR_BORDER)
        left_box.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)

        ctk.CTkLabel(
            left_box,
            text="🔥 Детектор хайпа и длина клипов",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_CYAN,
        ).pack(anchor="w", padx=16, pady=(14, 10))

        self.z_val = float(det.get("z_score_threshold", 3.8))
        self.min_viral_val = float(det.get("min_viral_score", 62.0))
        self.pre_val = float(det.get("pre_roll_seconds", 16))
        self.post_val = float(det.get("post_roll_seconds", 8))
        self.max_clip_val = float(det.get("max_clip_seconds", 60))
        self.cooldown_val = float(det.get("cooldown_seconds", 480))
        self.global_cooldown_val = float(det.get("global_cooldown_seconds", 360))
        self.min_msgs_val = int(det.get("min_messages_in_window", 14))
        self.min_users_val = int(det.get("min_unique_chatters", 11))

        self.slider_z = self._add_slider_control(
            left_box,
            title="Порог взрыва чата (Z-score)",
            subtitle="3.6–4.2 = только самые яркие и разрывные моменты без проходняка",
            from_=2.0,
            to=5.5,
            initial=max(2.0, min(5.5, self.z_val)),
            fmt="{:.2f}",
        )
        self.slider_min_viral = self._add_slider_control(
            left_box,
            title="Мин. рейтинг интересности клипа (фильтр отбора)",
            subtitle="60–75 = резать только действительно сильные моменты (меньше мусора)",
            from_=30.0,
            to=90.0,
            initial=max(30.0, min(90.0, self.min_viral_val)),
            fmt="{:.0f}/100",
        )
        self.slider_pre = self._add_slider_control(
            left_box,
            title="Захват видео ДО реакции чата (Pre-roll)",
            subtitle="Чат реагирует с задержкой — берем секунды перед взрывом чата",
            from_=10.0,
            to=45.0,
            initial=self.pre_val,
            fmt="{:.0f} сек",
        )
        self.slider_post = self._add_slider_control(
            left_box,
            title="Захват видео ПОСЛЕ пика (Post-roll)",
            subtitle="Сколько секунд дописать после спада волны в чате",
            from_=5.0,
            to=30.0,
            initial=self.post_val,
            fmt="{:.0f} сек",
        )
        self.slider_max_clip = self._add_slider_control(
            left_box,
            title="Максимальная длительность клипа",
            subtitle="Если хайп в чате продолжается, клип продлевается до этого лимита",
            from_=20.0,
            to=90.0,
            initial=self.max_clip_val,
            fmt="{:.0f} сек",
        )
        self.slider_cooldown = self._add_slider_control(
            left_box,
            title="Пауза между клипами на одном канале",
            subtitle="480 сек (8 мин) = максимум 1 лучший момент со стрима за 8 минут",
            from_=60.0,
            to=900.0,
            initial=max(60.0, min(900.0, self.cooldown_val)),
            fmt="{:.0f} сек",
        )
        self.slider_global_cooldown = self._add_slider_control(
            left_box,
            title="Общая пауза между клипами (по всем каналам)",
            subtitle="360 сек (6 мин) — чтобы 9 каналов не нарезали по 5 клипов подряд (мега-хайп ≥88 пробивает)",
            from_=0.0,
            to=900.0,
            initial=max(0.0, min(900.0, self.global_cooldown_val)),
            fmt="{:.0f} сек",
        )

        # Правая карточка: Кадр TikTok 9:16, Вебкамера и Субтитры Whisper
        right_box = ctk.CTkFrame(scroll, corner_radius=12, fg_color="#13131a", border_width=1, border_color=COLOR_BORDER)
        right_box.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=10)

        ctk.CTkLabel(
            right_box,
            text="🎬 Формат TikTok 9:16, ИИ-Вебка и Субтитры",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_CYAN,
        ).pack(anchor="w", padx=16, pady=(14, 10))

        # Выбор режима вебкамеры
        cam_row = ctk.CTkFrame(right_box, fg_color="transparent")
        cam_row.pack(fill="x", padx=16, pady=(4, 10))
        ctk.CTkLabel(
            cam_row, text="Расположение вебкамеры стримера:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            cam_row,
            text="В режиме Авто ИИ сам находит лицо и рамку вебки и ставит ее в верхний блок",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_MUTED,
        ).pack(anchor="w", pady=(0, 6))

        cur_cam_code = str(vid.get("webcam_position", "auto"))
        self.webcam_label_var = ctk.StringVar(
            value=WEBCAM_CODE_TO_LABEL.get(cur_cam_code, "🤖 Авто-поиск ИИ (лицо + рамка)")
        )
        ctk.CTkOptionMenu(
            cam_row,
            variable=self.webcam_label_var,
            values=list(WEBCAM_LABELS_TO_CODE.keys()),
            height=36,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
        ).pack(fill="x")

        self.slider_zoom = self._add_slider_control(
            right_box,
            title="Приближение основного экрана (Zoom)",
            subtitle="1.0 = вся ширина стрима целиком, 1.35–1.45 = крупнее центр игры/видео",
            from_=1.0,
            to=1.8,
            initial=float(vid.get("center_zoom", 1.35)),
            fmt="{:.2f}x",
        )

        # Модель Whisper и язык
        w_row = ctk.CTkFrame(right_box, fg_color="transparent")
        w_row.pack(fill="x", padx=16, pady=(4, 8))
        ctk.CTkLabel(w_row, text="Модель нейросети Faster-Whisper:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.sub_model_var = ctk.StringVar(value=str(sub.get("model_size", "large-v3-turbo")))
        ctk.CTkOptionMenu(
            w_row,
            variable=self.sub_model_var,
            values=["large-v3-turbo", "large-v3", "medium", "small"],
            height=36,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
        ).pack(fill="x")

        self.slider_fontsize = self._add_slider_control(
            right_box,
            title="Размер шрифта субтитров TikTok",
            subtitle="Крупный жирный текст с ярко-желтой подсветкой текущего слова",
            from_=48.0,
            to=96.0,
            initial=float(sub.get("font_size", 72)),
            fmt="{:.0f} px",
        )

        self.slider_words = self._add_slider_control(
            right_box,
            title="Слов одновременно на экране",
            subtitle="В стиле TikTok / Shorts лучше всего смотрится 2–4 слова",
            from_=1.0,
            to=6.0,
            initial=float(sub.get("words_per_phrase", 3)),
            fmt="{:.0f} сл.",
        )

        # Переключатели
        sw_box = ctk.CTkFrame(right_box, fg_color="transparent")
        sw_box.pack(fill="x", padx=16, pady=(6, 12))

        self.var_sub_enabled = ctk.BooleanVar(value=bool(sub.get("enabled", True)))
        ctk.CTkSwitch(
            sw_box,
            text="Вшивать динамические субтитры Faster-Whisper",
            variable=self.var_sub_enabled,
            progress_color=COLOR_PURPLE,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=4)

        self.var_smart_hook = ctk.BooleanVar(value=bool(sub.get("smart_hook", True)))
        ctk.CTkSwitch(
            sw_box,
            text="🎯 Умное начало клипа по фразе (Smart Hook без тишины и обрывков слов)",
            variable=self.var_smart_hook,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=4)

        self.var_audio_peaks = ctk.BooleanVar(value=bool(det.get("use_audio_peaks", True)))
        ctk.CTkSwitch(
            sw_box,
            text="🔊 Детектор пиков звука (учитывать крик и смех стримера вместе с чатом)",
            variable=self.var_audio_peaks,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=4)

        self.var_climax_zoom = ctk.BooleanVar(value=bool(vid.get("climax_zoom", True)))
        ctk.CTkSwitch(
            sw_box,
            text="🔍 Эффект приближения (Zoom-In +18%) на вебку в момент кульминации",
            variable=self.var_climax_zoom,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=4)

        self.var_use_nvenc = ctk.BooleanVar(value=bool(vid.get("use_nvenc", True)))
        ctk.CTkSwitch(
            sw_box,
            text="Ускорение рендера на видеокарте NVIDIA (h264_nvenc)",
            variable=self.var_use_nvenc,
            progress_color=COLOR_PURPLE,
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=4)

        # Блок баннерной рекламы (на всю ширину под двумя карточками)
        ban_cfg = vid.get("banner", {}) or {}
        ban_box = ctk.CTkFrame(scroll, corner_radius=12, fg_color="#13131a", border_width=1, border_color=COLOR_BORDER)
        ban_box.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 10))
        ban_box.grid_columnconfigure(0, weight=1)
        ban_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            ban_box,
            text="📢 Баннерная реклама поверх каждого клипа (видео с хромакеем / PNG)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_CYAN,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 6))

        ban_sw_row = ctk.CTkFrame(ban_box, fg_color="transparent")
        ban_sw_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))

        self.var_banner_enabled = ctk.BooleanVar(value=bool(ban_cfg.get("enabled", False)))
        ctk.CTkSwitch(
            ban_sw_row,
            text="Вставлять баннерную рекламу в каждый клип",
            variable=self.var_banner_enabled,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 18))

        self.var_banner_pause = ctk.BooleanVar(value=bool(ban_cfg.get("pause_video", True)))
        ctk.CTkSwitch(
            ban_sw_row,
            text="Стоп-кадр + только звук баннера",
            variable=self.var_banner_pause,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 14))

        self.var_banner_chromakey = ctk.BooleanVar(value=bool(ban_cfg.get("chromakey", True)))
        ctk.CTkSwitch(
            ban_sw_row,
            text="Авто-хромакей",
            variable=self.var_banner_chromakey,
            progress_color=COLOR_PURPLE,
            font=ctk.CTkFont(size=12),
        ).pack(side="left")

        ban_file_row = ctk.CTkFrame(ban_box, fg_color="transparent")
        ban_file_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))

        self.banner_path_entry = ctk.CTkEntry(
            ban_file_row,
            height=36,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            placeholder_text="Путь к видео (.mp4/.mov/.webm) или картинке (.png) баннера...",
            font=ctk.CTkFont(size=12),
        )
        self.banner_path_entry.insert(0, str(ban_cfg.get("path", "")))
        self.banner_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            ban_file_row,
            text="📂 Выбрать файл баннера",
            height=36,
            width=185,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.pick_banner_file,
        ).pack(side="left")

        ban_left = ctk.CTkFrame(ban_box, fg_color="transparent")
        ban_left.grid(row=3, column=0, sticky="nsew", pady=(0, 8))

        ban_time_wrap = ctk.CTkFrame(ban_left, fg_color="transparent")
        ban_time_wrap.pack(fill="x", padx=16, pady=(4, 8))
        ctk.CTkLabel(ban_time_wrap, text="Момент появления баннера:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", pady=(0, 4)
        )
        cur_ban_time = str(ban_cfg.get("position_time", "smart"))
        self.banner_time_var = ctk.StringVar(
            value=BANNER_TIME_CODE_TO_LABEL.get(
                cur_ban_time, "🧠 Умная интрига (перед кульминацией, без обрыва слов)"
            )
        )
        ctk.CTkOptionMenu(
            ban_time_wrap,
            variable=self.banner_time_var,
            values=list(BANNER_TIME_LABELS_TO_CODE.keys()),
            height=34,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
        ).pack(fill="x")

        self.slider_banner_vol = self._add_slider_control(
            ban_left,
            title="Громкость звука баннера",
            subtitle="0% = без звука, 85% = сбалансированно поверх стрима",
            from_=0.0,
            to=150.0,
            initial=round(float(ban_cfg.get("volume", 0.85)) * 100.0, 0),
            fmt="{:.0f}%",
        )

        ban_right = ctk.CTkFrame(ban_box, fg_color="transparent")
        ban_right.grid(row=3, column=1, sticky="nsew", pady=(0, 8))

        self.slider_banner_scale = self._add_slider_control(
            ban_right,
            title="Размер баннера (ширина от кадра)",
            subtitle="85–90% = крупно по центру как на примере",
            from_=40.0,
            to=100.0,
            initial=round(float(ban_cfg.get("width_ratio", 0.88)) * 100.0, 0),
            fmt="{:.0f}%",
        )
        self.slider_banner_y = self._add_slider_control(
            ban_right,
            title="Позиция баннера по вертикали",
            subtitle="48% = ровно по центру под вебкамерой и над субтитрами",
            from_=10.0,
            to=85.0,
            initial=round(float(ban_cfg.get("pos_y", 0.48)) * 100.0, 0),
            fmt="{:.0f}%",
        )

        # Блок авто-заливки в YouTube Shorts (Официальный YouTube Data API v3 + OAuth 2.0)
        yt_cfg = dict(DEFAULT_YOUTUBE_CFG)
        yt_cfg.update(self.cfg.get("youtube") or {})

        yt_box = ctk.CTkFrame(scroll, corner_radius=12, fg_color="#13131a", border_width=1, border_color=COLOR_BORDER)
        yt_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 10))
        yt_box.grid_columnconfigure(0, weight=1)
        yt_box.grid_columnconfigure(1, weight=1)

        yt_hdr = ctk.CTkFrame(yt_box, fg_color="transparent")
        yt_hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            yt_hdr,
            text="📺 Авто-заливка в YouTube Shorts (Официальный YouTube Data API v3)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#ff5252",
        ).pack(side="left")

        self.yt_status_lbl = ctk.CTkLabel(
            yt_hdr,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_MUTED,
        )
        self.yt_status_lbl.pack(side="right")

        # Строка выбора client_secrets.json + кнопка инструкции
        yt_sec_row = ctk.CTkFrame(yt_box, fg_color="transparent")
        yt_sec_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))

        self.yt_secrets_entry = ctk.CTkEntry(
            yt_sec_row,
            height=34,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            placeholder_text="Путь к файлу client_secrets.json из Google Cloud Console...",
            font=ctk.CTkFont(size=12),
        )
        self.yt_secrets_entry.insert(0, str(yt_cfg.get("client_secrets_path", "client_secrets.json")))
        self.yt_secrets_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            yt_sec_row,
            text="📂 Выбрать client_secrets.json",
            height=34,
            width=210,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.pick_yt_secrets_file,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            yt_sec_row,
            text="❓ Как получить ключ",
            height=34,
            width=165,
            corner_radius=10,
            fg_color="#232336",
            hover_color=COLOR_PURPLE,
            text_color=COLOR_CYAN,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.show_youtube_setup_help,
        ).pack(side="left")

        yt_auth_row = ctk.CTkFrame(yt_box, fg_color="transparent")
        yt_auth_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))

        self.yt_login_btn = ctk.CTkButton(
            yt_auth_row,
            text="🔑 Авторизовать YouTube (OAuth)",
            height=36,
            width=250,
            corner_radius=10,
            fg_color="#c4302b",
            hover_color="#e53935",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.start_youtube_login,
        )
        self.yt_login_btn.pack(side="left", padx=(0, 10))

        self.yt_check_btn = ctk.CTkButton(
            yt_auth_row,
            text="🔍 Проверить канал",
            height=36,
            width=155,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.check_youtube_session,
        )
        self.yt_check_btn.pack(side="left", padx=(0, 16))

        self.var_yt_enabled = ctk.BooleanVar(
            value=bool(yt_cfg.get("enabled", True)) and bool(yt_cfg.get("auto_upload", True))
        )
        ctk.CTkSwitch(
            yt_auth_row,
            text="Авто-заливка в фоне",
            variable=self.var_yt_enabled,
            progress_color="#c4302b",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 14))

        self.var_yt_delete_after = ctk.BooleanVar(value=bool(yt_cfg.get("delete_after_upload", True)))
        ctk.CTkSwitch(
            yt_auth_row,
            text="🗑 Удалять видео с ПК после публикации",
            variable=self.var_yt_delete_after,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        yt_left = ctk.CTkFrame(yt_box, fg_color="transparent")
        yt_left.grid(row=3, column=0, sticky="nsew", pady=(0, 10))

        yt_vis_wrap = ctk.CTkFrame(yt_left, fg_color="transparent")
        yt_vis_wrap.pack(fill="x", padx=16, pady=(4, 8))
        ctk.CTkLabel(yt_vis_wrap, text="Доступ при публикации:", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.yt_visibility_var = ctk.StringVar(
            value=YT_VISIBILITY_CODE_TO_LABEL.get(
                str(yt_cfg.get("visibility", "public")),
                "🌍 Открытый доступ (Public)",
            )
        )
        ctk.CTkOptionMenu(
            yt_vis_wrap,
            variable=self.yt_visibility_var,
            values=list(YT_VISIBILITY_LABELS_TO_CODE.keys()),
            height=34,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
        ).pack(fill="x")

        yt_title_tags_wrap = ctk.CTkFrame(yt_left, fg_color="transparent")
        yt_title_tags_wrap.pack(fill="x", padx=16, pady=(4, 6))
        ctk.CTkLabel(
            yt_title_tags_wrap, text="Хэштеги в названии каждого видео:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(0, 4))
        self.yt_title_tags_entry = ctk.CTkEntry(
            yt_title_tags_wrap,
            height=34,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            placeholder_text="#dinodrop #динодроп",
            font=ctk.CTkFont(size=12),
        )
        self.yt_title_tags_entry.insert(
            0, str(yt_cfg.get("title_hashtags", DEFAULT_YOUTUBE_CFG["title_hashtags"]))
        )
        self.yt_title_tags_entry.pack(fill="x")

        self.var_yt_auto_title = ctk.BooleanVar(value=bool(yt_cfg.get("auto_title", True)))
        ctk.CTkSwitch(
            yt_title_tags_wrap,
            text="Умное название под сюжет видео перед хэштегами (вкл = название + хэштеги)",
            variable=self.var_yt_auto_title,
            progress_color=COLOR_PURPLE,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(6, 2))

        self.var_yt_auto_comment = ctk.BooleanVar(value=bool(yt_cfg.get("auto_comment", True)))
        ctk.CTkSwitch(
            yt_title_tags_wrap,
            text="💬 Авто-комментарий от автора под каждым Short (для буста алгоритмов)",
            variable=self.var_yt_auto_comment,
            progress_color=COLOR_TEAL,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(2, 2))

        yt_tags_wrap = ctk.CTkFrame(yt_left, fg_color="transparent")
        yt_tags_wrap.pack(fill="x", padx=16, pady=(2, 4))
        ctk.CTkLabel(
            yt_tags_wrap, text="Описание (необязательно, можно оставить пустым):", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(0, 4))
        self.yt_hashtags_entry = ctk.CTkEntry(
            yt_tags_wrap,
            height=32,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            placeholder_text="Пусто = как на примере (Добавьте описание)",
            font=ctk.CTkFont(size=12),
        )
        self.yt_hashtags_entry.insert(0, str(yt_cfg.get("extra_hashtags", DEFAULT_YOUTUBE_CFG["extra_hashtags"])))
        self.yt_hashtags_entry.pack(fill="x")

        yt_right = ctk.CTkFrame(yt_box, fg_color="transparent")
        yt_right.grid(row=3, column=1, sticky="nsew", pady=(0, 10))

        self.slider_yt_viral = self._add_slider_control(
            yt_right,
            title="Мин. рейтинг виральности для авто-заливки",
            subtitle="0 = заливать все найденные клипы, >0 = только самые сильные",
            from_=0.0,
            to=95.0,
            initial=float(yt_cfg.get("min_viral_score", 0)),
            fmt="{:.0f}/100",
        )
        self.slider_yt_interval = self._add_slider_control(
            yt_right,
            title="Окно отбора лучшего клипа (раз в N мин)",
            subtitle="60 мин = за час нарезаются кандидаты, публикуется 1 ЛУЧШИЙ по хайпу, остальные удаляются",
            from_=0.0,
            to=360.0,
            initial=max(0.0, min(360.0, float(yt_cfg.get("min_interval_minutes", 60)))),
            fmt="{:.0f} мин",
        )
        self.slider_yt_maxday = self._add_slider_control(
            yt_right,
            title="Лимит авто-заливок в день (YouTube Shorts)",
            subtitle="Максимум публикаций в сутки (при 1 видео в час = до 24 шт/день)",
            from_=1.0,
            to=50.0,
            initial=max(1.0, min(50.0, float(yt_cfg.get("max_per_day", 24)))),
            fmt="{:.0f} шт/день",
        )

        self._refresh_yt_status_label()

        # Нижний блок: Ключевые слова и кнопки сохранения
        bottom_cfg = ctk.CTkFrame(scroll, corner_radius=12, fg_color="#13131a", border_width=1, border_color=COLOR_BORDER)
        bottom_cfg.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 12))

        ctk.CTkLabel(
            bottom_cfg,
            text="💬 Базовые триггерные слова чата (через запятую):",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(12, 4))

        kws = ", ".join(str(x) for x in det.get("hype_keywords", []))
        self.keywords_entry = ctk.CTkEntry(
            bottom_cfg,
            height=38,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            border_color=COLOR_BORDER,
            font=ctk.CTkFont(size=13),
        )
        self.keywords_entry.insert(0, kws)
        self.keywords_entry.pack(fill="x", padx=16, pady=(0, 12))

        b_row = ctk.CTkFrame(bottom_cfg, fg_color="transparent")
        b_row.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(
            b_row,
            text="💾 Сохранить все настройки",
            height=42,
            width=220,
            corner_radius=10,
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.save_settings_from_ui,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            b_row,
            text="🧹 Сбросить обученные профили каналов",
            height=42,
            width=260,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#38252a",
            text_color="#ff7675",
            font=ctk.CTkFont(size=13),
            command=self.reset_learned_profiles,
        ).pack(side="left")

    def _add_slider_control(
        self,
        parent: ctk.CTkFrame,
        title: str,
        subtitle: str,
        from_: float,
        to: float,
        initial: float,
        fmt: str,
    ) -> ctk.CTkSlider:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(fill="x", padx=16, pady=(4, 8))

        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT).pack(side="left")
        val_lbl = ctk.CTkLabel(
            top, text=fmt.format(initial), font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_CYAN
        )
        val_lbl.pack(side="right")

        ctk.CTkLabel(box, text=subtitle, font=ctk.CTkFont(size=11), text_color=COLOR_MUTED).pack(
            anchor="w", pady=(0, 4)
        )

        slider = ctk.CTkSlider(
            box,
            from_=from_,
            to=to,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
            progress_color=COLOR_PURPLE,
            command=lambda v: val_lbl.configure(text=fmt.format(float(v))),
        )
        slider.set(initial)
        slider._val_lbl = val_lbl
        slider._fmt = fmt
        slider.pack(fill="x")
        return slider

    def pick_banner_file(self) -> None:
        initial_dir = self.base_dir / "banners"
        if not initial_dir.exists():
            initial_dir = self.base_dir
        chosen = filedialog.askopenfilename(
            title="Выберите видео или картинку рекламного баннера",
            initialdir=str(initial_dir),
            filetypes=[
                ("Медиафайлы баннера", "*.mp4 *.mov *.webm *.mkv *.gif *.png *.jpg *.jpeg *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not chosen:
            return
        p = Path(chosen).resolve()
        try:
            rel = p.relative_to(self.base_dir.resolve())
            display_path = str(rel).replace("\\", "/")
        except ValueError:
            display_path = str(p)
        self.banner_path_entry.delete(0, "end")
        self.banner_path_entry.insert(0, display_path)
        self.var_banner_enabled.set(True)
        self.save_settings_from_ui(silent=False)

    def _sync_ui_from_report(self, rep, all_keywords: list[str]) -> None:
        """Обновляет ползунки и поле ключевых слов на вкладке Настройки после завершения обучения."""
        self.slider_z.set(rep.z_score_threshold)
        if hasattr(self.slider_z, "_val_lbl"):
            self.slider_z._val_lbl.configure(text=self.slider_z._fmt.format(rep.z_score_threshold))
        self.keywords_entry.delete(0, "end")
        self.keywords_entry.insert(0, ", ".join(str(x) for x in all_keywords))

    # =================== КАРТОЧКИ СТРИМЕРОВ ===================

    def add_channels_from_entry(self) -> None:
        raw = (self.channel_entry.get() or "").strip()
        if not raw:
            return
        new_chs = parse_channels_list(raw)
        added = []
        for ch in new_chs:
            if ch not in self.channels:
                self.channels.append(ch)
                added.append(ch)
        self.channel_entry.delete(0, "end")
        if not self.selected_chart_channel and self.channels:
            self.selected_chart_channel = self.channels[0]
        self.save_settings_from_ui(silent=True)
        self.refresh_streamers_cards()
        if added:
            self.append_log(f"✅ Добавлены стримеры: {', '.join(added)}", "cyan")
            threading.Thread(target=self._bg_sync_emotes, args=(list(added),), daemon=True).start()

    def _bg_sync_emotes(self, channels: list[str]) -> None:
        if not channels:
            return
        try:
            self.emote_mgr.sync_channels(
                channels,
                force=False,
                log_cb=lambda m: self.append_log(m, "cyan"),
            )
            for mon in list(self.chat_monitors.values()):
                mon.reload_channel_emotes()
            self.after(0, self.refresh_streamers_cards)
        except Exception:
            pass

    def remove_streamer(self, ch: str) -> None:
        if self.is_running or self.is_training:
            messagebox.showwarning("Остановите мониторинг", "Сначала остановите мониторинг перед удалением канала.")
            return
        if ch in self.channels:
            self.channels.remove(ch)
        if self.selected_chart_channel == ch:
            self.selected_chart_channel = self.channels[0] if self.channels else ""
        self.save_settings_from_ui(silent=True)
        self.refresh_streamers_cards()
        self._draw_chart()

    def open_webcam_editor(self, ch: str) -> None:
        profiles = self.cfg.setdefault("channel_profiles", {})
        cur_prof = profiles.get(ch) or {}
        init_crop = cur_prof.get("webcam_crop")

        def _on_saved(channel_name: str, crop_dict: dict) -> None:
            prof = self.cfg.setdefault("channel_profiles", {}).setdefault(channel_name, {})
            if "baseline_mean" not in prof:
                prof["baseline_mean"] = 8.0
                prof["z_score_threshold"] = float(self.slider_z.get())
                prof["min_messages_in_window"] = 6
            prof["webcam_crop"] = crop_dict
            self._save_config_dict(self.cfg)
            self._rendered_cards_state = None
            self.refresh_streamers_cards()
            self.append_log(
                f"📷 [@{channel_name}] Сохранена индивидуальная рамка вебкамеры: "
                f"x={crop_dict['x']:.2f}, y={crop_dict['y']:.2f}, w={crop_dict['w']:.2f}, h={crop_dict['h']:.2f}",
                "success",
            )

        WebcamEditorDialog(
            parent=self,
            channel=ch,
            base_dir=self.base_dir,
            cfg=self.cfg,
            initial_crop=init_crop,
            on_save_crop=_on_saved,
        )

    def select_streamer_for_chart(self, ch: str) -> None:
        self.selected_chart_channel = ch
        self.chart_title_lbl.configure(
            text=f"📈 Живой график скорости чата [{ch}] (столбик = 10 сек, красный = ВЗРЫВ ЧАТА)"
        )
        self.refresh_streamers_cards()
        self._draw_chart()

    def _format_profile_text(self, ch: str, profiles: dict) -> tuple[str, str]:
        prof = profiles.get(ch)
        active_z = float((self.cfg.get("detector") or {}).get("z_score_threshold", 3.3))
        em_cnt, _, _ = self.emote_mgr.get_channel_emote_stats(ch)
        em_str = f"🎭 7TV/BTTV: {em_cnt}" if em_cnt > 0 else "🎭 7TV: глобальные"
        if prof:
            has_cam = "📷 Вебка запомнена" if prof.get("webcam_crop") else "📷 Вебка: ИИ-авто"
            kws = ", ".join(prof.get("custom_keywords", [])[:4]) or "базовые"
            prof_txt = (
                f"✅ ИИ-профиль • порог Z≥{active_z:.2f} • {has_cam} • {em_str} • Сленг: {kws}"
            )
            return prof_txt, COLOR_MUTED
        return f"⚙️ Профиль еще не обучен ({em_str}) — нажмите «🧠 Обучить и Тест (до стрима)» слева внизу", COLOR_YELLOW

    def refresh_streamers_cards(self) -> None:
        profiles = self.cfg.get("channel_profiles", {}) or {}
        trained_cnt = sum(1 for c in self.channels if c in profiles)
        live_cnt = sum(1 for r in self.recorders.values() if r.is_live)

        self.sb_streamers_lbl.configure(text=f"Стримеров: {len(self.channels)} (Онлайн: {live_cnt})")
        self.sb_trained_lbl.configure(text=f"Обучено ИИ-профилей: {trained_cnt}/{len(self.channels)}")

        desired_state = (tuple(self.channels), self.selected_chart_channel)
        if self._rendered_cards_state == desired_state and self.channels:
            # Обновляем текст карточек на месте без пересоздания кнопок (без мерцания!)
            for ch in self.channels:
                w_dict = self._card_widgets.get(ch)
                if not w_dict:
                    continue
                live_str = self.channel_live_state.get(ch, "⏹ Ожидание запуска")
                live_col = COLOR_GREEN if "ОНЛАЙН" in live_str else COLOR_YELLOW
                w_dict["live"].configure(text=live_str, text_color=live_col)

                chat_now = self.channel_chat_state.get(ch, "")
                w_dict["chat"].configure(text=f"💬 {chat_now}" if chat_now else "")

                prof_txt, prof_color = self._format_profile_text(ch, profiles)
                w_dict["prof"].configure(text=prof_txt, text_color=prof_color)
            return

        self._rendered_cards_state = desired_state
        self._card_widgets.clear()
        for child in self.streamers_scroll.winfo_children():
            child.destroy()

        if not self.channels:
            ctk.CTkLabel(
                self.streamers_scroll,
                text="Список пуст. Вставьте ник или ссылку на стримера в поле выше!",
                font=ctk.CTkFont(size=13),
                text_color=COLOR_MUTED,
            ).pack(pady=35)
            return

        for ch in self.channels:
            is_sel = (ch == self.selected_chart_channel)
            card = ctk.CTkFrame(
                self.streamers_scroll,
                corner_radius=10,
                fg_color="#1d1d28" if is_sel else COLOR_CARD,
                border_width=2 if is_sel else 1,
                border_color=COLOR_PURPLE if is_sel else COLOR_BORDER,
            )
            card.pack(fill="x", padx=4, pady=4)

            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=8)

            left_info = ctk.CTkFrame(row, fg_color="transparent")
            left_info.pack(side="left", fill="x", expand=True)

            top_line = ctk.CTkFrame(left_info, fg_color="transparent")
            top_line.pack(anchor="w")

            ctk.CTkLabel(
                top_line,
                text=f"@{ch}",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=COLOR_TEXT,
            ).pack(side="left", padx=(0, 10))

            live_str = self.channel_live_state.get(ch, "⏹ Ожидание запуска")
            live_col = COLOR_GREEN if "ОНЛАЙН" in live_str else COLOR_YELLOW
            lbl_live = ctk.CTkLabel(
                top_line,
                text=live_str,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=live_col,
            )
            lbl_live.pack(side="left", padx=(0, 12))

            chat_now = self.channel_chat_state.get(ch, "")
            lbl_chat = ctk.CTkLabel(
                top_line,
                text=f"💬 {chat_now}" if chat_now else "",
                font=ctk.CTkFont(size=12),
                text_color=COLOR_CYAN,
            )
            lbl_chat.pack(side="left")

            prof_txt, prof_color = self._format_profile_text(ch, profiles)
            lbl_prof = ctk.CTkLabel(
                left_info,
                text=prof_txt,
                font=ctk.CTkFont(size=11),
                text_color=prof_color,
            )
            lbl_prof.pack(anchor="w", pady=(2, 0))

            self._card_widgets[ch] = {"live": lbl_live, "chat": lbl_chat, "prof": lbl_prof}

            btns = ctk.CTkFrame(row, fg_color="transparent")
            btns.pack(side="right")

            ctk.CTkButton(
                btns,
                text="📷 Настроить вебку",
                width=136,
                height=30,
                corner_radius=8,
                fg_color=COLOR_INPUT,
                hover_color=COLOR_TEAL_HOVER,
                text_color=COLOR_CYAN,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda c=ch: self.open_webcam_editor(c),
            ).pack(side="left", padx=4)

            ctk.CTkButton(
                btns,
                text="📈 График",
                width=84,
                height=30,
                corner_radius=8,
                fg_color=COLOR_PURPLE if is_sel else COLOR_INPUT,
                hover_color=COLOR_PURPLE_HOVER,
                font=ctk.CTkFont(size=12),
                command=lambda c=ch: self.select_streamer_for_chart(c),
            ).pack(side="left", padx=4)

            ctk.CTkButton(
                btns,
                text="✕",
                width=34,
                height=30,
                corner_radius=8,
                fg_color=COLOR_INPUT,
                hover_color=COLOR_RED,
                text_color="#ff7675",
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda c=ch: self.remove_streamer(c),
            ).pack(side="left", padx=(4, 0))

    # =================== ВИЗУАЛЬНАЯ ГАЛЕРЕЯ КЛИПОВ С ПРЕВЬЮ ===================

    def _get_clip_thumbnail(self, mp4_path: Path) -> Optional[ctk.CTkImage]:
        key = f"{mp4_path.name}_{int(mp4_path.stat().st_mtime)}"
        if key in self._thumb_cache:
            return self._thumb_cache[key]

        thumb_dir = self.base_dir / self.cfg.get("paths", {}).get("buffer_dir", "buffer") / "_thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        jpg_path = thumb_dir / f"{key}.jpg"

        if not jpg_path.exists():
            cmd = [
                get_ffmpeg_path(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "0.5",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=180:320:force_original_aspect_ratio=decrease",
                str(jpg_path),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=10, creationflags=WIN_NO_WINDOW)
            except Exception:
                return None

        if jpg_path.exists():
            try:
                pil_img = Image.open(jpg_path)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(76, 135))
                self._thumb_cache[key] = ctk_img
                return ctk_img
            except Exception:
                return None
        return None

    def refresh_clips_gallery(self) -> None:
        for child in self.clips_scroll.winfo_children():
            child.destroy()

        clips_dir = self.base_dir / self.cfg.get("paths", {}).get("clips_dir", "clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

        with self._yt_cond:
            self._yt_candidates_pool = [
                j for j in self._yt_candidates_pool if Path(j["video_path"]).exists()
            ]
            self._yt_candidates_pool.sort(key=lambda j: float(j.get("hype_rank", 0.0)), reverse=True)
            pool_snapshot = list(self._yt_candidates_pool)
            deadline_ts = self._yt_round_deadline_ts

        rank_by_name: dict[str, tuple[int, int, dict]] = {}
        total_pool = len(pool_snapshot)
        for idx, job_item in enumerate(pool_snapshot, 1):
            rank_by_name[Path(job_item["video_path"]).name] = (idx, total_pool, job_item)

        def _clip_sort_key(p: Path) -> tuple[int, float]:
            rec_info = rank_by_name.get(p.name)
            if rec_info is not None:
                # Активные кандидаты текущего раунда сортируются строго по рейтингу хайпа (№1 лидер сверху!)
                return (1, float(rec_info[2].get("hype_rank", 0.0)))
            return (0, p.stat().st_mtime if p.exists() else 0.0)

        mp4_files = sorted(
            clips_dir.glob("*.mp4"),
            key=_clip_sort_key,
            reverse=True,
        )
        pool_txt = f" • В отборе сейчас: {total_pool} шт." if total_pool > 0 else ""
        self.gallery_title_lbl.configure(
            text=f"🎬 Готовые вертикальные клипы TikTok 9:16 ({len(mp4_files)}{pool_txt})"
        )

        if not mp4_files:
            ctk.CTkLabel(
                self.clips_scroll,
                text="Пока нет готовых клипов. Нажмите «🧠 Обучить и Тест (до стрима)», чтобы создать первый пробный клип!",
                font=ctk.CTkFont(size=14),
                text_color=COLOR_MUTED,
            ).pack(pady=60)
            return

        for f in mp4_files[:40]:
            try:
                st = f.stat()
            except OSError:
                continue

            card = ctk.CTkFrame(
                self.clips_scroll,
                corner_radius=12,
                fg_color="#13131a",
                border_width=1,
                border_color=COLOR_BORDER,
            )
            card.pack(fill="x", padx=10, pady=6)

            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=10)

            thumb = self._get_clip_thumbnail(f)
            if thumb is not None:
                lbl_img = ctk.CTkLabel(inner, text="", image=thumb)
                lbl_img.pack(side="left", padx=(0, 14))

            info = ctk.CTkFrame(inner, fg_color="transparent")
            info.pack(side="left", fill="both", expand=True)

            title_row = ctk.CTkFrame(info, fg_color="transparent")
            title_row.pack(anchor="w", fill="x", pady=(4, 2))

            ctk.CTkLabel(
                title_row,
                text=f.name,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=COLOR_TEXT,
            ).pack(side="left", padx=(0, 10))

            low_name = f.name.lower()
            if "без_баннера" in low_name or "no_banner" in low_name:
                ctk.CTkLabel(
                    title_row,
                    text="✨ БЕЗ БАННЕРА",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#11382d",
                    text_color=COLOR_GREEN,
                    corner_radius=6,
                    padx=8,
                    pady=2,
                ).pack(side="left")
            elif "с_баннером" in low_name or "_banner" in low_name:
                ctk.CTkLabel(
                    title_row,
                    text="📢 С БАННЕРОМ",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#35204d",
                    text_color=COLOR_YELLOW,
                    corner_radius=6,
                    padx=8,
                    pady=2,
                ).pack(side="left")

            yt_rec = self.yt_uploader.get_uploaded_record(f.name)
            if yt_rec is not None:
                ctk.CTkLabel(
                    title_row,
                    text="📺 НА YOUTUBE",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#3d1414",
                    text_color="#ff6b6b",
                    corner_radius=6,
                    padx=8,
                    pady=2,
                ).pack(side="left", padx=(8, 0))

            cand_info = rank_by_name.get(f.name)
            if cand_info is not None:
                c_pos, c_tot, c_job = cand_info
                c_rank = float(c_job.get("hype_rank", 0.0))
                rem_m = max(1, int(round(max(0.0, deadline_ts - time.time()) / 60.0)))
                if c_pos == 1:
                    ctk.CTkLabel(
                        title_row,
                        text=f"👑 ЛИДЕР ОТБОРА (№1 из {c_tot} • хайп {c_rank:.0f} • публ. ~{rem_m}м)",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        fg_color="#4a3700",
                        text_color="#ffd32a",
                        corner_radius=6,
                        padx=8,
                        pady=2,
                    ).pack(side="left", padx=(8, 0))
                else:
                    top_rank = float(pool_snapshot[0].get("hype_rank", 0.0)) if pool_snapshot else c_rank
                    ctk.CTkLabel(
                        title_row,
                        text=f"🎯 КАНДИДАТ №{c_pos} из {c_tot} (хайп {c_rank:.0f} • у лидера {top_rank:.0f})",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        fg_color="#1e2738",
                        text_color=COLOR_CYAN,
                        corner_radius=6,
                        padx=8,
                        pady=2,
                    ).pack(side="left", padx=(8, 0))

            dt_str = datetime.fromtimestamp(st.st_mtime).strftime("%d.%m.%Y в %H:%M:%S")
            size_mb = st.st_size / (1024 * 1024)
            ctk.CTkLabel(
                info,
                text=f"📅 Создан: {dt_str}   •   📦 Размер: {size_mb:.1f} МБ   •   📱 Формат: TikTok 9:16 (1080x1920)",
                font=ctk.CTkFont(size=12),
                text_color=COLOR_MUTED,
            ).pack(anchor="w", pady=(0, 10))

            btn_row = ctk.CTkFrame(info, fg_color="transparent")
            btn_row.pack(anchor="w")

            ctk.CTkButton(
                btn_row,
                text="▶ Смотреть клип",
                width=145,
                height=34,
                corner_radius=8,
                fg_color=COLOR_PURPLE,
                hover_color=COLOR_PURPLE_HOVER,
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda p=f: os.startfile(str(p.resolve())),
            ).pack(side="left", padx=(0, 8))

            if yt_rec is not None and yt_rec.get("url"):
                ctk.CTkButton(
                    btn_row,
                    text="🔗 Открыть Short",
                    width=148,
                    height=34,
                    corner_radius=8,
                    fg_color="#1d3b2a",
                    hover_color="#26543a",
                    text_color=COLOR_GREEN,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    command=lambda u=str(yt_rec["url"]): webbrowser.open(u),
                ).pack(side="left", padx=(0, 8))
            else:
                ctk.CTkButton(
                    btn_row,
                    text="📺 На YouTube",
                    width=140,
                    height=34,
                    corner_radius=8,
                    fg_color="#c4302b",
                    hover_color="#e53935",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    command=lambda p=f: self._open_youtube_upload_dialog(p),
                ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                btn_row,
                text="📂 Показать в папке",
                width=150,
                height=34,
                corner_radius=8,
                fg_color=COLOR_INPUT,
                hover_color="#323242",
                font=ctk.CTkFont(size=12),
                command=lambda p=f: subprocess.Popen(["explorer", "/select,", str(p.resolve())]),
            ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                btn_row,
                text="🗑 Удалить",
                width=100,
                height=34,
                corner_radius=8,
                fg_color=COLOR_INPUT,
                hover_color=COLOR_RED,
                text_color="#ff7675",
                font=ctk.CTkFont(size=12),
                command=lambda p=f: self._delete_clip_file(p),
            ).pack(side="left")

    def _delete_clip_file(self, path: Path) -> None:
        if path.exists() and messagebox.askyesno("Удалить клип?", f"Удалить файл {path.name}?"):
            try:
                path.unlink()
            except OSError:
                pass
            self.refresh_clips_gallery()

    # =================== YOUTUBE SHORTS (API v3 OAUTH, ОЧЕРЕДЬ, ДИАЛОГ) ===================

    def pick_yt_secrets_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Выберите скачанный JSON-файл OAuth клиента Google Cloud (client_secrets.json)",
            initialdir=str(Path.home() / "Downloads"),
            filetypes=[("JSON файлы Google OAuth", "*.json"), ("Все файлы", "*.*")],
        )
        if not chosen:
            return
        src = Path(chosen).resolve()
        dst = (self.base_dir / "client_secrets.json").resolve()
        try:
            # Копируем в папку проекта, чтобы чистка папки Загрузки не сломала авторизацию
            if src != dst:
                dst.write_bytes(src.read_bytes())
            display_path = "client_secrets.json"
        except Exception:
            display_path = str(src)
        self.yt_secrets_entry.delete(0, "end")
        self.yt_secrets_entry.insert(0, display_path)
        self.save_settings_from_ui(silent=True)
        self._refresh_yt_status_label()
        self.append_log(f"🔑 Файл ключей Google OAuth подключён: {display_path}", "success")
        if messagebox.askyesno(
            "Ключ подключён!",
            "Файл client_secrets.json сохранён.\nОткрыть ваш обычный браузер для авторизации YouTube-канала прямо сейчас?",
        ):
            self.start_youtube_login()

    def show_youtube_setup_help(self) -> None:
        dlg = ctk.CTkToplevel(self)
        dlg.title("❓ Как бесплатно подключить YouTube Data API v3")
        dlg.geometry("680x500")
        dlg.configure(fg_color=COLOR_BG)
        dlg.transient(self)
        dlg.grab_set()

        wrap = ctk.CTkFrame(dlg, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        wrap.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(
            wrap,
            text="📘 Пошаговая настройка официального YouTube Data API v3 (бесплатно)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLOR_CYAN,
        ).pack(anchor="w", padx=16, pady=(14, 8))

        box = ctk.CTkTextbox(
            wrap, corner_radius=10, fg_color="#121218", border_width=1, border_color=COLOR_BORDER,
            font=ctk.CTkFont(size=13), wrap="word"
        )
        box.insert("1.0", SETUP_HELP_TEXT)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        btns = ctk.CTkFrame(wrap, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkButton(
            btns,
            text="🌐 Открыть Google Cloud Console",
            height=38,
            corner_radius=10,
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: webbrowser.open("https://console.cloud.google.com/apis/library/youtube.googleapis.com"),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btns,
            text="📂 Выбрать скачанный JSON",
            height=38,
            corner_radius=10,
            fg_color=COLOR_TEAL,
            hover_color=COLOR_TEAL_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: (dlg.destroy(), self.pick_yt_secrets_file()),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            btns,
            text="Закрыть",
            height=38,
            width=100,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            command=dlg.destroy,
        ).pack(side="right")

    def _refresh_yt_status_label(self) -> None:
        if not hasattr(self, "yt_status_lbl"):
            return
        info = self.yt_uploader.get_auth_info()
        today_cnt, _ = self.yt_uploader.get_today_upload_stats()
        max_day = int(self.yt_uploader.cfg.get("max_per_day", 50))
        if info.get("logged_in"):
            ch = str(info.get("channel_name") or "").strip()
            ch_txt = f" («{ch}»)" if ch else ""
            self.yt_status_lbl.configure(
                text=f"🟢 API v3 подключён{ch_txt} • Сегодня залито: {today_cnt}/{max_day}",
                text_color=COLOR_GREEN,
            )
        elif self.yt_uploader.has_client_secrets():
            self.yt_status_lbl.configure(
                text="🟡 Ключ найден — нажмите «🔑 Авторизовать YouTube (OAuth)»",
                text_color=COLOR_CYAN,
            )
        else:
            self.yt_status_lbl.configure(
                text="⚪ Нужен client_secrets.json (нажмите «❓ Как получить ключ»)",
                text_color=COLOR_YELLOW,
            )

    def start_youtube_login(self) -> None:
        if self._yt_busy:
            messagebox.showinfo("Подождите", "Сейчас уже выполняется операция с YouTube API.")
            return
        self.save_settings_from_ui(silent=True)
        if not self.yt_uploader.has_client_secrets():
            self.show_youtube_setup_help()
            return
        self._yt_busy = True
        self.yt_login_btn.configure(text="⏳ Подтвердите вход в браузере...", state="disabled")

        def _worker() -> None:
            try:
                ok, msg = self.yt_uploader.authorize_oauth(log_cb=lambda m: self.append_log(m, "cyan"))
                self.append_log(f"📺 {msg}", "success" if ok else "warn")

                def _done() -> None:
                    self._refresh_yt_status_label()
                    if ok:
                        messagebox.showinfo("YouTube подключён!", msg)
                    else:
                        messagebox.showwarning("Авторизация не завершена", msg)

                self.after(0, _done)
            finally:
                self._yt_busy = False
                self.after(0, lambda: self.yt_login_btn.configure(text="🔑 Авторизовать YouTube (OAuth)", state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def check_youtube_session(self) -> None:
        if self._yt_busy:
            messagebox.showinfo("Подождите", "Сейчас уже выполняется операция с YouTube API.")
            return
        self.save_settings_from_ui(silent=True)
        self._yt_busy = True
        self.yt_check_btn.configure(text="⏳ Проверка...", state="disabled")

        def _worker() -> None:
            try:
                ok, msg = self.yt_uploader.verify_session()
                self.append_log(f"📺 Проверка YouTube API: {msg}", "success" if ok else "warn")
                self.after(0, self._refresh_yt_status_label)
                self.after(0, lambda: messagebox.showinfo("Статус YouTube Data API v3", f"{'✅' if ok else '⚠️'} {msg}"))
            finally:
                self._yt_busy = False
                self.after(0, lambda: self.yt_check_btn.configure(text="🔍 Проверить канал", state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _open_youtube_upload_dialog(self, video_path: Path) -> None:
        if not self.yt_uploader.is_logged_in_cached():
            if not self.yt_uploader.has_client_secrets():
                self.show_youtube_setup_help()
            elif messagebox.askyesno(
                "Нужна авторизация YouTube",
                "Вы ещё не авторизовали свой YouTube-канал.\nОткрыть ваш обычный браузер для OAuth-входа сейчас?",
            ):
                self.start_youtube_login()
            return

        self.save_settings_from_ui(silent=True)
        meta = self.yt_uploader.get_clip_meta(video_path.name)
        ch = str(meta.get("channel") or "") or parse_channel_from_filename(video_path.name)
        m_tag = str(meta.get("moment_tag") or "")
        tr_txt = str(meta.get("transcript") or "")
        auto_title, auto_desc = build_shorts_metadata(
            channel=ch,
            moment_tag=m_tag,
            transcript_text=tr_txt,
            extra_hashtags=str(self.yt_uploader.cfg.get("extra_hashtags", DEFAULT_YOUTUBE_CFG["extra_hashtags"])),
            title_hashtags=str(self.yt_uploader.cfg.get("title_hashtags", DEFAULT_YOUTUBE_CFG["title_hashtags"])),
            auto_title=bool(self.yt_uploader.cfg.get("auto_title", True)),
        )

        dlg = ctk.CTkToplevel(self)
        dlg.title("📺 Публикация в YouTube Shorts")
        dlg.geometry("560x440")
        dlg.configure(fg_color=COLOR_BG)
        dlg.transient(self)
        dlg.grab_set()

        wrap = ctk.CTkFrame(dlg, corner_radius=14, fg_color=COLOR_CARD, border_width=1, border_color=COLOR_BORDER)
        wrap.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(
            wrap,
            text=f"📺 Загрузка клипа: {video_path.name}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT,
        ).pack(anchor="w", padx=16, pady=(14, 8))

        ctk.CTkLabel(wrap, text="Заголовок Shorts (до 100 символов):", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=16
        )
        title_entry = ctk.CTkEntry(
            wrap, height=36, corner_radius=8, fg_color=COLOR_INPUT, border_color=COLOR_BORDER, font=ctk.CTkFont(size=13)
        )
        title_entry.insert(0, auto_title)
        title_entry.pack(fill="x", padx=16, pady=(4, 10))

        ctk.CTkLabel(wrap, text="Описание и хэштеги:", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=16
        )
        desc_box = ctk.CTkTextbox(
            wrap, height=110, corner_radius=8, fg_color=COLOR_INPUT, border_width=1, border_color=COLOR_BORDER, font=ctk.CTkFont(size=12)
        )
        desc_box.insert("1.0", auto_desc)
        desc_box.pack(fill="x", padx=16, pady=(4, 10))

        ctk.CTkLabel(wrap, text="Уровень доступа:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16)
        vis_var = ctk.StringVar(
            value=YT_VISIBILITY_CODE_TO_LABEL.get(
                str(self.yt_uploader.cfg.get("visibility", "public")),
                "🌍 Открытый доступ (Public)",
            )
        )
        ctk.CTkOptionMenu(
            wrap,
            variable=vis_var,
            values=list(YT_VISIBILITY_LABELS_TO_CODE.keys()),
            height=34,
            corner_radius=8,
            fg_color=COLOR_INPUT,
            button_color=COLOR_PURPLE,
            button_hover_color=COLOR_PURPLE_HOVER,
        ).pack(fill="x", padx=16, pady=(4, 14))

        def _submit() -> None:
            t_val = (title_entry.get() or "").strip()[:100]
            d_val = (desc_box.get("1.0", "end") or "").strip()
            v_code = YT_VISIBILITY_LABELS_TO_CODE.get(vis_var.get(), "public")
            dlg.destroy()
            self._enqueue_youtube_upload(
                video_path=video_path,
                channel=ch,
                moment_tag="",
                transcript_text="",
                title=t_val,
                description=d_val,
                visibility=v_code,
                is_auto=False,
            )

        btn_bar = ctk.CTkFrame(wrap, fg_color="transparent")
        btn_bar.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkButton(
            btn_bar,
            text="🚀 Опубликовать в YouTube Shorts",
            height=40,
            corner_radius=10,
            fg_color="#c4302b",
            hover_color="#e53935",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=_submit,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            btn_bar,
            text="Отмена",
            height=40,
            width=100,
            corner_radius=10,
            fg_color=COLOR_INPUT,
            hover_color="#323242",
            command=dlg.destroy,
        ).pack(side="right")

    def _can_accept_into_auto_yt_queue(self, viral_score: float) -> tuple[bool, str]:
        """Проверяет до рендера, не исчерпан ли дневной лимит YouTube."""
        if not bool(self.yt_uploader.cfg.get("enabled", False)):
            return True, ""
        reached, cnt, mx = self.yt_uploader.is_daily_limit_reached()
        if reached:
            return False, f"дневной лимит YouTube ({cnt}/{mx}) уже выполнен"
        return True, ""

    def _enqueue_youtube_upload(
        self,
        video_path: Path,
        channel: str,
        moment_tag: str = "",
        transcript_text: str = "",
        title: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        is_auto: bool = False,
        viral_score: float = 100.0,
        peak_score: float = 50.0,
        z_score: float = 4.0,
        words_cnt: int = 0,
    ) -> None:
        hype_rank = compute_clip_hype_rank(
            peak_score=peak_score,
            z_score=z_score,
            viral_score=viral_score,
            transcript_text=transcript_text,
            words_cnt=words_cnt,
        )
        _sp_delta, sp_note, _is_bait = evaluate_transcript_hype(transcript_text, words_cnt=words_cnt)
        job = {
            "video_path": Path(video_path),
            "channel": channel,
            "moment_tag": moment_tag,
            "transcript_text": transcript_text,
            "title": title,
            "description": description,
            "visibility": visibility,
            "is_auto": is_auto,
            "viral_score": float(viral_score),
            "peak_score": float(peak_score),
            "z_score": float(z_score),
            "words_cnt": int(words_cnt),
            "speech_note": sp_note,
            "hype_rank": hype_rank,
            "created_ts": time.time(),
        }

        if not is_auto:
            with self._yt_cond:
                self._yt_manual_jobs.append(job)
                self._yt_cond.notify_all()
            self.append_log(f"📺 [ручная заливка] Добавлен в очередь YouTube Shorts: {Path(video_path).name}", "cyan")
            return

        # РЕЖИМ ТУРНИРНОГО ОТБОРА («👑 Король часа»):
        # 1. Все нарезанные клипы-кандидаты СОХРАНЯЮТСЯ в папке clips/ и видны в галерее с их местом (№1, №2, №3...).
        # 2. Если мониторинг только запущен и YouTube не на часовой паузе — первый раунд отбора длится всего 12 минут,
        #    чтобы не ждать целый час до первой публикации! Последующие раунды идут раз в 60 минут.
        # 3. В конце раунда публикуется №1 лучший клип из всех накопленных кандидатов, после чего все кандидаты раунда очищаются.
        interval_sec = max(0.0, float(self.yt_uploader.cfg.get("min_interval_minutes", 60)) * 60.0)
        prev_leader_info = ""
        is_new_leader = False
        first_in_round = False
        my_pos = 1
        total_cand = 1
        rem_min = 0

        with self._yt_cond:
            now_ts = time.time()
            self._yt_candidates_pool = [
                j for j in self._yt_candidates_pool if Path(j["video_path"]).exists()
            ]
            self._yt_candidates_pool.sort(key=lambda j: float(j.get("hype_rank", 0.0)), reverse=True)

            if not self._yt_candidates_pool:
                first_in_round = True
                self._yt_candidates_pool.append(job)
                wait_from_last = self.yt_uploader.seconds_until_next_auto_upload()
                if wait_from_last > 60.0:
                    eff_wait = wait_from_last
                else:
                    # Первый запуск (или час с прошлой заливки уже прошёл): копим кандидатов 12 минут, а не целый час!
                    eff_wait = min(interval_sec, 12.0 * 60.0) if interval_sec > 0 else 0.0
                self._yt_round_deadline_ts = now_ts + eff_wait
                rem_min = max(1, int(round(eff_wait / 60.0))) if eff_wait >= 30 else 0
                my_pos = 1
                total_cand = 1
                self._yt_cond.notify_all()
            else:
                prev_top = self._yt_candidates_pool[0]
                prev_top_rank = float(prev_top.get("hype_rank", 0.0))
                prev_leader_info = f"@{prev_top['channel']} (хайп {prev_top_rank:.0f})"
                self._yt_candidates_pool.append(job)
                self._yt_candidates_pool.sort(key=lambda j: float(j.get("hype_rank", 0.0)), reverse=True)
                total_cand = len(self._yt_candidates_pool)
                my_pos = next(
                    (i for i, j in enumerate(self._yt_candidates_pool, 1) if j is job),
                    total_cand,
                )
                is_new_leader = my_pos == 1 and hype_rank > prev_top_rank
                rem_min = max(1, int(round(max(0.0, self._yt_round_deadline_ts - now_ts) / 60.0)))
                self._yt_cond.notify_all()

        if first_in_round:
            when_txt = f"через ~{rem_min} мин" if rem_min > 0 else "сейчас"
            self.append_log(
                f"👑 [@{channel}] Стартовал раунд отбора! Кандидат №1 (хайп {hype_rank:.0f}, пик чата {peak_score:.0f}, "
                f"🧠 речь: {sp_note}). Публикация лучшего из накопленных клипов {when_txt}.",
                "cyan",
            )
        elif is_new_leader:
            self.append_log(
                f"👑🔥 [@{channel}] НОВЫЙ ЛИДЕР ОТБОРА (№1 из {total_cand} клипов в папке • хайп {hype_rank:.0f}, "
                f"🧠 речь: {sp_note} — обошёл {prev_leader_info})! До публикации лучшего ~{rem_min} мин.",
                "success",
            )
        else:
            self.append_log(
                f"🎯 [@{channel}] Добавлен кандидат в отбор: место №{my_pos} из {total_cand} (хайп {hype_rank:.0f}, "
                f"🧠 речь: {sp_note} • лидер сейчас {prev_leader_info}, до публикации ~{rem_min} мин).",
                "info",
            )
        self.after(0, self.refresh_clips_gallery)
        self.after(0, self._refresh_yt_status_label)

    def _yt_worker_loop(self) -> None:
        while True:
            job: Optional[dict] = None
            with self._yt_cond:
                while True:
                    if self._yt_manual_jobs:
                        job = self._yt_manual_jobs.pop(0)
                        break
                    self._yt_candidates_pool = [
                        j for j in self._yt_candidates_pool if Path(j["video_path"]).exists()
                    ]
                    if self._yt_candidates_pool:
                        self._yt_candidates_pool.sort(
                            key=lambda j: float(j.get("hype_rank", 0.0)), reverse=True
                        )
                        now_ts = time.time()
                        wait_round = max(0.0, self._yt_round_deadline_ts - now_ts)
                        wait_api = self.yt_uploader.seconds_until_next_auto_upload()
                        wait_sec = max(wait_round, wait_api)
                        if wait_sec <= 0.5:
                            job = self._yt_candidates_pool[0]
                            job["total_candidates"] = len(self._yt_candidates_pool)
                            self._yt_candidates_pool.clear()
                            self._yt_round_deadline_ts = 0.0
                            break
                        self._yt_cond.wait(timeout=min(5.0, wait_sec))
                        continue
                    self._yt_cond.wait(timeout=5.0)

            if job is None:
                continue

            try:
                if not Path(job["video_path"]).exists():
                    continue
                if job["is_auto"]:
                    ok_el, skip_why = self.yt_uploader.check_auto_upload_eligibility(
                        job["video_path"], float(job.get("viral_score", 100.0)), ignore_interval=True
                    )
                    if not ok_el:
                        self.yt_uploader.delete_clip_and_siblings(Path(job["video_path"]), cleanup_all_other_clips=False)
                        self.append_log(f"📺 Пропуск авто-заливки ({job['video_path'].name}): {skip_why} (удалён с ПК)", "muted")
                        self.after(0, self.refresh_clips_gallery)
                        continue
                    tot_cand = int(job.get("total_candidates", 1))
                    self.append_log(
                        f"🏆📺 Часовой отбор завершён! Публикую ЛУЧШИЙ клип часа (№1 из {tot_cand} кандидатов): "
                        f"@{job['channel']} (хайп {job.get('hype_rank', 0):.0f})...",
                        "success",
                    )
                self._yt_busy = True
                res = self.yt_uploader.upload_video(
                    video_path=job["video_path"],
                    title=job["title"],
                    description=job["description"],
                    channel=job["channel"],
                    moment_tag=job["moment_tag"],
                    transcript_text=job["transcript_text"],
                    visibility=job["visibility"],
                    is_auto=job["is_auto"],
                    log_cb=lambda m: self.append_log(m, "cyan"),
                )
                if res.ok:
                    url_part = f" -> {res.video_url}" if res.video_url else ""
                    priv_part = f" [{res.actual_privacy}]" if res.actual_privacy else ""
                    del_part = " • 🗑 все временные клипы удалены с ПК" if res.deleted_local else ""
                    self.append_log(
                        f"✅📺 ОПУБЛИКОВАНО В YOUTUBE SHORTS{priv_part}: «{res.title}»{url_part}{del_part}",
                        "success",
                    )
                    if res.warning:
                        self.append_log(f"⚠️📺 {res.warning}", "warn")
                    self.after(0, self.refresh_clips_gallery)
                    self.after(0, self._refresh_yt_status_label)
                else:
                    self.append_log(f"❌📺 Ошибка загрузки в YouTube: {res.error}", "spike")
                    self.after(0, self._refresh_yt_status_label)
                    if res.needs_login and not job["is_auto"]:
                        self.after(
                            0,
                            lambda e=res.error: messagebox.showwarning("Требуется авторизация YouTube", e),
                        )
            except Exception as exc:
                self.append_log(f"❌📺 Сбой загрузчика YouTube: {exc!r}", "spike")
            finally:
                self._yt_busy = False

    # =================== ЛОГИКА ОБУЧЕНИЯ И МОНИТОРИНГА ===================

    def append_log(self, text: str, tag: str = "info") -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        line = f"[{now_str}] {text}\n"

        def _do() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line, tag)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(0, _do)

    def _draw_chart(self) -> None:
        c = self.chart_canvas
        c.delete("all")
        w = max(100, c.winfo_width())
        h = max(60, c.winfo_height())

        ch = self.selected_chart_channel
        history = self.chart_history_by_channel.get(ch, [])

        if not history:
            msg = (
                f"Ожидание данных из чата @{ch} (обновление каждые 10 сек)..."
                if ch
                else "Добавьте стримера сверху, чтобы видеть живой график чата..."
            )
            c.create_text(w // 2, h // 2, text=msg, fill=COLOR_MUTED, font=("Segoe UI", 10))
            return

        max_slots = 30
        data = history[-max_slots:]
        max_val = max([max(s, m * 1.5, 15.0) for s, m, _ in data] + [20.0])

        bar_gap = 6
        bar_w = max(8, (w - 24 - bar_gap * max_slots) / max_slots)

        for idx, (score, _mean_val, is_spike) in enumerate(data):
            x0 = 12 + idx * (bar_w + bar_gap)
            x1 = x0 + bar_w
            bar_h = int((score / max_val) * (h - 24))
            y1 = h - 8
            y0 = max(10, y1 - max(3, bar_h))

            color = COLOR_RED if is_spike else COLOR_PURPLE
            c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            if score > 0:
                c.create_text(
                    (x0 + x1) / 2,
                    max(7, y0 - 7),
                    text=f"{int(score)}",
                    fill=COLOR_MUTED if not is_spike else "#ff8080",
                    font=("Segoe UI", 8),
                )

        # Линия текущей живой нормы чата (медиана)
        latest_mean = data[-1][1]
        if latest_mean > 0:
            mean_h = int((latest_mean / max_val) * (h - 24))
            y_mean = max(10, (h - 8) - max(3, mean_h))
            c.create_line(10, y_mean, w - 10, y_mean, fill=COLOR_CYAN, dash=(4, 4), width=1)
            c.create_text(
                w - 14,
                max(9, y_mean - 8),
                anchor="e",
                text=f"норма ~{latest_mean:.0f}",
                fill=COLOR_CYAN,
                font=("Segoe UI", 8, "bold"),
            )

    def save_settings_from_ui(self, silent: bool = False) -> bool:
        try:
            self.cfg["channels"] = list(self.channels)
            self.cfg["channel"] = self.channels[0] if self.channels else ""
            stream_cfg = self.cfg.setdefault("stream", {})
            stream_cfg["quality"] = self.quality_var.get()
            stream_cfg["check_interval"] = 300

            det = self.cfg.setdefault("detector", {})
            det["z_score_threshold"] = round(float(self.slider_z.get()), 2)
            det["min_viral_score"] = round(float(self.slider_min_viral.get()), 1)
            det["pre_roll_seconds"] = round(float(self.slider_pre.get()), 1)
            det["post_roll_seconds"] = round(float(self.slider_post.get()), 1)
            det["max_clip_seconds"] = round(float(self.slider_max_clip.get()), 1)
            det["cooldown_seconds"] = round(float(self.slider_cooldown.get()), 1)
            det["global_cooldown_seconds"] = round(float(self.slider_global_cooldown.get()), 1)
            det["use_audio_peaks"] = bool(self.var_audio_peaks.get())
            det["hype_keywords"] = [
                kw.strip() for kw in self.keywords_entry.get().split(",") if kw.strip()
            ]

            vid = self.cfg.setdefault("video", {})
            vid["mode"] = "tiktok"
            vid["webcam_position"] = WEBCAM_LABELS_TO_CODE.get(self.webcam_label_var.get(), "auto")
            vid["center_zoom"] = round(float(self.slider_zoom.get()), 2)
            vid["climax_zoom"] = bool(self.var_climax_zoom.get())
            vid["use_nvenc"] = bool(self.var_use_nvenc.get())

            ban = vid.setdefault("banner", {})
            ban["enabled"] = bool(self.var_banner_enabled.get())
            ban["pause_video"] = bool(self.var_banner_pause.get())
            ban["path"] = (self.banner_path_entry.get() or "").strip()
            ban["position_time"] = BANNER_TIME_LABELS_TO_CODE.get(self.banner_time_var.get(), "smart")
            ban["width_ratio"] = round(float(self.slider_banner_scale.get()) / 100.0, 2)
            ban["pos_y"] = round(float(self.slider_banner_y.get()) / 100.0, 2)
            ban["volume"] = round(float(self.slider_banner_vol.get()) / 100.0, 2)
            ban["chromakey"] = bool(self.var_banner_chromakey.get())
            ban.setdefault("chromakey_color", "auto")
            ban.setdefault("chromakey_similarity", 0.14)
            ban.setdefault("chromakey_blend", 0.06)
            if self.clipper is not None:
                self.clipper.update_banner_cfg(ban)

            sub = self.cfg.setdefault("subtitles", {})
            sub["enabled"] = bool(self.var_sub_enabled.get())
            sub["smart_hook"] = bool(self.var_smart_hook.get())
            sub["model_size"] = self.sub_model_var.get()
            if sub.get("device", "cuda") == "cuda":
                sub["compute_type"] = "int8_float16" if sub["model_size"] in ("large-v3-turbo", "large-v3", "medium") else "float16"
            sub["words_per_phrase"] = int(round(float(self.slider_words.get())))
            sub["font_size"] = int(round(float(self.slider_fontsize.get())))

            yt = self.cfg.setdefault("youtube", {})
            yt_on = bool(self.var_yt_enabled.get())
            yt["enabled"] = yt_on
            yt["auto_upload"] = yt_on
            yt["visibility"] = YT_VISIBILITY_LABELS_TO_CODE.get(self.yt_visibility_var.get(), "public")
            yt["min_viral_score"] = int(round(float(self.slider_yt_viral.get())))
            yt["min_interval_minutes"] = int(round(float(self.slider_yt_interval.get())))
            yt["max_per_day"] = int(round(float(self.slider_yt_maxday.get())))
            yt["client_secrets_path"] = (self.yt_secrets_entry.get() or "client_secrets.json").strip()
            yt.setdefault("token_path", DEFAULT_YOUTUBE_CFG["token_path"])
            yt.setdefault("category_id", DEFAULT_YOUTUBE_CFG["category_id"])
            yt["title_hashtags"] = (self.yt_title_tags_entry.get() or "").strip()
            yt["auto_title"] = bool(self.var_yt_auto_title.get())
            yt["auto_comment"] = bool(self.var_yt_auto_comment.get())
            yt["extra_hashtags"] = (self.yt_hashtags_entry.get() or "").strip()
            yt["delete_after_upload"] = bool(self.var_yt_delete_after.get())
            self.yt_uploader.update_cfg(yt)
            with self._yt_cond:
                self._yt_cond.notify_all()
            self._refresh_yt_status_label()

            self._save_config_dict(self.cfg)
            # Применяем новые пороги сразу и к активным мониторам, если мониторинг уже идет
            for mon in self.chat_monitors.values():
                mon.z_threshold = det["z_score_threshold"]
                mon.min_viral_score = det["min_viral_score"]
                mon.pre_roll = det["pre_roll_seconds"]
                mon.post_roll = det["post_roll_seconds"]
                mon.max_clip_seconds = det["max_clip_seconds"]
                mon.cooldown_seconds = det["cooldown_seconds"]
                mon.use_audio_peaks = det["use_audio_peaks"]
            if not silent:
                self.refresh_streamers_cards()
                self.append_log("💾 Настройки сохранены в config.yaml!", "success")
            return True
        except Exception as exc:
            messagebox.showerror("Ошибка настроек", str(exc))
            return False

    def reset_learned_profiles(self) -> None:
        if messagebox.askyesno("Сбросить обучение?", "Удалить все индивидуально обученные профили стримеров?"):
            self.cfg["channel_profiles"] = {}
            self._save_config_dict(self.cfg)
            self.refresh_streamers_cards()
            self.append_log("Обученные профили стримеров сброшены.", "warn")

    def start_prestream_training(self) -> None:
        if self.is_running or self.is_training:
            messagebox.showinfo("Уже запущено", "Дождитесь завершения текущей задачи или остановите мониторинг.")
            return
        if (self.channel_entry.get() or "").strip():
            self.add_channels_from_entry()
        if not self.channels:
            messagebox.showwarning("Нет стримеров", "Добавьте хотя бы одного стримера в верхнее поле!")
            return

        self.save_settings_from_ui(silent=True)
        self.show_page("dashboard")

        self.is_training = True
        self.train_btn.configure(text="⏳ Идет ИИ-обучение...", state="disabled")
        self.start_btn.configure(state="disabled")
        make_clip = bool(self.make_test_clip_var.get())

        threading.Thread(
            target=self._training_worker_thread,
            args=(list(self.channels), make_clip),
            daemon=True,
        ).start()

    def _training_worker_thread(self, channels: list[str], make_clip: bool) -> None:
        try:
            self.append_log(
                f"🧠 Старт пред-стримового ИИ-обучения и теста для: {', '.join(channels)}",
                "cyan",
            )
            paths_cfg = self.cfg.get("paths", {})
            base_buffer_dir = self.base_dir / paths_cfg.get("buffer_dir", "buffer")
            clips_dir = self.base_dir / paths_cfg.get("clips_dir", "clips")
            video_cfg = self.cfg.get("video", {})
            sub_cfg = self.cfg.get("subtitles", {})

            clipper = None
            if make_clip:
                self.append_log("Загрузка Faster-Whisper (CUDA) и детектора вебкамеры YuNet...", "muted")
                sub_gen = SubtitleGenerator(sub_cfg=sub_cfg, video_cfg=video_cfg)
                if sub_gen.enabled:
                    sub_gen.load_model()
                clipper = VideoClipper(
                    buffer_dir=base_buffer_dir,
                    clips_dir=clips_dir,
                    video_cfg=video_cfg,
                    sub_generator=sub_gen,
                )

            trainer = AutoTrainer(self.base_dir, self.cfg)
            profiles = self.cfg.setdefault("channel_profiles", {})
            # Удаляем ошибочные ключи вроде 'videos', если они остались от старых запусков
            for bad_k in ("videos", "clips", "about"):
                profiles.pop(bad_k, None)

            reports = []
            for idx, ch in enumerate(channels, 1):
                self.after(
                    0,
                    lambda c=ch, i=idx, n=len(channels): self.sb_gpu_lbl.configure(
                        text=f"🧠 Обучение @{c} ({i}/{n})...", text_color=COLOR_YELLOW
                    ),
                )
                rep = trainer.train_and_test_channel(
                    channel=ch,
                    clipper=clipper,
                    make_test_clip=make_clip,
                    log_cb=lambda msg: self.append_log(msg, "info"),
                )
                reports.append(rep)
                prev_cam = (profiles.get(ch) or {}).get("webcam_crop")
                prof_dict = rep.to_profile_dict()
                if "webcam_crop" not in prof_dict and prev_cam:
                    prof_dict["webcam_crop"] = prev_cam
                profiles[ch] = prof_dict

                # Сохраняем только профиль конкретного канала!
                # Глобальные detector.hype_keywords и пользовательский ползунок Z-score НЕ перезаписываем
                # и НЕ засоряем словами других каналов.
                self._save_config_dict(self.cfg)
                self.after(0, self.refresh_streamers_cards)
                if rep.test_clip_path:
                    self.after(0, self.refresh_clips_gallery)

            self.append_log(
                "✅ Обучение и тест завершены! Профили каналов, пороги и координаты вебкамер применены и сохранены.",
                "success",
            )
            if reports:
                summary_lines = []
                for r in reports:
                    cam_str = (
                        f"Найдена ({r.webcam_crop.get('preset_name')}, x={r.webcam_crop.get('x')}, y={r.webcam_crop.get('y')})"
                        if r.webcam_crop
                        else "Авто-режим"
                    )
                    kw_str = ", ".join(r.custom_keywords) if r.custom_keywords else "стандартные"
                    clip_str = r.test_clip_path.name if r.test_clip_path else "выключен"
                    summary_lines.append(
                        f"🎯 Канал @{r.channel}:\n"
                        f"  • Проанализировано: {r.analyzed_messages} сообщ. ({r.analyzed_windows} окон из {r.analyzed_vods} стримов)\n"
                        f"  • Норма чата: {r.baseline_mean:.1f} очк/10с (±{r.baseline_std:.1f})\n"
                        f"  • Применён порог Z-score: {r.z_score_threshold:.2f}\n"
                        f"  • Мин. сообщений за 10с: {r.min_messages_in_window} (мин. зрителей: {r.min_unique_chatters})\n"
                        f"  • Мин. балл хайпа: {r.min_hype_score:.1f}\n"
                        f"  • Вебкамера стримера: {cam_str}\n"
                        f"  • Добавлены сленг/эмоуты: {kw_str}\n"
                        f"  • Тестовый клип (30 сек): {clip_str}"
                    )
                full_summary = "\n\n".join(summary_lines)
                self.after(
                    0,
                    lambda s=full_summary: messagebox.showinfo(
                        "🧠 Результаты ИИ-обучения применены!",
                        f"Настройки успешно подобраны и применены:\n\n{s}",
                    ),
                )
        except Exception as exc:
            self.append_log(f"❌ Ошибка при обучении: {exc!r}", "spike")
        finally:
            self.is_training = False

            def _restore() -> None:
                self.train_btn.configure(text="🧠 Обучить и Тест (до стрима)", state="normal")
                self.start_btn.configure(state="normal")
                self.sb_gpu_lbl.configure(text="⚡ RTX 4050 CUDA + NVENC", text_color=COLOR_CYAN)
                self.refresh_streamers_cards()
                self.refresh_clips_gallery()

            self.after(0, _restore)

    def toggle_monitoring(self) -> None:
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self) -> None:
        if self.is_training:
            return
        if (self.channel_entry.get() or "").strip():
            self.add_channels_from_entry()
        if not self.channels:
            messagebox.showwarning("Добавьте стримеров", "Добавьте хотя бы одного стримера в список!")
            return

        self.save_settings_from_ui(silent=True)
        self.is_running = True
        self.chart_history_by_channel.clear()
        self._draw_chart()

        self.start_btn.configure(text="⏹ ОСТАНОВИТЬ МОНИТОРИНГ", fg_color=COLOR_RED, hover_color=COLOR_RED_HOVER)
        self.train_btn.configure(state="disabled")
        self.manual_clip_btn.configure(state="normal")

        self.worker_thread = threading.Thread(
            target=self._run_async_engine, args=(list(self.channels),), daemon=True
        )
        self.worker_thread.start()

    def stop_monitoring(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.append_log("⏹ Остановка мониторинга всех каналов...", "warn")

        for r in self.recorders.values():
            r.stop()
        for m in self.chat_monitors.values():
            m.stop()
        if self.loop and self.loop.is_running() and getattr(self, "_engine_main_task", None) is not None:
            self.loop.call_soon_threadsafe(self._engine_main_task.cancel)

        self.start_btn.configure(text="▶ ЗАПУСТИТЬ МОНИТОРИНГ", fg_color=COLOR_PURPLE, hover_color=COLOR_PURPLE_HOVER)
        self.train_btn.configure(state="normal")
        self.manual_clip_btn.configure(state="disabled")
        for ch in self.channels:
            self.channel_live_state[ch] = "⏹ Остановлено"
        self.refresh_streamers_cards()

    def trigger_manual_clip(self) -> None:
        if not self.is_running or not self.loop or self.clip_queue is None:
            return
        target_rec: Optional[StreamRecorder] = None
        if self.selected_chart_channel in self.recorders and self.recorders[self.selected_chart_channel].is_live:
            target_rec = self.recorders[self.selected_chart_channel]
        else:
            for r in self.recorders.values():
                if r.is_live:
                    target_rec = r
                    break

        if target_rec is None:
            messagebox.showinfo(
                "Нет активных эфиров",
                "Ручной клип можно сделать, когда хотя бы один из добавленных стримеров сейчас ОНЛАЙН!",
            )
            return

        now = time.time()
        ev = ClipEvent(
            start_ts=now - 30.0,
            end_ts=now,
            peak_score=100.0,
            z_score=5.0,
            total_messages=0,
            sample_messages=["Ручной клип по кнопке"],
            channel=target_rec.channel,
        )
        self.loop.call_soon_threadsafe(self.clip_queue.put_nowait, (target_rec, ev))
        self.append_log(f"✂️ [@{target_rec.channel}] Запрошен ручной клип последних 30 секунд эфира!", "cyan")

    def _run_async_engine(self, channels: list[str]) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            self._engine_main_task = loop.create_task(self._engine_coro(channels))
            loop.run_until_complete(self._engine_main_task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.append_log(f"Ошибка фонового потока: {exc!r}", "spike")
        finally:
            self._engine_main_task = None
            try:
                loop.close()
            except Exception:
                pass
            self.loop = None

    async def _engine_coro(self, channels: list[str]) -> None:
        paths_cfg = self.cfg.get("paths", {})
        base_buffer_dir = self.base_dir / paths_cfg.get("buffer_dir", "buffer")
        clips_dir = self.base_dir / paths_cfg.get("clips_dir", "clips")

        stream_cfg = self.cfg.get("stream", {})
        detector_cfg = self.cfg.get("detector", {})
        video_cfg = self.cfg.get("video", {})
        sub_cfg = self.cfg.get("subtitles", {})
        profiles = self.cfg.get("channel_profiles", {}) or {}

        self.append_log("Инициализация Faster-Whisper и GPU RTX...", "muted")
        self.sub_generator = SubtitleGenerator(sub_cfg=sub_cfg, video_cfg=video_cfg)
        if self.sub_generator.enabled:
            await asyncio.to_thread(self.sub_generator.load_model)

        self.clipper = VideoClipper(
            buffer_dir=base_buffer_dir,
            clips_dir=clips_dir,
            video_cfg=video_cfg,
            sub_generator=self.sub_generator,
        )

        nvenc_txt = "NVENC RTX" if self.clipper.nvenc_available else "CPU x264"
        whisper_txt = f"Whisper {self.sub_generator.model_size}"
        self.after(0, lambda: self.sb_gpu_lbl.configure(text=f"⚡ {nvenc_txt} • {whisper_txt}", text_color=COLOR_CYAN))
        self.append_log(f"🚀 Мониторинг запущен для: {', '.join(channels)}", "success")

        self.clip_queue = asyncio.Queue()
        self.recorders.clear()
        self.chat_monitors.clear()

        for ch in channels:
            ch_buf = base_buffer_dir / ch

            def make_status_cb(c_name: str):
                def _cb(is_live: bool, msg: str) -> None:
                    self.append_log(f"🎥 [@{c_name}] {msg}", "success" if is_live else "warn")
                    self.channel_live_state[c_name] = "🟢 ОНЛАЙН • Запись" if is_live else "⏳ Оффлайн • Ожидание"
                    self.after(0, self.refresh_streamers_cards)
                return _cb

            rec = StreamRecorder(
                channel=ch,
                buffer_dir=ch_buf,
                stream_cfg=stream_cfg,
                on_status_change=make_status_cb(ch),
            )
            self.recorders[ch] = rec

            def on_window_stats(stats: WindowStats) -> None:
                c_name = stats.channel
                aud_str = f" • 🔊 Звук Z{stats.audio_z:+.1f}" if abs(stats.audio_z) >= 0.5 else ""
                react_pct = int(round(stats.reaction_ratio * 100))
                self.channel_chat_state[c_name] = (
                    f"{stats.msg_count} сообщ./10с • Хайп {stats.hype_score:.1f}/{stats.mean_score:.1f} "
                    f"(Z {stats.z_score:+.2f}, эмоций {react_pct}%){aud_str}"
                )
                hist = self.chart_history_by_channel.setdefault(c_name, [])
                hist.append((stats.hype_score, stats.mean_score, stats.is_triggered))

                def _upd_ui() -> None:
                    if self.selected_chart_channel == c_name:
                        if stats.top_messages:
                            self.top_msg_lbl.configure(text=f"💬 {stats.top_messages[0][:55]}")
                        self._draw_chart()
                    self.refresh_streamers_cards()

                self.after(0, _upd_ui)

                r_obj = self.recorders.get(c_name)
                if not r_obj or not r_obj.is_live:
                    return

                top_sample = f" | Топ: {stats.top_messages[0][:42]}" if stats.top_messages else ""
                aud_log = f" | 🔊 Звук Z:{stats.audio_z:+.2f} ({stats.audio_db:.0f}dB)" if stats.audio_db > -55 else ""
                tag_lbl = stats.moment_tag or "🔥 Хайп"
                if stats.is_triggered:
                    self.append_log(
                        f"🔥 [@{c_name}] {tag_lbl}! Сообщ: {stats.msg_count} (эмоций {react_pct}%) | "
                        f"Хайп: {stats.hype_score:.1f} (норма: {stats.mean_score:.1f}, Z: {stats.z_score:+.2f}, "
                        f"Z_эмоц: {stats.reaction_z:+.2f}){aud_log}{top_sample}",
                        "spike",
                    )
                else:
                    self.append_log(
                        f"💬 [@{c_name}] Чат (10с): {stats.msg_count:2d} сообщ. ({react_pct:2d}% эмоц.) | "
                        f"Хайп: {stats.hype_score:5.1f} (норма: {stats.mean_score:4.1f}, Z: {stats.z_score:+.2f}){aud_log}{top_sample}",
                        "muted",
                    )

            def make_clip_cb(r_obj: StreamRecorder):
                def _cb(event: ClipEvent) -> None:
                    if r_obj.is_live and self.clip_queue is not None:
                        self.clip_queue.put_nowait((r_obj, event))
                return _cb

            mon = TwitchChatMonitor(
                channel=ch,
                detector_cfg=detector_cfg,
                on_clip_detected=make_clip_cb(rec),
                on_window_stats=on_window_stats,
                learned_profile=profiles.get(ch),
                get_audio_stats=rec.get_recent_audio_stats,
                is_stream_live=lambda r=rec: r.is_live,
            )
            self.chat_monitors[ch] = mon

        async def clip_worker() -> None:
            assert self.clip_queue is not None
            assert self.clipper is not None
            while self.is_running:
                rec, event = await self.clip_queue.get()
                try:
                    rec.lock_buffer()
                    is_manual = bool(event.sample_messages and "Ручной клип" in event.sample_messages[0])
                    segments = rec.get_segments_for_range(
                        event.start_ts,
                        event.end_ts,
                        min_segments=0 if is_manual else 5,
                    )
                    if not segments:
                        self.append_log(
                            f"⚠️ [@{rec.channel}] Недостаточно сегментов в буфере (<20 сек непрерывной записи, ждём накопления буфера).",
                            "warn",
                        )
                        continue

                    approx_sec = len(segments) * rec.segment_duration
                    m_tag = getattr(event, "moment_tag", "🔥 Хайп")
                    v_sc = float(getattr(event, "viral_score", 0.0))

                    if not is_manual:
                        # 1. Глобальная пауза между нарезками по всем 9 каналам (мега-момент >= 88 пробивает!)
                        g_cool = float((self.cfg.get("detector") or {}).get("global_cooldown_seconds", 360.0))
                        now_g = time.time()
                        if g_cool > 0 and self._last_global_clip_ts > 0 and (now_g - self._last_global_clip_ts) < g_cool and v_sc < 88.0:
                            rem_s = int(round(g_cool - (now_g - self._last_global_clip_ts)))
                            self.append_log(
                                f"⏭ [@{rec.channel}] Пропущен момент ({m_tag}, индекс {v_sc:.0f}/100): "
                                f"общая пауза между каналами (осталось {rem_s}с; мега-хайп ≥88 пробивает).",
                                "muted",
                            )
                            continue
                        # 2. Не тратим GPU и диск, если дневной лимит YouTube выполнен или в очереди уже ждут более сильные клипы
                        can_q, why_q = self._can_accept_into_auto_yt_queue(v_sc)
                        if not can_q:
                            self.append_log(
                                f"⏭ [@{rec.channel}] Рендер пропущен ({m_tag}, индекс {v_sc:.0f}/100): {why_q}.",
                                "muted",
                            )
                            continue

                    self.append_log(
                        f"✂️ [@{rec.channel}] Сборка TikTok-клипа ({m_tag}, ~{approx_sec} сек, пик: {event.peak_score:.1f}, индекс: {v_sc:.0f}/100)...",
                        "warn",
                    )
                    self.after(
                        0,
                        lambda c=rec.channel: self.sb_gpu_lbl.configure(
                            text=f"⚙️ Рендер [@{c}]...", text_color=COLOR_YELLOW
                        ),
                    )

                    target_dur = max(25.0, float(detector_cfg.get("max_clip_seconds", 37.0)))
                    out_path, words_cnt, elapsed = await asyncio.to_thread(
                        self.clipper.create_clip_from_segments,
                        segments,
                        rec.channel,
                        event.peak_score,
                        (profiles.get(rec.channel) or {}).get("webcam_crop"),
                        target_dur,
                        not is_manual,
                    )
                    if out_path:
                        if not is_manual:
                            self._last_global_clip_ts = time.time()
                        ban_log = (
                            f" | 📢 Баннер: {self.clipper.last_banner_pause_info}"
                            if self.clipper.last_banner_pause_info
                            else ""
                        )
                        self.append_log(
                            f"✅ [@{rec.channel}] ГОТОВ КЛИП за {elapsed:.1f} сек! Файл: {out_path.name} (слов: {words_cnt}){ban_log}",
                            "success",
                        )
                        self.after(0, self.refresh_clips_gallery)

                        # Запоминаем сюжет и расшифровку Whisper для умного названия (и для авто, и для ручной заливки)
                        tr_text = self.clipper.last_transcript_text
                        self.yt_uploader.save_clip_meta(out_path.name, rec.channel, m_tag, tr_text)

                        # Авто-отбор «Король часа» в YouTube Shorts (1 лучший клип в час, остальные удаляются с ПК!)
                        if not is_manual and bool(self.yt_uploader.cfg.get("enabled", False)):
                            can_up, why_not = self.yt_uploader.check_auto_upload_eligibility(
                                out_path, v_sc, ignore_interval=True
                            )
                            if can_up:
                                self._enqueue_youtube_upload(
                                    video_path=out_path,
                                    channel=rec.channel,
                                    moment_tag=m_tag,
                                    transcript_text=self.clipper.last_transcript_text,
                                    is_auto=True,
                                    viral_score=v_sc,
                                    peak_score=float(event.peak_score),
                                    z_score=float(event.z_score),
                                    words_cnt=words_cnt,
                                )
                            else:
                                self.yt_uploader.delete_clip_and_siblings(out_path, cleanup_all_other_clips=False)
                                self.append_log(
                                    f"🗑 [@{rec.channel}] Клип не прошёл фильтр YouTube ({why_not}) и удалён с ПК.",
                                    "muted",
                                )
                                self.after(0, self.refresh_clips_gallery)
                    elif self.clipper.last_skip_reason:
                        self.append_log(f"⏭ [@{rec.channel}] {self.clipper.last_skip_reason}.", "warn")
                    else:
                        self.append_log(f"❌ [@{rec.channel}] Не удалось собрать клип.", "spike")
                except Exception as exc:
                    self.append_log(f"❌ [@{rec.channel}] Ошибка при сборке клипа: {exc!r}", "spike")
                finally:
                    rec.unlock_buffer()
                    self.after(
                        0,
                        lambda: self.sb_gpu_lbl.configure(
                            text=f"⚡ {nvenc_txt} • {whisper_txt}", text_color=COLOR_CYAN
                        ),
                    )
                    self.clip_queue.task_done()

        worker_task = asyncio.create_task(clip_worker())
        all_tasks = [r.run() for r in self.recorders.values()] + [m.run() for m in self.chat_monitors.values()]
        try:
            await asyncio.gather(*all_tasks)
        finally:
            worker_task.cancel()

    def _periodic_ui_tick(self) -> None:
        if self.is_running and self.recorders:
            for ch, rec in self.recorders.items():
                if rec.is_live:
                    segs = list(rec.buffer_dir.glob("seg_*.ts"))
                    sec = len(segs) * rec.segment_duration
                    self.channel_live_state[ch] = f"🟢 ОНЛАЙН • Буфер ~{sec}с"
            self.refresh_streamers_cards()
        self.after(3000, self._periodic_ui_tick)

    def open_clips_folder(self) -> None:
        clips_dir = self.base_dir / self.cfg.get("paths", {}).get("clips_dir", "clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(clips_dir.resolve())])

    def on_close(self) -> None:
        self.stop_monitoring()
        self.destroy()


if __name__ == "__main__":
    app = TwitchClipperApp()
    app.mainloop()
