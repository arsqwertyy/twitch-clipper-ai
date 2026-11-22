import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from face_detector import FaceWebcamDetector, WebcamCrop
from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path, get_streamlink_cmd
from trainer import AutoTrainer

CANVAS_W = 680
CANVAS_H = 382  # ровно 16:9
PREVIEW_W = 250
PREVIEW_H = 144  # пропорции верхнего блока TikTok (1080:624)
TARGET_ASPECT = 1080.0 / 624.0


class WebcamEditorDialog(ctk.CTkToplevel):
    """
    Визуальный редактор рамки вебкамеры стримера мышкой:
    - Загружает реальный стоп-кадр со стрима или последнего VOD стримера.
    - Позволяет мышкой двигать зеленую рамку, тянуть за уголки или менять размер ползунком/колесиком.
    - Справа в реальном времени показывает, как будет выглядеть верхний блок в TikTok 9:16.
    """

    def __init__(
        self,
        parent: ctk.CTk,
        channel: str,
        base_dir: Path,
        cfg: dict,
        initial_crop: Optional[dict],
        on_save_crop: Callable[[str, dict], None],
    ):
        super().__init__(parent)
        self.channel = channel
        self.base_dir = base_dir
        self.cfg = cfg
        self.on_save_crop = on_save_crop
        self.detector = FaceWebcamDetector()

        self.title(f"📷 Настройка вебкамеры — @{channel}")
        self.geometry("1000x560")
        self.resizable(False, False)
        self.configure(fg_color="#0e0e12")
        self.transient(parent)
        self.grab_set()

        # Текущие нормализованные координаты (0.0 .. 1.0)
        if initial_crop and "x" in initial_crop:
            self.cx = float(initial_crop["x"])
            self.cy = float(initial_crop["y"])
            self.cw = float(initial_crop["w"])
            self.ch = float(initial_crop["h"])
        else:
            self.cx, self.cy, self.cw, self.ch = 0.0, 0.04, 0.22, 0.23

        self._pil_frame: Optional[Image.Image] = None
        self._tk_canvas_img: Optional[ImageTk.PhotoImage] = None
        self._ctk_preview_img: Optional[ctk.CTkImage] = None
        self._vod_sample_index = 0
        self._Sample_ts_path: Optional[Path] = None

        # Состояние перетаскивания мышкой
        self._drag_mode: Optional[str] = None  # "move" или "resize"
        self._drag_start_xy: tuple[int, int] = (0, 0)
        self._drag_start_box: tuple[float, float, float, float] = (self.cx, self.cy, self.cw, self.ch)

        self._build_ui()
        self._load_frame_async()

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self, fg_color="#181820", corner_radius=12)
        top.pack(fill="x", padx=14, pady=(12, 8))

        self.status_lbl = ctk.CTkLabel(
            top,
            text=f"⏳ Загрузка кадра со стрима @{self.channel}...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00f2fe",
        )
        self.status_lbl.pack(side="left", padx=14, pady=10)

        ctk.CTkButton(
            top,
            text="🔄 Другой кадр со стрима",
            height=32,
            corner_radius=8,
            fg_color="#252532",
            hover_color="#343446",
            font=ctk.CTkFont(size=12),
            command=self._next_vod_frame,
        ).pack(side="right", padx=(6, 12), pady=8)

        ctk.CTkButton(
            top,
            text="🤖 Найти ИИ автоматически",
            height=32,
            corner_radius=8,
            fg_color="#00b894",
            hover_color="#00997a",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._auto_detect_on_current,
        ).pack(side="right", padx=6, pady=8)

        # Основная область: слева Canvas 16:9, справа предпросмотр и управление
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        left_frame = ctk.CTkFrame(body, fg_color="#181820", corner_radius=12)
        left_frame.pack(side="left", fill="both", padx=(0, 10))

        ctk.CTkLabel(
            left_frame,
            text="Перетащите зеленую рамку мышкой на вебкамеру стримера (колесико мыши = размер):",
            font=ctk.CTkFont(size=12),
            text_color="#9c9cab",
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#09090c",
            highlightthickness=1,
            highlightbackground="#2a2a38",
            cursor="fleur",
        )
        self.canvas.pack(padx=12, pady=(0, 12))

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)

        # Правая панель: Предпросмотр верхнего блока TikTok и ползунок размера
        right_frame = ctk.CTkFrame(body, fg_color="#181820", corner_radius=12, width=275)
        right_frame.pack(side="right", fill="both", expand=True)
        right_frame.pack_propagate(False)

        ctk.CTkLabel(
            right_frame,
            text="📱 Предпросмотр в TikTok:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f5f6fa",
        ).pack(anchor="w", padx=14, pady=(12, 4))

        ctk.CTkLabel(
            right_frame,
            text="Верхний блок (1080x624):",
            font=ctk.CTkFont(size=11),
            text_color="#9c9cab",
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.preview_lbl = ctk.CTkLabel(
            right_frame,
            text="Ожидание кадра...",
            width=PREVIEW_W,
            height=PREVIEW_H,
            fg_color="#0b0b0e",
            corner_radius=8,
        )
        self.preview_lbl.pack(padx=14, pady=(0, 14))

        ctk.CTkLabel(
            right_frame,
            text="Размер рамки вебкамеры:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(4, 2))

        self.size_slider = ctk.CTkSlider(
            right_frame,
            from_=0.12,
            to=0.48,
            button_color="#8b44f7",
            progress_color="#8b44f7",
            command=self._on_size_slider,
        )
        self.size_slider.set(self.cw)
        self.size_slider.pack(fill="x", padx=14, pady=(2, 10))

        self.coords_lbl = ctk.CTkLabel(
            right_frame,
            text="",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#9c9cab",
        )
        self.coords_lbl.pack(anchor="w", padx=14, pady=(0, 12))

        ctk.CTkButton(
            right_frame,
            text="💾 Сохранить рамку",
            height=42,
            corner_radius=10,
            fg_color="#8b44f7",
            hover_color="#732ce0",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._save_and_close,
        ).pack(side="bottom", fill="x", padx=14, pady=14)

    def _next_vod_frame(self) -> None:
        self._vod_sample_index += 1
        self.status_lbl.configure(text=f"⏳ Загрузка другого кадра @{self.channel}...", text_color="#fbc531")
        self._load_frame_async()

    def _load_frame_async(self) -> None:
        threading.Thread(target=self._fetch_frame_worker, daemon=True).start()

    def _fetch_frame_worker(self) -> None:
        buf_dir = self.base_dir / self.cfg.get("paths", {}).get("buffer_dir", "buffer")
        ch_buf = buf_dir / self.channel
        tmp_dir = buf_dir / "_cam_editor"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts_path = tmp_dir / f"edit_{self.channel}.ts"
        jpg_path = tmp_dir / f"edit_{self.channel}.jpg"

        try:
            # 1. Если сейчас идет запись живого эфира этого канала — берем свежий сегмент из буфера
            live_segs = sorted(ch_buf.glob("seg_*.ts")) if ch_buf.exists() else []
            if len(live_segs) >= 2:
                ts_path = live_segs[-2]
            else:
                # 2. Иначе берем кадр из последнего VOD канала на Twitch
                trainer = AutoTrainer(self.base_dir, self.cfg)
                info = trainer._fetch_channel_info(self.channel)
                vids = info.get("videos", [])
                if not vids:
                    self.after(
                        0,
                        lambda: self.status_lbl.configure(
                            text=f"⚠️ У @{self.channel} нет доступных VOD. Выделите область ориентировочно.",
                            text_color="#fbc531",
                        ),
                    )
                    return

                vod = vids[0]
                dur = max(600, int(vod.get("lengthSeconds") or 3600))
                fractions = [0.45, 0.25, 0.65, 0.80, 0.15]
                frac = fractions[self._vod_sample_index % len(fractions)]
                seek_sec = int(dur * frac)

                res = subprocess.run(
                    [*get_streamlink_cmd(), "--stream-url", f"https://www.twitch.tv/videos/{vod['id']}", "best"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    creationflags=WIN_NO_WINDOW,
                )
                m3u8 = res.stdout.strip()
                if not m3u8.startswith("http"):
                    return

                subprocess.run(
                    [
                        get_ffmpeg_path(),
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(seek_sec),
                        "-i",
                        m3u8,
                        "-t",
                        "3",
                        "-an",
                        "-c:v",
                        "copy",
                        "-f",
                        "mpegts",
                        str(ts_path),
                    ],
                    capture_output=True,
                    timeout=25,
                    creationflags=WIN_NO_WINDOW,
                )

            if not ts_path.exists():
                return

            self._Sample_ts_path = ts_path
            subprocess.run(
                [
                    get_ffmpeg_path(),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(ts_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=1280:720",
                    str(jpg_path),
                ],
                capture_output=True,
                timeout=15,
                creationflags=WIN_NO_WINDOW,
            )

            if jpg_path.exists():
                pil_img = Image.open(jpg_path).convert("RGB")
                self._pil_frame = pil_img
                self.after(0, self._on_frame_loaded)
        except Exception as exc:
            self.after(0, lambda e=exc: self.status_lbl.configure(text=f"Ошибка загрузки кадра: {e!r}"))

    def _on_frame_loaded(self) -> None:
        self.status_lbl.configure(
            text=f"✅ Кадр @{self.channel} загружен! Передвиньте зеленую рамку на вебкамеру и нажмите «Сохранить».",
            text_color="#00e676",
        )
        self._redraw_canvas()

    def _auto_detect_on_current(self) -> None:
        if self._Sample_ts_path and self._Sample_ts_path.exists():
            cam = self.detector.detect_webcam_for_video(self._Sample_ts_path, channel=self.channel)
            if cam and not cam.is_fullscreen_face:
                self.cx, self.cy, self.cw, self.ch = cam.x, cam.y, cam.w, cam.h
                self.size_slider.set(self.cw)
                self._redraw_canvas()
                self.status_lbl.configure(
                    text=f"🤖 ИИ нашел вебкамеру ({cam.preset_name})! При желании подправьте мышкой.",
                    text_color="#00f2fe",
                )
            else:
                messagebox.showinfo("ИИ-поиск", "На этом кадре угловая вебка не обнаружена. Попробуйте «Другой кадр» или выделите вручную.")

    def _on_size_slider(self, val: float) -> None:
        new_w = float(val)
        req_w_over_h = TARGET_ASPECT * (9.0 / 16.0)
        new_h = new_w / req_w_over_h
        center_x = self.cx + self.cw / 2.0
        center_y = self.cy + self.ch / 2.0
        self.cw = max(0.10, min(0.60, new_w))
        self.ch = max(0.10, min(0.60, new_h))
        self.cx = max(0.0, min(1.0 - self.cw, center_x - self.cw / 2.0))
        self.cy = max(0.0, min(1.0 - self.ch, center_y - self.ch / 2.0))
        self._redraw_canvas()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        delta = 0.015 if event.delta > 0 else -0.015
        new_w = max(0.12, min(0.48, self.cw + delta))
        self.size_slider.set(new_w)
        self._on_size_slider(new_w)

    def _on_mouse_down(self, event: tk.Event) -> None:
        x0 = self.cx * CANVAS_W
        y0 = self.cy * CANVAS_H
        x1 = (self.cx + self.cw) * CANVAS_W
        y1 = (self.cy + self.ch) * CANVAS_H

        self._drag_start_xy = (event.x, event.y)
        self._drag_start_box = (self.cx, self.cy, self.cw, self.ch)

        # Если кликнули около правого нижнего угла зеленой рамки — меняем размер
        if abs(event.x - x1) <= 18 and abs(event.y - y1) <= 18:
            self._drag_mode = "resize"
        elif x0 <= event.x <= x1 and y0 <= event.y <= y1:
            self._drag_mode = "move"
        else:
            # Клик вне рамки — центрируем рамку в точке клика и двигаем
            self.cx = max(0.0, min(1.0 - self.cw, (event.x / CANVAS_W) - self.cw / 2.0))
            self.cy = max(0.0, min(1.0 - self.ch, (event.y / CANVAS_H) - self.ch / 2.0))
            self._drag_start_box = (self.cx, self.cy, self.cw, self.ch)
            self._drag_mode = "move"
            self._redraw_canvas()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self._drag_mode:
            return
        dx = (event.x - self._drag_start_xy[0]) / CANVAS_W
        dy = (event.y - self._drag_start_xy[1]) / CANVAS_H
        sx, sy, sw, sh = self._drag_start_box

        if self._drag_mode == "move":
            self.cx = max(0.0, min(1.0 - sw, sx + dx))
            self.cy = max(0.0, min(1.0 - sh, sy + dy))
        elif self._drag_mode == "resize":
            req_w_over_h = TARGET_ASPECT * (9.0 / 16.0)
            new_w = max(0.12, min(0.55, min(1.0 - sx, sw + dx)))
            new_h = new_w / req_w_over_h
            if sy + new_h <= 1.0:
                self.cw = new_w
                self.ch = new_h
                self.size_slider.set(self.cw)

        self._redraw_canvas()

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._drag_mode = None

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self._pil_frame is not None:
            resized = self._pil_frame.resize((CANVAS_W, CANVAS_H), Image.Resampling.BILINEAR)
            self._tk_canvas_img = ImageTk.PhotoImage(resized)
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_canvas_img)

            # Обновляем превью вырезанной вебки справа
            fw, fh = self._pil_frame.size
            box = (
                int(self.cx * fw),
                int(self.cy * fh),
                int(min(fw, (self.cx + self.cw) * fw)),
                int(min(fh, (self.cy + self.ch) * fh)),
            )
            if box[2] > box[0] + 10 and box[3] > box[1] + 10:
                crop_pil = self._pil_frame.crop(box).resize((PREVIEW_W, PREVIEW_H), Image.Resampling.BILINEAR)
                self._ctk_preview_img = ctk.CTkImage(
                    light_image=crop_pil, dark_image=crop_pil, size=(PREVIEW_W, PREVIEW_H)
                )
                self.preview_lbl.configure(text="", image=self._ctk_preview_img)

        x0 = int(self.cx * CANVAS_W)
        y0 = int(self.cy * CANVAS_H)
        x1 = int((self.cx + self.cw) * CANVAS_W)
        y1 = int((self.cy + self.ch) * CANVAS_H)

        # Зеленая рамка вебкамеры + уголок для изменения размера
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00e676", width=3)
        self.canvas.create_rectangle(x1 - 8, y1 - 8, x1 + 4, y1 + 4, fill="#00e676", outline="#000000")
        self.canvas.create_text(
            x0 + 6,
            max(12, y0 - 10),
            anchor="w",
            text=f"ВЕБКА @{self.channel.upper()}",
            fill="#00e676",
            font=("Segoe UI", 9, "bold"),
        )

        self.coords_lbl.configure(
            text=f"X: {self.cx:.2f}  Y: {self.cy:.2f}\nШирина: {self.cw:.2f}  Высота: {self.ch:.2f}"
        )

    def _save_and_close(self) -> None:
        crop_dict = WebcamCrop(
            x=round(self.cx, 4),
            y=round(self.cy, 4),
            w=round(self.cw, 4),
            h=round(self.ch, 4),
            confidence=1.0,
            preset_name="manual_custom",
        ).to_dict()
        self.on_save_crop(self.channel, crop_dict)
        self.destroy()

    def destroy(self) -> None:
        tmp_dir = self.base_dir / self.cfg.get("paths", {}).get("buffer_dir", "buffer") / "_cam_editor"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        super().destroy()
