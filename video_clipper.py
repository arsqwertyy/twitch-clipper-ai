import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from audio_analyzer import BannerPlacementResult, SmartTrimResult, compute_smart_banner_time
from face_detector import FaceWebcamDetector, WebcamCrop
from ffmpeg_utils import WIN_NO_WINDOW, check_nvenc_support, get_ffmpeg_path
from subtitle_burner import SubtitleGenerator


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class VideoClipper:
    """
    Склеивает .ts сегменты из кольцевого буфера, автоматически находит вебкамеру с лицом стримера
    (через нейросеть YuNet), генерирует субтитры Faster-Whisper и собирает вертикальный TikTok-ролик 9:16:
    [Сверху: крупно вебка стримера] + [По центру/снизу: экран самой трансляции] + [Баннер] + [Динамические субтитры].
    """

    def __init__(
        self,
        buffer_dir: Path,
        clips_dir: Path,
        video_cfg: dict,
        sub_generator: SubtitleGenerator,
    ):
        self.buffer_dir = buffer_dir.resolve()
        self.clips_dir = clips_dir.resolve()
        self.base_dir = Path(__file__).resolve().parent
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.video_cfg = video_cfg
        self.sub_generator = sub_generator
        self.face_detector = FaceWebcamDetector()

        self.mode: str = str(video_cfg.get("mode", "tiktok"))
        self.webcam_mode: str = str(video_cfg.get("webcam_position", "auto"))
        self.width: int = int(video_cfg.get("width", 1080))
        self.height: int = int(video_cfg.get("height", 1920))
        self.center_zoom: float = max(1.0, float(video_cfg.get("center_zoom", 1.35)))
        self.blur_strength: int = max(4, int(video_cfg.get("blur_strength", 25)))
        self.climax_zoom: bool = bool(video_cfg.get("climax_zoom", True))
        self.banner_cfg: dict = dict(video_cfg.get("banner") or {})
        self._banner_meta_cache: dict[tuple[str, int], dict] = {}
        self.last_skip_reason: str = ""
        self.last_banner_pause_info: str = ""
        self.last_transcript_text: str = ""

        self.ffmpeg_exe = get_ffmpeg_path()
        want_nvenc = bool(video_cfg.get("use_nvenc", True))
        self.nvenc_available = want_nvenc and check_nvenc_support(self.ffmpeg_exe)

    def update_banner_cfg(self, banner_cfg: dict) -> None:
        self.banner_cfg = dict(banner_cfg or {})

    def _resolve_banner_path(self) -> Optional[Path]:
        if not self.banner_cfg.get("enabled", False):
            return None
        raw_path = str(self.banner_cfg.get("path", "") or "").strip()
        if not raw_path:
            return None
        p = Path(raw_path)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return p if p.exists() and p.is_file() else None

    def _probe_media_meta(self, path: Path) -> tuple[float, bool, bool, int]:
        """Возвращает (duration_sec, has_audio, has_alpha, fps_int) через ffmpeg -i."""
        try:
            r = subprocess.run(
                [self.ffmpeg_exe, "-hide_banner", "-i", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=WIN_NO_WINDOW,
            )
            txt = r.stderr or ""
        except Exception:
            return 0.0, False, False, 30

        dur = 0.0
        m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
        if m_dur:
            dur = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))
        has_audio = bool(re.search(r"Stream #\d+:\d+.*Audio:", txt))
        has_alpha = bool(re.search(r"Stream #\d+:\d+.*Video:.*\b(rgba|argb|bgra|abgr|yuva\w*|pal8)\b", txt))
        fps = 30
        m_fps = re.search(r"Stream #\d+:\d+.*Video:.*?(\d+(?:\.\d+)?)\s*fps", txt)
        if m_fps:
            try:
                fps_val = float(m_fps.group(1))
                fps = 60 if fps_val >= 45.0 else 30
            except ValueError:
                fps = 30
        return dur, has_audio, has_alpha, fps

    def _inspect_banner_asset(self, banner_path: Path) -> dict:
        """Кэширует метаданные баннера и автоопределяет цвет хромакея по углам кадра."""
        mtime = int(banner_path.stat().st_mtime)
        cache_key = (str(banner_path), mtime)
        if cache_key in self._banner_meta_cache:
            return self._banner_meta_cache[cache_key]

        is_image = banner_path.suffix.lower() in IMAGE_EXTS
        dur, has_audio, has_alpha, _fps = self._probe_media_meta(banner_path)
        if is_image or dur <= 0.0:
            dur = max(1.0, float(self.banner_cfg.get("static_duration", 3.5)))
            has_audio = False
        elif dur > 1.5:
            # Аккуратно подрезаем ~0.12с пустого хвоста в конце анимации исчезновения
            dur = round(dur - 0.12, 2)

        detected_hex: Optional[str] = None
        if not has_alpha:
            tmp_png = self.buffer_dir / f"_ban_probe_{mtime}.png"
            try:
                if is_image:
                    img = Image.open(banner_path)
                    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
                        has_alpha = True
                    arr = np.array(img.convert("RGB"))
                else:
                    probe_t = min(0.25, max(0.0, dur * 0.1))
                    subprocess.run(
                        [
                            self.ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", f"{probe_t:.2f}", "-i", str(banner_path),
                            "-frames:v", "1", str(tmp_png),
                        ],
                        capture_output=True,
                        timeout=15,
                        creationflags=WIN_NO_WINDOW,
                    )
                    arr = np.array(Image.open(tmp_png).convert("RGB")) if tmp_png.exists() else None

                if arr is not None and not has_alpha and arr.shape[0] > 20 and arr.shape[1] > 20:
                    corners = np.array([arr[5, 5], arr[5, -6], arr[-6, 5], arr[-6, -6]], dtype=np.float32)
                    med = np.median(corners, axis=0)
                    max_dist = float(np.max(np.linalg.norm(corners - med, axis=1)))
                    if max_dist <= 28.0:
                        r_c, g_c, b_c = (int(round(med[0])), int(round(med[1])), int(round(med[2])))
                        detected_hex = f"0x{r_c:02x}{g_c:02x}{b_c:02x}"
            except Exception:
                pass
            finally:
                tmp_png.unlink(missing_ok=True)

        info = {
            "is_image": is_image,
            "duration": dur,
            "has_audio": has_audio,
            "has_alpha": has_alpha,
            "chromakey_hex": detected_hex,
        }
        self._banner_meta_cache[cache_key] = info
        return info

    def create_clip_from_segments(
        self,
        segments: list[Path],
        channel: str,
        peak_score: float,
        fallback_webcam_crop: Optional[dict] = None,
        max_duration: Optional[float] = None,
        skip_silent_afk: bool = False,
    ) -> tuple[Optional[Path], int, float]:
        """
        Полный цикл создания одного клипа:
        1. Склеивание .ts сегментов во временный исходник
        2. Авто-поиск лица/вебкамеры стримера на кадрах клипа (YuNet ONNX)
        3. Распознавание речи через Faster-Whisper и генерация .ass субтитров
        4. Финальный рендер в формат TikTok 9:16 (Вебка сверху + Экран снизу + Субтитры)
        """
        t0 = time.perf_counter()
        if not segments:
            return None, 0, 0.0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{channel}_{stamp}_score{int(peak_score)}"
        self.last_skip_reason = ""
        self.last_banner_pause_info = ""
        self.last_transcript_text = ""

        has_active_banner = self._resolve_banner_path() is not None

        concat_list_path = self.buffer_dir / f"concat_{job_id}.txt"
        raw_mp4_path = self.buffer_dir / f"raw_{job_id}.mp4"
        ass_path = self.buffer_dir / f"subs_{job_id}.ass"
        prefix = "с_баннером_" if has_active_banner else "клип_"
        final_mp4_path = self.clips_dir / f"{prefix}{job_id}.mp4"

        try:
            concat_lines = []
            for seg in segments:
                if seg.exists() and seg.stat().st_size > 1024:
                    safe_path = str(seg.resolve()).replace("\\", "/")
                    concat_lines.append(f"file '{safe_path}'")

            if not concat_lines:
                return None, 0, 0.0

            concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

            concat_cmd = [
                self.ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(raw_mp4_path),
            ]
            res = subprocess.run(
                concat_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                creationflags=WIN_NO_WINDOW,
            )
            if res.returncode != 0 or not raw_mp4_path.exists():
                print(f"[clipper] Ошибка склейки сегментов: {res.stderr.strip()[-500:]}")
                return None, 0, 0.0

            # 2. Авто-определение области вебкамеры со стримером
            cam_crop: Optional[WebcamCrop] = None
            if self.mode == "tiktok" and self.webcam_mode != "none":
                cam_crop = self.face_detector.detect_webcam_for_video(
                    video_path=raw_mp4_path,
                    target_aspect=self.width / 620.0,
                    position_mode=self.webcam_mode,
                    fallback_crop=fallback_webcam_crop,
                    channel=channel,
                )

            # 3. Генерируем .ass субтитры через Faster-Whisper и рассчитываем умную подрезку Smart Hook
            effective_max_dur = max_duration
            if has_active_banner and max_duration is not None and bool(self.banner_cfg.get("pause_video", True)):
                ban_p = self._resolve_banner_path()
                if ban_p is not None and max_duration >= 22.0:
                    ban_dur_est = float(self._inspect_banner_asset(ban_p)["duration"])
                    effective_max_dur = max(15.0, max_duration - ban_dur_est)

            word_count, trim_info = self.sub_generator.generate_ass_and_trim(
                raw_mp4_path, ass_path, max_duration=effective_max_dur
            )
            self.last_transcript_text = " ".join(
                str(w.word).strip() for w in (getattr(self.sub_generator, "last_words", None) or [])
            )

            # Отсеиваем короткие обрывки (< 18 сек) и пустые моменты без речи стримера (< 6 слов),
            # чтобы никогда не сохранять 7-секундные ролики или тишину, где ничего не происходит!
            if skip_silent_afk:
                if trim_info is not None:
                    trimmed_len = float(trim_info.trim_end - trim_info.trim_start)
                    if trimmed_len < 18.0:
                        self.last_skip_reason = f"пропущен слишком короткий фрагмент ({trimmed_len:.1f}с < 18с)"
                        return None, 0, time.perf_counter() - t0
                    if trim_info.is_silent_afk:
                        self.last_skip_reason = "пропущен пустой/AFK момент (тишина)"
                        return None, 0, time.perf_counter() - t0
                if self.sub_generator.enabled and word_count < 6:
                    self.last_skip_reason = f"пропущен пустой момент без речи стримера (слов: {word_count} < 6)"
                    return None, 0, time.perf_counter() - t0

            has_subs = word_count > 0 and ass_path.exists()

            # 4. Рендерим итоговое видео (с подрезкой по началу фразы и зумом на кульминации)
            self._render_final_video(
                raw_mp4_path=raw_mp4_path,
                ass_path=ass_path if has_subs else None,
                final_mp4_path=final_mp4_path,
                cam_crop=cam_crop,
                trim_info=trim_info,
                use_banner=has_active_banner,
            )

            elapsed = time.perf_counter() - t0
            return (final_mp4_path if final_mp4_path.exists() else None), word_count, elapsed

        finally:
            for temp_file in (concat_list_path, raw_mp4_path, ass_path):
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass

    @staticmethod
    def _pick_clean_pause_time(
        target_t: float,
        clip_dur: float,
        words: list,
        phrases: Optional[list] = None,
        trim_info: Optional[SmartTrimResult] = None,
        position_mode: str = "smart",
        pause_video: bool = True,
        ban_dur: float = 3.5,
    ) -> BannerPlacementResult:
        """Подбирает идеальную точку вставки баннера (интрига перед кульминацией + чистая граница фразы)."""
        return compute_smart_banner_time(
            clip_dur=clip_dur,
            position_mode=position_mode,
            words=words,
            phrases=phrases,
            trim_info=trim_info,
            pause_video=pause_video,
            ban_dur=ban_dur,
        )

    def _render_final_video(
        self,
        raw_mp4_path: Path,
        ass_path: Optional[Path],
        final_mp4_path: Path,
        cam_crop: Optional[WebcamCrop] = None,
        trim_info: Optional[SmartTrimResult] = None,
        use_banner: bool = True,
    ) -> None:
        """
        Собирает вертикальный кадр 1080x1920:
        - Если найдена угловая вебкамера (cam_crop и не fullscreen):
          Сверху (1080x620) крупно показывается лицо/вебка стримера,
          ниже (Y=626) — экран самой трансляции, внизу — динамические субтитры.
        - Если стример на весь экран (Just Chatting) или без вебки:
          Показывается центральный кроп трансляции на размытом фоне.
        """
        raw_mp4_path = raw_mp4_path.resolve()
        final_mp4_path = final_mp4_path.resolve()

        if self.mode == "tiktok":
            w, h = self.width, self.height
            small_w, small_h = max(180, w // 4), max(320, h // 4)
            sigma = max(4, self.blur_strength // 2)
            zoom = self.center_zoom

            use_split_cam = (cam_crop is not None) and (not cam_crop.is_fullscreen_face)

            if use_split_cam and cam_crop is not None:
                cam_h_px = int(round(h * 0.325))  # ~624px сверху под вебкамеру стримера
                cx = max(0.0, min(0.95, float(cam_crop.x)))
                cy = max(0.0, min(0.95, float(cam_crop.y)))
                cw = max(0.05, min(1.0 - cx, float(cam_crop.w)))
                ch = max(0.05, min(1.0 - cy, float(cam_crop.h)))
                max_fg_h = h - cam_h_px - 6

                # Если вебка слева, чуть сдвигаем кроп основного экрана вправо (и наоборот),
                # чтобы центр игры/видео был виден идеально и не дублировал угол с вебкой
                if zoom > 1.05:
                    if cx + cw / 2.0 < 0.35:
                        fg_x_expr = f"(iw-iw/{zoom:.2f})*0.82"
                    elif cx + cw / 2.0 > 0.65:
                        fg_x_expr = f"(iw-iw/{zoom:.2f})*0.18"
                    else:
                        fg_x_expr = f"(iw-iw/{zoom:.2f})*0.50"
                else:
                    fg_x_expr = "0"

                if self.climax_zoom and trim_info is not None:
                    peak_rel = max(0.8, trim_info.peak_audio_time - trim_info.trim_start)
                    t_z0 = max(0.4, peak_rel - 0.35)
                    t_z1 = t_z0 + 3.2
                    zw = int(round(w * 1.18))
                    zh = int(round(cam_h_px * 1.18))
                    cam_branch = (
                        f"[cam_in]crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},"
                        f"scale={w}:{cam_h_px}:force_original_aspect_ratio=increase,"
                        f"crop={w}:{cam_h_px},split=2[cam_base][cam_z_in];"
                        f"[cam_z_in]scale={zw}:{zh},crop={w}:{cam_h_px}:(iw-{w})/2:(ih-{cam_h_px})*0.42[cam_z];"
                        f"[cam_base][cam_z]overlay=0:0:enable='between(t\\,{t_z0:.2f}\\,{t_z1:.2f})'[cam];"
                    )
                else:
                    cam_branch = (
                        f"[cam_in]crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},"
                        f"scale={w}:{cam_h_px}:force_original_aspect_ratio=increase,"
                        f"crop={w}:{cam_h_px}[cam];"
                    )

                base_filter = (
                    f"[0:v]split=3[bg_in][fg_in][cam_in];"
                    f"[bg_in]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,"
                    f"crop={small_w}:{small_h},gblur=sigma={sigma},"
                    f"scale={w}:{h},eq=brightness=-0.10:saturation=1.1[bg];"
                    f"{cam_branch}"
                    f"[fg_in]crop=iw/{zoom:.2f}:ih:{fg_x_expr}:0,scale={w}:-2,crop={w}:min(ih\\,{max_fg_h})[fg];"
                    f"[bg][cam]overlay=0:0[tmp1];"
                    f"[tmp1][fg]overlay=(W-w)/2:{cam_h_px + 6},"
                    f"drawbox=x=0:y={cam_h_px}:w={w}:h=6:color=#18181b@1:t=fill[base_v]"
                )
            else:
                base_filter = (
                    f"[0:v]split=2[bg_in][fg_in];"
                    f"[bg_in]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,"
                    f"crop={small_w}:{small_h},gblur=sigma={sigma},"
                    f"scale={w}:{h},eq=brightness=-0.08:saturation=1.1[bg];"
                    f"[fg_in]crop=iw/{zoom:.2f}:ih,scale={w}:-2[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base_v]"
                )
        else:
            w, h = 1920, 1080
            base_filter = "[0:v]scale=1920:1080[base_v]"

        # Подготовка рекламного баннера (видео с хромакеем / прозрачностью или PNG) + звук баннера
        banner_path = self._resolve_banner_path() if use_banner else None
        extra_inputs: list[str] = []
        audio_map = "0:a?"
        filter_parts = [base_filter]
        temp_ban_ass_path: Optional[Path] = None

        if banner_path is not None:
            raw_dur, main_has_audio, _, main_fps = self._probe_media_meta(raw_mp4_path)
            if trim_info is not None and trim_info.trim_end > trim_info.trim_start + 3.0:
                clip_dur = trim_info.trim_end - trim_info.trim_start
            else:
                clip_dur = raw_dur if raw_dur > 0 else 30.0

            ban_info = self._inspect_banner_asset(banner_path)
            ban_dur = float(ban_info["duration"])
            is_img = bool(ban_info["is_image"])
            pause_video = bool(self.banner_cfg.get("pause_video", True))

            width_ratio = max(0.35, min(1.0, float(self.banner_cfg.get("width_ratio", 0.88))))
            pos_y = max(0.05, min(0.90, float(self.banner_cfg.get("pos_y", 0.48))))
            ban_w = max(120, int(round((w * width_ratio) / 2.0)) * 2)

            use_ck = bool(self.banner_cfg.get("chromakey", True))
            ck_cfg_color = str(self.banner_cfg.get("chromakey_color", "auto") or "auto").strip()
            ck_color: Optional[str] = None
            if use_ck and not ban_info["has_alpha"]:
                if ck_cfg_color.lower() != "auto":
                    ck_color = ck_cfg_color if ck_cfg_color.startswith("0x") else f"0x{ck_cfg_color.lstrip('#')}"
                else:
                    ck_color = ban_info["chromakey_hex"]

            if ck_color:
                sim = max(0.02, min(0.5, float(self.banner_cfg.get("chromakey_similarity", 0.14))))
                blend = max(0.0, min(0.3, float(self.banner_cfg.get("chromakey_blend", 0.06))))
                key_chain = f"format=rgba,chromakey={ck_color}:{sim:.2f}:{blend:.2f},"
            else:
                key_chain = "format=rgba,"

            if is_img:
                extra_inputs.extend(["-loop", "1", "-t", f"{ban_dur:.2f}", "-i", str(banner_path)])
            else:
                extra_inputs.extend(["-i", str(banner_path)])

            ban_vol = max(0.0, min(2.0, float(self.banner_cfg.get("volume", 1.0))))
            pos_time = str(self.banner_cfg.get("position_time", "smart"))
            words_list = getattr(self.sub_generator, "last_words", None) or []
            phrases_list = getattr(self.sub_generator, "last_phrases", None) or []

            if pause_video and clip_dur >= 4.0:
                # РЕЖИМ КАК В ПРИМЕРЕ (IMG_0417.MP4) + УМНАЯ ИНТРИГА:
                # В момент появления баннера (строго на границе фразы перед кульминацией)
                # основное видео и звук СТАВЯТСЯ НА ПАУЗУ, играет ТОЛЬКО анимация и озвучка баннера,
                # а затем клип продолжается ровно с того же кадра — прямо в развязку!
                placement = self._pick_clean_pause_time(
                    target_t=clip_dur * 0.50,
                    clip_dur=clip_dur,
                    words=words_list,
                    phrases=phrases_list,
                    trim_info=trim_info,
                    position_mode=pos_time,
                    pause_video=True,
                    ban_dur=ban_dur,
                )
                t_pause = placement.pause_time
                self.last_banner_pause_info = f"{t_pause:.1f}с ({placement.reason})"
                frame_step = max(0.03, 1.5 / float(main_fps))

                # Готовим чистую версию субтитров под стоп-кадр (без зависших за баннером слов и спойлеров)
                effective_ass_path = ass_path
                if ass_path is not None and hasattr(self.sub_generator, "write_ass_for_pause"):
                    cand_ass = ass_path.with_name(f"{ass_path.stem}_ban{ass_path.suffix}")
                    try:
                        if self.sub_generator.write_ass_for_pause(cand_ass, t_pause) and cand_ass.exists():
                            effective_ass_path = cand_ass
                            temp_ban_ass_path = cand_ass
                    except Exception:
                        effective_ass_path = ass_path

                # Сначала накладываем субтитры на основное видео (пока таймкоды 1-в-1 совпадают с речью)
                if effective_ass_path is not None:
                    filter_parts.append(
                        f"[base_v]ass={effective_ass_path.name},setsar=1,fps={main_fps},format=yuv420p[main_v]"
                    )
                else:
                    filter_parts.append(f"[base_v]setsar=1,fps={main_fps},format=yuv420p[main_v]")

                # Делим основное видео на 3 части: [до паузы] + [стоп-кадр с баннером] + [продолжение]
                filter_parts.append("[main_v]split=3[v_p1_in][v_frz_in][v_p2_in]")
                filter_parts.append(f"[v_p1_in]trim=start=0:end={t_pause:.3f},setpts=PTS-STARTPTS[v_p1]")
                filter_parts.append(
                    f"[v_frz_in]trim=start={t_pause:.3f}:end={min(clip_dur, t_pause + frame_step):.3f},"
                    f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={ban_dur:.3f},"
                    f"trim=duration={ban_dur:.3f},setpts=PTS-STARTPTS[v_frz_bg]"
                )
                filter_parts.append(f"[v_p2_in]trim=start={t_pause:.3f},setpts=PTS-STARTPTS[v_p2]")

                filter_parts.append(
                    f"[1:v]fps={main_fps},{key_chain}scale={ban_w}:-2,"
                    f"trim=duration={ban_dur:.3f},setpts=PTS-STARTPTS[ban_v]"
                )
                filter_parts.append(
                    f"[v_frz_bg][ban_v]overlay=(W-w)/2:(H-h)*{pos_y:.3f}:"
                    f"eof_action=pass:shortest=1,setsar=1,format=yuv420p[v_mid]"
                )

                if main_has_audio:
                    filter_parts.append("[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[a_p1_in][a_p2_in]")
                    filter_parts.append(f"[a_p1_in]atrim=start=0:end={t_pause:.3f},asetpts=PTS-STARTPTS[a_p1]")
                    if ban_info["has_audio"] and ban_vol > 0.01:
                        filter_parts.append(
                            f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={ban_vol:.2f},"
                            f"apad=whole_dur={ban_dur:.3f},atrim=duration={ban_dur:.3f},asetpts=PTS-STARTPTS[a_mid]"
                        )
                    else:
                        filter_parts.append(
                            f"anullsrc=r=48000:cl=stereo,atrim=duration={ban_dur:.3f},asetpts=PTS-STARTPTS[a_mid]"
                        )
                    filter_parts.append(f"[a_p2_in]atrim=start={t_pause:.3f},asetpts=PTS-STARTPTS[a_p2]")
                    filter_parts.append("[v_p1][a_p1][v_mid][a_mid][v_p2][a_p2]concat=n=3:v=1:a=1[outv][outa]")
                    audio_map = "[outa]"
                else:
                    filter_parts.append("[v_p1][v_mid][v_p2]concat=n=3:v=1:a=0[outv]")
            else:
                # Режим без паузы (поверх идущего видео)
                placement = self._pick_clean_pause_time(
                    target_t=clip_dur * 0.50,
                    clip_dur=clip_dur,
                    words=words_list,
                    phrases=phrases_list,
                    trim_info=trim_info,
                    position_mode=pos_time,
                    pause_video=False,
                    ban_dur=ban_dur,
                )
                t_ban_start = max(1.0, min(max(1.0, clip_dur - ban_dur - 0.5), placement.pause_time))
                self.last_banner_pause_info = f"{t_ban_start:.1f}с ({placement.reason})"
                t_ban_end = min(clip_dur, t_ban_start + max(0.3, ban_dur))

                filter_parts.append(
                    f"[1:v]{key_chain}scale={ban_w}:-2,setpts=PTS-STARTPTS+{t_ban_start:.2f}/TB[ban_v]"
                )
                filter_parts.append(
                    f"[base_v][ban_v]overlay=(W-w)/2:(H-h)*{pos_y:.3f}:"
                    f"eof_action=pass:enable='between(t\\,{t_ban_start:.2f}\\,{t_ban_end:.2f})'[with_ban]"
                )
                if ass_path is not None:
                    filter_parts.append(f"[with_ban]ass={ass_path.name},setsar=1,format=yuv420p[outv]")
                else:
                    filter_parts.append("[with_ban]setsar=1,format=yuv420p[outv]")

                if ban_info["has_audio"] and main_has_audio and ban_vol > 0.01:
                    delay_ms = int(round(t_ban_start * 1000))
                    filter_parts.append(
                        f"[0:a]aresample=48000,aformat=channel_layouts=stereo[main_a];"
                        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,"
                        f"volume={ban_vol:.2f},adelay={delay_ms}|{delay_ms}[ban_a];"
                        f"[main_a][ban_a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[outa]"
                    )
                    audio_map = "[outa]"
        else:
            if ass_path is not None:
                filter_parts.append(f"[base_v]ass={ass_path.name},setsar=1,format=yuv420p[outv]")
            else:
                filter_parts.append("[base_v]setsar=1,format=yuv420p[outv]")

        filter_complex = ";".join(filter_parts)

        if self.nvenc_available:
            video_codec_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-b:v",
                "6M",
                "-maxrate",
                "8M",
            ]
        else:
            video_codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
            ]

        seek_args: list[str] = []
        if trim_info is not None and trim_info.trim_end > trim_info.trim_start + 3.0:
            if trim_info.trim_start > 0.05:
                seek_args.extend(["-ss", f"{trim_info.trim_start:.2f}"])
            seek_args.extend(["-to", f"{trim_info.trim_end:.2f}"])

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *seek_args,
            "-i",
            str(raw_mp4_path),
            *extra_inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            audio_map,
            *video_codec_args,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(final_mp4_path),
        ]

        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.buffer_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                creationflags=WIN_NO_WINDOW,
            )
        finally:
            if temp_ban_ass_path is not None and temp_ban_ass_path.exists():
                try:
                    temp_ban_ass_path.unlink()
                except OSError:
                    pass
        if res.returncode != 0 and self.nvenc_available:
            print(f"[clipper] NVENC ошибка, переключаюсь на libx264: {res.stderr.strip()[-300:]}")
            self.nvenc_available = False
            self._render_final_video(
                raw_mp4_path,
                ass_path,
                final_mp4_path,
                cam_crop=cam_crop,
                trim_info=trim_info,
                use_banner=use_banner,
            )
        elif res.returncode != 0:
            raise RuntimeError(f"Ошибка рендера FFmpeg: {res.stderr.strip()[-800:]}")
