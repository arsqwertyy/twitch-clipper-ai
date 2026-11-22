import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path


@dataclass
class WebcamCrop:
    """Нормализованные координаты (0.0 .. 1.0) области вебкамеры на исходном 16:9 кадре."""
    x: float
    y: float
    w: float
    h: float
    is_fullscreen_face: bool = False
    confidence: float = 0.0
    preset_name: str = "auto"

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "w": round(self.w, 4),
            "h": round(self.h, 4),
            "preset_name": self.preset_name,
        }


WEBCAM_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.24, 0.30),
    "middle_left": (0.0, 0.20, 0.24, 0.30),
    "bottom_left": (0.0, 0.66, 0.24, 0.32),
    "top_right": (0.76, 0.0, 0.24, 0.30),
    "middle_right": (0.76, 0.20, 0.24, 0.30),
    "bottom_right": (0.76, 0.66, 0.24, 0.32),
}


class FaceWebcamDetector:
    """
    Универсальный ИИ-детектор вебкамеры для любых стримеров:
    1. Сканирует левую и правую половины кадра в высоком разрешении (720p) через нейросеть YuNet ONNX,
       чтобы находить даже маленькие игровые вебкамеры (CS2, Dota 2, PUBG, GTA).
    2. Определяет точные границы прямоугольной рамки вебкамеры по усредненным градиентам кадров.
    3. Совмещает найденное лицо с границами рамки вебки (и симметрией до края экрана), чтобы
       в верхний блок никогда не попадал кусок игры или видео за спиной стримера.
    4. Кэширует найденную вебкамеру для каждого канала, чтобы при резких движениях, смехе руками
       у лица или выходе из-за стола сплит-экран не пропадал.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "yunet.onnx"
        self.model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        self._channel_cache: dict[str, WebcamCrop] = {}

    def _get_session(self) -> Optional[ort.InferenceSession]:
        if self._session is not None:
            return self._session
        if not self.model_path.exists():
            return None
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            return self._session
        except Exception:
            return None

    def detect_webcam_for_video(
        self,
        video_path: Path,
        target_aspect: float = 1080.0 / 624.0,
        position_mode: str = "auto",
        fallback_crop: Optional[dict] = None,
        channel: str = "",
    ) -> Optional[WebcamCrop]:
        if position_mode in WEBCAM_PRESETS:
            px, py, pw, ph = WEBCAM_PRESETS[position_mode]
            return WebcamCrop(x=px, y=py, w=pw, h=ph, confidence=1.0, preset_name=position_mode)

        if position_mode == "none":
            return None

        clean_ch = channel.replace("test_", "").strip().lower()
        frames_720p = self._extract_sample_frames_720p(video_path, num_frames=6)
        if not frames_720p:
            return self._get_cached_or_fallback(clean_ch, fallback_crop)

        # 1. Ищем неподвижную прямоугольную рамку вебки у краев экрана (на уменьшенных до 640x360 копиях)
        frames_360p = [f[::2, ::2, :] for f in frames_720p]
        rect_crop = self._detect_webcam_rectangle(frames_360p)

        # 2. Запускаем YuNet в двух режимах:
        #    а) Высокое разрешение по краям (левая и правая трети из 1280x720) — видит даже мелкие вебки
        #    б) Полный кадр 16:9 — видит крупные лица
        sess = self._get_session()
        corner_faces: list[tuple[float, float, float, float, float]] = []
        center_faces: list[tuple[float, float, float, float, float]] = []

        if sess is not None:
            for f720, f360 in zip(frames_720p, frames_360p):
                # Поиск на полном кадре 640x360
                faces_full = self._detect_faces_16x9(sess, f360, score_thresh=0.50)
                # Поиск в высоком разрешении на левом и правом краях (ширина 480px из 1280px = 37.5% с каждого края)
                faces_sides = self._detect_faces_highres_sides(sess, f720, score_thresh=0.54)

                all_faces = self._merge_detections(faces_full + faces_sides)
                for cx, cy, fw, fh, score in all_faces:
                    is_periphery = (cx < 0.36 or cx > 0.64 or cy < 0.34 or cy > 0.66)
                    if is_periphery and fh <= 0.24:
                        corner_faces.append((cx, cy, fw, fh, score))
                    else:
                        center_faces.append((cx, cy, fw, fh, score))

        # Случай А: Найдено лицо стримера в углу/сбоку экрана
        if len(corner_faces) >= 1:
            cluster = self._pick_consistent_cluster(corner_faces)
            if cluster:
                med_cx = float(np.median([f[0] for f in cluster]))
                med_cy = float(np.median([f[1] for f in cluster]))
                med_fh = float(np.median([f[3] for f in cluster]))
                med_score = float(np.max([f[4] for f in cluster]))

                # Если также найдена четкая прямоугольная рамка вебки, и лицо находится внутри нее —
                # вписываем кроп строго внутрь рамки вебки с центром на лице стримера!
                if rect_crop is not None and self._point_in_crop(med_cx, med_cy, rect_crop):
                    result = self._fit_face_inside_rect(med_cx, med_cy, med_fh, rect_crop, target_aspect, med_score)
                else:
                    result = self._box_around_face(
                        cx=med_cx,
                        cy=med_cy,
                        fh=med_fh,
                        target_aspect=target_aspect,
                        confidence=med_score,
                        is_fullscreen=False,
                    )
                if clean_ch:
                    self._channel_cache[clean_ch] = result
                return result

        # Случай Б: Нет углового лица, а по центру крупное лицо (IRL / Just Chatting на весь экран)
        if len(center_faces) >= 2:
            cluster = self._pick_consistent_cluster(center_faces)
            if cluster:
                med_fh = float(np.median([f[3] for f in cluster]))
                med_score = float(np.max([f[4] for f in cluster]))
                if med_fh >= 0.16:
                    return WebcamCrop(
                        x=0.0,
                        y=0.0,
                        w=1.0,
                        h=1.0,
                        is_fullscreen_face=True,
                        confidence=med_score,
                        preset_name="fullscreen_face",
                    )

        # Случай В: Лицо в углу в эту секунду не видно (стример отошел/закрыл лицо руками/темно),
        # но видна четкая прямоугольная рамка вебкамеры в углу!
        if rect_crop is not None:
            result = self._adjust_rect_aspect(rect_crop, target_aspect)
            if clean_ch and clean_ch not in self._channel_cache:
                self._channel_cache[clean_ch] = result
            return result

        # Случай Г: Используем ранее запомненную вебкамеру этого стримера (из кэша или профиля)
        cached = self._get_cached_or_fallback(clean_ch, fallback_crop)
        if cached is not None:
            return cached

        return None

    def _get_cached_or_fallback(
        self, clean_ch: str, fallback_crop: Optional[dict]
    ) -> Optional[WebcamCrop]:
        if clean_ch and clean_ch in self._channel_cache:
            return self._channel_cache[clean_ch]
        if fallback_crop and "x" in fallback_crop:
            return WebcamCrop(
                x=float(fallback_crop["x"]),
                y=float(fallback_crop["y"]),
                w=float(fallback_crop["w"]),
                h=float(fallback_crop["h"]),
                confidence=0.6,
                preset_name=str(fallback_crop.get("preset_name", "learned_profile")),
            )
        return None

    @staticmethod
    def _point_in_crop(cx: float, cy: float, crop: WebcamCrop) -> bool:
        return (crop.x - 0.03) <= cx <= (crop.x + crop.w + 0.03) and (
            crop.y - 0.03
        ) <= cy <= (crop.y + crop.h + 0.03)

    @staticmethod
    def _fit_face_inside_rect(
        cx: float,
        cy: float,
        fh: float,
        rect: WebcamCrop,
        target_aspect: float,
        confidence: float,
    ) -> WebcamCrop:
        """
        Вырезает область с пропорциями верхнего блока TikTok (target_aspect) СТРОГО внутри
        найденной прямоугольной рамки вебкамеры rect, центрируясь на лице стримера (cx, cy).
        """
        # Переводим ширину/высоту рамки в пиксельные пропорции 16:9
        # Нам нужно: (w_norm * 16) / (h_norm * 9) == target_aspect
        req_w_over_h = target_aspect * (9.0 / 16.0)

        w = rect.w
        h = w / req_w_over_h
        if h > rect.h:
            h = rect.h
            w = min(rect.w, h * req_w_over_h)

        # Центрируем на лице стримера, но не выходим за границы рамки rect!
        target_cy = cy + fh * 0.10
        x0 = float(np.clip(cx - w / 2.0, rect.x, rect.x + rect.w - w))
        y0 = float(np.clip(target_cy - h / 2.0, rect.y, rect.y + rect.h - h))

        return WebcamCrop(
            x=round(x0, 4),
            y=round(y0, 4),
            w=round(w, 4),
            h=round(h, 4),
            is_fullscreen_face=False,
            confidence=round(confidence, 3),
            preset_name=f"hybrid_{rect.preset_name}",
        )

    @staticmethod
    def _adjust_rect_aspect(rect: WebcamCrop, target_aspect: float) -> WebcamCrop:
        req_w_over_h = target_aspect * (9.0 / 16.0)
        w = rect.w
        h = w / req_w_over_h
        if h > rect.h:
            h = rect.h
            w = min(rect.w, h * req_w_over_h)
        x0 = rect.x + (rect.w - w) / 2.0
        y0 = rect.y + (rect.h - h) / 2.0
        return WebcamCrop(
            x=round(x0, 4),
            y=round(y0, 4),
            w=round(w, 4),
            h=round(h, 4),
            is_fullscreen_face=False,
            confidence=rect.confidence,
            preset_name=rect.preset_name,
        )

    @staticmethod
    def _detect_webcam_rectangle(frames_bgr: list[np.ndarray]) -> Optional[WebcamCrop]:
        if not frames_bgr:
            return None

        grays = [np.mean(f.astype(np.float32), axis=2) for f in frames_bgr]
        gx_list = [np.abs(g[:, 2:] - g[:, :-2]) for g in grays]
        gy_list = [np.abs(g[2:, :] - g[:-2, :]) for g in grays]

        gx = np.mean(gx_list, axis=0)  # (360, 638)
        gy = np.mean(gy_list, axis=0)  # (358, 640)

        best_score = 0.0
        best_crop: Optional[WebcamCrop] = None

        for side in ("left", "right"):
            for w in range(90, 225, 3):
                x_edge = w - 1 if side == "left" else (639 - w)
                x_slice = slice(0, w) if side == "left" else slice(640 - w, 640)

                for y0 in [0] + list(range(20, 245, 8)):
                    for h in range(60, 145, 3):
                        y1 = y0 + h
                        if y1 > 358:
                            continue
                        aspect = (w / 640.0 * 16.0) / (h / 360.0 * 9.0)
                        if not (1.10 <= aspect <= 2.10):
                            continue

                        v_band = gx[y0:y1, max(0, x_edge - 1) : min(638, x_edge + 2)]
                        v_edge = float(np.mean(v_band))
                        v_sharp = float(np.mean(np.max(v_band, axis=1) > 12.0))
                        if v_sharp < 0.72:
                            continue

                        if y1 <= 354:
                            h_band_bot = gy[y1 - 1 : y1 + 2, x_slice]
                            h_edge_bot = float(np.mean(h_band_bot))
                            h_sharp_bot = float(np.mean(np.max(h_band_bot, axis=0) > 12.0))
                        else:
                            h_edge_bot, h_sharp_bot = v_edge, 0.85

                        if y0 >= 6:
                            h_band_top = gy[y0 - 1 : y0 + 2, x_slice]
                            h_edge_top = float(np.mean(h_band_top))
                            h_sharp_top = float(np.mean(np.max(h_band_top, axis=0) > 12.0))
                        else:
                            h_edge_top, h_sharp_top = h_edge_bot, h_sharp_bot

                        h_sharp = (
                            min(h_sharp_bot, h_sharp_top)
                            if (y0 >= 6 and y1 <= 354)
                            else max(h_sharp_bot, h_sharp_top)
                        )
                        if h_sharp < 0.72:
                            continue

                        score = (v_edge + 0.5 * (h_edge_bot + h_edge_top)) * (v_sharp * h_sharp)
                        if score > best_score and score >= 22.0:
                            best_score = score
                            x0_norm = 0.0 if side == "left" else (640 - w) / 640.0
                            best_crop = WebcamCrop(
                                x=round(x0_norm, 4),
                                y=round(y0 / 360.0, 4),
                                w=round(w / 640.0, 4),
                                h=round(h / 360.0, 4),
                                is_fullscreen_face=False,
                                confidence=round(min(0.99, score / 100.0), 3),
                                preset_name=f"rect_{side}",
                            )

        return best_crop

    @staticmethod
    def _pick_consistent_cluster(
        faces: list[tuple[float, float, float, float, float]],
    ) -> list[tuple[float, float, float, float, float]]:
        best_cluster: list[tuple[float, float, float, float, float]] = []
        for anchor in faces:
            acx, acy = anchor[0], anchor[1]
            cluster = [
                f for f in faces if abs(f[0] - acx) <= 0.07 and abs(f[1] - acy) <= 0.09
            ]
            if len(cluster) > len(best_cluster) or (
                len(cluster) == len(best_cluster)
                and sum(x[4] for x in cluster) > sum(x[4] for x in best_cluster)
            ):
                best_cluster = cluster
        return best_cluster

    @staticmethod
    def _box_around_face(
        cx: float,
        cy: float,
        fh: float,
        target_aspect: float,
        confidence: float,
        is_fullscreen: bool,
    ) -> WebcamCrop:
        cam_h = float(np.clip(fh * 2.65, 0.16, 0.34))
        cam_w = float(np.clip(cam_h * target_aspect * (9.0 / 16.0), 0.16, 0.38))

        # Симметрия относительно края экрана: не даем рамке выйти за пределы вебкамеры в основное видео
        dist_to_side = cx if cx < 0.5 else (1.0 - cx)
        max_w_by_symmetry = max(0.15, dist_to_side * 1.98)
        if cam_w > max_w_by_symmetry:
            cam_w = max_w_by_symmetry
            cam_h = float(np.clip(cam_w * (16.0 / 9.0) / target_aspect, 0.15, 0.34))

        box_cx = cx
        box_cy = cy + fh * 0.12

        x0 = float(np.clip(box_cx - cam_w / 2.0, 0.0, 1.0 - cam_w))
        y0 = float(np.clip(box_cy - cam_h / 2.0, 0.0, 1.0 - cam_h))

        side_str = "left" if cx < 0.5 else "right"
        vert_str = "top" if cy < 0.45 else ("bottom" if cy > 0.62 else "mid")
        return WebcamCrop(
            x=round(x0, 4),
            y=round(y0, 4),
            w=round(cam_w, 4),
            h=round(cam_h, 4),
            is_fullscreen_face=is_fullscreen,
            confidence=round(confidence, 3),
            preset_name=f"auto_{vert_str}_{side_str}",
        )

    @staticmethod
    def _extract_sample_frames_720p(video_path: Path, num_frames: int = 6) -> list[np.ndarray]:
        """Извлекает несколько кадров 1280x720 (истинные 16:9) для точного поиска даже маленьких вебок."""
        cmd = [
            get_ffmpeg_path(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            "fps=1/2,scale=1280:720",
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=25, creationflags=WIN_NO_WINDOW)
            frame_bytes = 1280 * 720 * 3
            total = len(res.stdout) // frame_bytes
            frames = []
            for i in range(total):
                chunk = res.stdout[i * frame_bytes : (i + 1) * frame_bytes]
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape(720, 1280, 3)
                frames.append(arr)
            return frames
        except Exception:
            return []

    @classmethod
    def _detect_faces_highres_sides(
        cls,
        sess: ort.InferenceSession,
        frame_bgr_720p: np.ndarray,
        score_thresh: float = 0.54,
    ) -> list[tuple[float, float, float, float, float]]:
        """
        Вырезает левый край (ширина 320px из 1280) и правый край (ширина 320px из 1280)
        на высоте 640px и склеивает их в один тензор 640x640 в родном разрешении 720p!
        Это увеличивает мелкие игровые вебки в 2 раза без потери скорости.
        """
        canvas = np.zeros((1, 640, 640, 3), dtype=np.float32)
        # Берем 640 строк по высоте (с небольшим шагом 720 -> 640)
        y_idx = np.linspace(0, 719, 640).astype(np.int32)
        left_strip = frame_bgr_720p[y_idx, :320, :]
        right_strip = frame_bgr_720p[y_idx, 1280 - 320 :, :]

        combined = np.concatenate([left_strip, right_strip], axis=1).astype(np.float32)
        norm_img = np.power(combined / 255.0, 0.78) * 255.0
        canvas[0] = norm_img
        tensor = canvas.transpose(0, 3, 1, 2)

        outs = dict(
            zip([o.name for o in sess.get_outputs()], sess.run(None, {"input": tensor}))
        )
        found: list[tuple[float, float, float, float, float]] = []

        for s in (8, 16, 32):
            cls_out = outs[f"cls_{s}"][0, :, 0]
            obj_out = outs[f"obj_{s}"][0, :, 0]
            bbox = outs[f"bbox_{s}"][0]
            scores = np.sqrt(np.clip(cls_out, 0.0, 1.0) * np.clip(obj_out, 0.0, 1.0))

            idxs = np.where(scores >= score_thresh)[0]
            grid_w = 640 // s
            for idx in idxs:
                sc = float(scores[idx])
                row, col = divmod(int(idx), grid_w)
                px = float((col + bbox[idx, 0]) * s)
                py = float((row + bbox[idx, 1]) * s)
                pw = float(np.exp(bbox[idx, 2]) * s)
                ph = float(np.exp(bbox[idx, 3]) * s)

                # Не берем артефакты прямо на шве склейки x=320
                if 295 <= px <= 345:
                    continue

                if px < 320:
                    cx = px / 1280.0
                else:
                    cx = (1280.0 - 320.0 + (px - 320.0)) / 1280.0

                cy = py / 640.0
                w = pw / 1280.0
                h = ph / 640.0
                if 0.012 <= w <= 0.25 and 0.02 <= h <= 0.35:
                    found.append((cx, cy, w, h, sc))

        return found

    @staticmethod
    def _detect_faces_16x9(
        sess: ort.InferenceSession,
        frame_bgr_360p: np.ndarray,
        score_thresh: float = 0.52,
    ) -> list[tuple[float, float, float, float, float]]:
        canvas = np.zeros((1, 640, 640, 3), dtype=np.float32)
        norm_img = np.power(frame_bgr_360p.astype(np.float32) / 255.0, 0.78) * 255.0
        canvas[0, :360, :640, :] = norm_img
        tensor = canvas.transpose(0, 3, 1, 2)

        outs = dict(
            zip([o.name for o in sess.get_outputs()], sess.run(None, {"input": tensor}))
        )
        found: list[tuple[float, float, float, float, float]] = []

        for s in (8, 16, 32):
            cls_out = outs[f"cls_{s}"][0, :, 0]
            obj_out = outs[f"obj_{s}"][0, :, 0]
            bbox = outs[f"bbox_{s}"][0]
            scores = np.sqrt(np.clip(cls_out, 0.0, 1.0) * np.clip(obj_out, 0.0, 1.0))

            idxs = np.where(scores >= score_thresh)[0]
            grid_w = 640 // s
            for idx in idxs:
                sc = float(scores[idx])
                row, col = divmod(int(idx), grid_w)
                cx = float((col + bbox[idx, 0]) * s / 640.0)
                cy = float((row + bbox[idx, 1]) * s / 360.0)
                w = float(np.exp(bbox[idx, 2]) * s / 640.0)
                h = float(np.exp(bbox[idx, 3]) * s / 360.0)
                if cy <= 1.02 and 0.015 <= w <= 0.65 and 0.025 <= h <= 0.85:
                    found.append((cx, cy, w, h, sc))

        return found

    @staticmethod
    def _merge_detections(
        faces: list[tuple[float, float, float, float, float]],
    ) -> list[tuple[float, float, float, float, float]]:
        faces.sort(key=lambda x: x[4], reverse=True)
        filtered: list[tuple[float, float, float, float, float]] = []
        for cand in faces:
            if all(abs(cand[0] - ex[0]) > 0.05 or abs(cand[1] - ex[1]) > 0.07 for ex in filtered):
                filtered.append(cand)
        return filtered[:8]
