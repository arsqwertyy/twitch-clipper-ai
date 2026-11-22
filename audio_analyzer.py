import math
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path


@dataclass
class SmartTrimResult:
    """Результат умной подрезки начала и конца клипа по фразам Whisper и пикам звука."""
    trim_start: float
    trim_end: float
    peak_audio_time: float
    peak_audio_db: float
    reason: str
    is_silent_afk: bool = False
    climax_onset_time: float = -1.0
    rms_env: Optional[tuple[float, ...]] = None
    rms_step_sec: float = 0.25


@dataclass
class BannerPlacementResult:
    """Результат выбора оптимальной точки вставки рекламного баннера."""
    pause_time: float
    reason: str
    gap_sec: float = 0.0
    lead_before_climax: float = 0.0
    is_phrase_boundary: bool = False


SPEECH_EMOTIONAL_ROOTS = (
    "еба", "бля", "сук", "пизд", "аху", "оху", "ниху", "жест", "шок", "крик",
    "смерт", "убил", "капец", "треш", "дичь", "боже", "черт", "чёрт", "втф",
    "ахах", "хаха", "стой", "куда", "чего", "помоги", "ужас", "кошмар", "гений",
    "легенд", "разрыв", "сюда", "имба", "кринж", "позор",
)

# Служебные/связующие слова без пунктуации, на которых нельзя обрывать речь на паузу
HANGING_TAIL_WORDS = frozenset({
    "и", "а", "но", "да", "или", "либо", "то", "что", "чтобы", "чтоб", "как", "если", "когда",
    "пока", "где", "куда", "откуда", "зачем", "почему", "в", "во", "на", "с", "со", "к", "ко",
    "по", "за", "из", "изо", "у", "о", "об", "обо", "от", "ото", "до", "для", "без", "при",
    "про", "под", "над", "перед", "между", "через", "не", "ни", "бы", "ли", "же", "вот",
    "типа", "короче", "просто", "вообще", "прям", "прямо", "очень", "самый", "этот", "эта",
    "это", "эти", "мой", "моя", "моё", "твой", "его", "её", "их", "наш", "ваш", "свой",
    "я", "ты", "он", "она", "оно", "мы", "вы", "они", "меня", "тебя", "мне", "тебе",
})


class AudioPeakTracker:
    """
    Отслеживает громкость и резкость ГОЛОСОВОГО диапазона (250–3600 Гц) по 5-секундным .ts сегментам.
    Игнорирует низкочастотные взрывы и саб-бас игр, фокусируясь на криках, смехе и эмоциях стримера.
    """

    def __init__(self, history_size: int = 30):
        self._rms_history: deque[float] = deque(maxlen=history_size)
        self._peak_history: deque[float] = deque(maxlen=history_size)
        self._seen_segments: set[str] = set()
        self.last_z_score: float = 0.0
        self.last_db: float = -60.0

    def analyze_recent_segments(self, segments: list[Path]) -> tuple[float, float]:
        """
        Анализирует новые завершенные сегменты (последние 1-2 файла по 5 сек).
        Возвращает: (audio_z_score, peak_db).
        """
        if not segments:
            return 0.0, -60.0

        # Берем последние 2 завершенных сегмента (~10 секунд эфира)
        recent = segments[-2:]
        energies: list[float] = []

        for seg in recent:
            pcm = self._extract_8k_audio(seg)
            if pcm.size < 800:
                continue
            # Разбиваем на окна по 0.25 сек (2000 сэмплов при 8 кГц)
            n_frames = pcm.size // 2000
            if n_frames < 1:
                continue
            frames = pcm[: n_frames * 2000].reshape(n_frames, 2000)
            frame_rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-7
            energies.extend(frame_rms.tolist())

        if not energies:
            return 0.0, -60.0

        latest_name = recent[-1].name
        is_new_segment = latest_name not in self._seen_segments
        if is_new_segment:
            if len(self._seen_segments) > 500:
                self._seen_segments.clear()
            self._seen_segments.add(latest_name)

        arr = np.array(energies, dtype=np.float32)
        # Комбинируем 92-й перцентиль голосовой громкости и контраст всплеска (p95 - медиана),
        # чтобы ровный громкий фон (музыка/стрельба) давал меньше шума, чем резкий смех или крик.
        p92 = float(np.percentile(arr, 92))
        p95 = float(np.percentile(arr, 95))
        med = float(np.median(arr))
        burst_val = 0.72 * p92 + 0.28 * max(0.0, p95 - med)
        mean_rms = float(np.mean(arr))
        peak_db = float(20.0 * math.log10(max(p92, 1e-5)))

        if len(self._peak_history) >= 6:
            hist_arr = np.array(self._peak_history, dtype=np.float32)
            long_med = float(np.median(hist_arr))
            mad = float(np.median(np.abs(hist_arr - long_med)))
            robust_std = 1.4826 * mad
            recent_med = float(np.median(hist_arr[-5:]))
            hist_base = max(long_med, recent_med)
            hist_std = max(0.03, robust_std, hist_base * 0.28)
            z_score = (burst_val - hist_base) / hist_std
        else:
            z_score = 0.0

        # Всегда пополняем скользящую историю новых сегментов:
        # медиана + MAD не залипают при переходе от тихой сцены к громкой игре/музыке.
        if is_new_segment:
            self._rms_history.append(mean_rms)
            self._peak_history.append(burst_val)

        self.last_z_score = round(float(z_score), 2)
        self.last_db = round(peak_db, 1)
        return self.last_z_score, self.last_db

    @staticmethod
    def _extract_8k_audio(seg_path: Path) -> np.ndarray:
        cmd = [
            get_ffmpeg_path(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(seg_path),
            "-vn",
            # Полосовой фильтр человеческого голоса / смеха / крика (отсекает саб-бас и взрывы игр)
            "-af",
            "highpass=f=250,lowpass=f=3600",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=8, creationflags=WIN_NO_WINDOW)
            if res.returncode != 0 or not res.stdout:
                return np.array([], dtype=np.float32)
            return np.frombuffer(res.stdout, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)


def _clamp_to_max_duration(
    trim_start: float,
    trim_end: float,
    peak_time: float,
    min_clip_duration: float,
    max_clip_duration: Optional[float],
) -> tuple[float, float, bool]:
    """Гарантирует, что длина клипа не превышает max_clip_duration, сохраняя точку пика."""
    if max_clip_duration is None or max_clip_duration <= 0:
        return trim_start, trim_end, False
    limit = max(min_clip_duration, float(max_clip_duration) - 0.15)
    if (trim_end - trim_start) <= limit:
        return trim_start, trim_end, False

    excess = (trim_end - trim_start) - limit
    # Сначала аккуратно подрезаем лишнее с начала (оставляя контекст перед пиком), остаток — с конца
    max_cut_start = max(0.0, (peak_time - 6.0) - trim_start)
    cut_start = min(excess * 0.5, max_cut_start)
    trim_start += cut_start
    rem = excess - cut_start
    if rem > 0:
        trim_end = max(trim_start + min_clip_duration, trim_end - rem)
    if (trim_end - trim_start) > limit:
        trim_start = max(0.0, trim_end - limit)
    return trim_start, trim_end, True


def compute_smart_hook_trim(
    audio_16k: np.ndarray,
    words: list,
    min_clip_duration: float = 20.0,
    max_clip_duration: Optional[float] = None,
) -> SmartTrimResult:
    """
    Умная обрезка клипа по началу фразы и пикам звука (Smart Hook):
    1. Находит огибающую громкости (шаг 0.2 сек) и точку главного звукового пика (смех/крик/реакция).
    2. Если в самом начале клипа идет тишина — подрезает ее до начала первой реплики или звука (-0.25с).
    3. Если клип начался посреди оборванного предложения (в первые 0.3с), а через 0.5–3.8с после паузы
       начинается новая целая фраза — переносит старт клипа ровно на начало этой новой фразы!
    4. В конце клипа убирает пустую тишину или случайно захваченное начало следующей оборванной фразы,
       при этом гарантированно сохраняя весь пик громкости (смех/крик).
    """
    total_duration = float(audio_16k.size) / 16000.0 if audio_16k.size > 0 else 0.0

    # 1. Считаем огибающую громкости по окнам 0.25 сек (4000 сэмплов)
    step_samples = 4000
    step_sec = 0.25
    n_steps = audio_16k.size // step_samples
    rms_tuple: Optional[tuple[float, ...]] = None
    if n_steps >= 4:
        frames = audio_16k[: n_steps * step_samples].reshape(n_steps, step_samples)
        rms_env = np.sqrt(np.mean(frames**2, axis=1)) + 1e-7
        rms_tuple = tuple(round(float(v), 5) for v in rms_env)
        peak_idx = int(np.argmax(rms_env))
        peak_time = peak_idx * step_sec
        peak_db = float(20.0 * math.log10(float(rms_env[peak_idx])))

        # Порог активного звука (речь/смех/игра выше тишины, не выше -31 дБ чтобы не резать тихий геймплей)
        noise_floor = float(np.percentile(rms_env, 20))
        active_thresh = min(0.028, max(noise_floor * 2.0, float(np.percentile(rms_env, 45)) * 0.55, 0.010))
        active_indices = np.where(rms_env >= active_thresh)[0]
        first_sound_t = float(active_indices[0] * step_sec) if active_indices.size > 0 else 0.0
        last_sound_t = float((active_indices[-1] + 1) * step_sec) if active_indices.size > 0 else total_duration

        # Уточняем точку кульминации с учетом эмоциональных слов/восклицаний из Whisper
        if words:
            emo_score_env = rms_env / (float(np.max(rms_env)) + 1e-7)
            for w in words:
                lw = str(w.word).lower()
                if "!" in lw or "?" in lw or any(r in lw for r in SPEECH_EMOTIONAL_ROOTS):
                    idx_w = min(n_steps - 1, max(0, int(float(w.start) / step_sec)))
                    lo_i = max(0, idx_w - 2)
                    hi_i = min(n_steps, idx_w + 3)
                    emo_score_env[lo_i:hi_i] += 0.35
            peak_idx = int(np.argmax(emo_score_env))
            peak_time = peak_idx * step_sec

        # Ищем начало самой кульминации (climax_onset_time): отступаем от пика назад,
        # пока громкость держится заметно выше медианы клипа (смех / крик / взрыв).
        med_rms = float(np.median(rms_env))
        peak_rms = float(rms_env[peak_idx])
        onset_thresh = max(med_rms * 1.45, peak_rms * 0.42)
        onset_idx = peak_idx
        dip_used = False
        for j in range(peak_idx - 1, max(-1, peak_idx - 17), -1):
            if float(rms_env[j]) >= onset_thresh:
                onset_idx = j
                dip_used = False
            elif not dip_used and j > 0 and float(rms_env[j - 1]) >= onset_thresh:
                dip_used = True
            else:
                break
        climax_onset_time = onset_idx * step_sec
        if words:
            for w in words:
                ws = float(w.start)
                if climax_onset_time - 0.9 <= ws <= peak_time + 0.15:
                    lw = str(w.word).lower()
                    if "!" in lw or any(r in lw for r in SPEECH_EMOTIONAL_ROOTS):
                        climax_onset_time = min(climax_onset_time, ws)
    else:
        peak_time = total_duration * 0.5
        climax_onset_time = max(0.0, peak_time - 0.8)
        peak_db = -20.0
        first_sound_t = 0.0
        last_sound_t = total_duration

    if total_duration <= min_clip_duration + 1.0:
        return SmartTrimResult(
            trim_start=0.0,
            trim_end=round(total_duration, 2),
            peak_audio_time=round(peak_time, 2),
            peak_audio_db=round(peak_db, 1),
            reason="короткий исходник",
            is_silent_afk=(not words and peak_db < -38.0),
            climax_onset_time=round(climax_onset_time, 2),
            rms_env=rms_tuple,
            rms_step_sec=step_sec,
        )

    if not words:
        # Если слов нет (только смех/крик/игра), режем только пустую тишину по краям
        is_afk = peak_db < -38.0
        t_start = max(0.0, min(first_sound_t - 0.3, peak_time - 4.0, total_duration - min_clip_duration))
        t_end = min(total_duration, max(last_sound_t + 0.5, peak_time + 3.0, t_start + min_clip_duration))
        t_start, t_end, _ = _clamp_to_max_duration(t_start, t_end, peak_time, min_clip_duration, max_clip_duration)
        return SmartTrimResult(
            round(t_start, 2),
            round(t_end, 2),
            round(peak_time, 2),
            round(peak_db, 1),
            "тишина/AFK" if is_afk else "по звуку",
            is_silent_afk=is_afk,
            climax_onset_time=round(climax_onset_time, 2),
            rms_env=rms_tuple,
            rms_step_sec=step_sec,
        )

    # 2. Умный поиск старта по началу фразы (Smart Hook)
    first_w_start = float(words[0].start)
    trim_start = 0.0
    reason_parts = []

    if first_w_start >= 1.2:
        # В начале была пауза перед первой репликой — подрезаем тишину до старта речи (но не режем ранний крик/смех)
        trim_start = max(0.0, min(first_w_start - 0.25, max(0.0, first_sound_t - 0.25)))
        if trim_start >= 0.5:
            reason_parts.append(f"убрана тишина в начале ({trim_start:.1f}с)")
    elif first_w_start <= 0.45 and len(words) >= 5:
        # Запись началась прямо посреди фразы! Ищем конец этой оборванной фразы в первые 4.0 секунды
        for i in range(len(words) - 1):
            w_cur = words[i]
            w_next = words[i + 1]
            if w_cur.end > 4.0:
                break
            pause = float(w_next.start) - float(w_cur.end)
            ends_punct = str(w_cur.word).strip().endswith((".", "!", "?"))
            if (pause >= 0.42 or (ends_punct and pause >= 0.18)) and float(w_next.start) <= 4.2:
                candidate_start = max(0.0, float(w_next.start) - 0.20)
                # Проверяем, что мы не отрежем главный звуковой пик и оставим достаточную длину клипа
                if candidate_start <= (peak_time - 0.5) and (total_duration - candidate_start) >= min_clip_duration:
                    trim_start = candidate_start
                    reason_parts.append(f"старт с целой фразы (+{trim_start:.1f}с)")
                break

    # Если клип длинный (>36с) и кульминация далеко впереди (>18с), аккуратно убираем затянутую подводку,
    # начиная с чистой фразы за ~11–14с до пика (для высокого удержания в TikTok / Shorts!)
    if trim_start < 2.0 and total_duration > 36.0 and peak_time > 18.0 and len(words) >= 12:
        max_intro_cut = min(9.5, peak_time - 11.0, total_duration - 24.0)
        if max_intro_cut >= 3.0:
            best_hook = 0.0
            for i in range(len(words) - 1):
                w_cur = words[i]
                w_next = words[i + 1]
                cand = float(w_next.start) - 0.20
                if cand < 2.5:
                    continue
                if cand > max_intro_cut:
                    break
                pause = float(w_next.start) - float(w_cur.end)
                ends_punct = str(w_cur.word).strip().endswith((".", "!", "?"))
                if (ends_punct and pause >= 0.22) or pause >= 0.55:
                    best_hook = cand
            if best_hook >= 2.5:
                trim_start = best_hook
                reason_parts.append(f"убрана затянутая подводка (+{trim_start:.1f}с)")

    # 3. Умный поиск конца клипа (чтобы не обрывать на начале нового предложения в последнюю секунду)
    trim_end = total_duration
    if len(words) >= 4:
        # Если в последние 1.8 секунды после длинной паузы (>=0.55с) началось 1-2 слова новой фразы без точки в конце
        for i in range(len(words) - 1, max(0, len(words) - 4), -1):
            w_cur = words[i]
            w_prev = words[i - 1]
            tail_len = total_duration - float(w_cur.start)
            pause = float(w_cur.start) - float(w_prev.end)
            last_word_punct = str(words[-1].word).strip().endswith((".", "!", "?"))
            if tail_len <= 1.8 and pause >= 0.55 and not last_word_punct:
                candidate_end = min(total_duration, float(w_prev.end) + 0.35)
                if candidate_end > (peak_time + 1.5) and (candidate_end - trim_start) >= min_clip_duration:
                    trim_end = candidate_end
                    reason_parts.append(f"убран обрывок фразы в конце (-{total_duration - trim_end:.1f}с)")
                break

    # Если после последнего слова и последнего громкого звука осталась длинная тишина (> 1.8 сек)
    last_active_end = max(float(words[-1].end) + 0.45, last_sound_t + 0.4, peak_time + 2.0)
    if trim_end == total_duration and (total_duration - last_active_end) >= 1.5:
        candidate_end = max(trim_start + min_clip_duration, last_active_end)
        if candidate_end < total_duration - 0.5:
            trim_end = candidate_end
            reason_parts.append(f"убрана тишина в конце (-{total_duration - trim_end:.1f}с)")

    trim_start, trim_end, was_clamped = _clamp_to_max_duration(
        trim_start, trim_end, peak_time, min_clip_duration, max_clip_duration
    )
    if was_clamped and max_clip_duration is not None:
        reason_parts.append(f"лимит {max_clip_duration:.0f}с")

    reason_str = ", ".join(reason_parts) if reason_parts else "без подрезки краев"
    return SmartTrimResult(
        trim_start=round(trim_start, 2),
        trim_end=round(trim_end, 2),
        peak_audio_time=round(peak_time, 2),
        peak_audio_db=round(peak_db, 1),
        reason=reason_str,
        climax_onset_time=round(climax_onset_time, 2),
        rms_env=rms_tuple,
        rms_step_sec=step_sec,
    )


def compute_smart_banner_time(
    clip_dur: float,
    position_mode: str = "smart",
    words: Optional[list] = None,
    phrases: Optional[list] = None,
    trim_info: Optional[SmartTrimResult] = None,
    pause_video: bool = True,
    ban_dur: float = 3.5,
) -> BannerPlacementResult:
    """
    Умный подбор точки вставки рекламного баннера (режим «интрига перед развязкой» + без обрыва слов):
    - В режиме 'smart' находит начало главной кульминации (смех/крик/панчлайн) и ставит паузу
      с баннером строго ПЕРЕД развязкой на естественной границе предложения или фразы Whisper,
      чтобы зритель гарантированно досмотрел баннер ради концовки.
    - Никогда не режет слово посередине и избегает «подвисших» предлогов/союзов ('и', 'в', 'что', 'не').
    - В режимах 'middle' / 'start' / 'end' также сдвигает вставку на чистую границу фразы и тихий участок.
    - Если слов нет (чистый геймплей/смех), находит самую тихую паузу по огибающей звука перед пиком.
    """
    words = words or []
    clip_dur = max(1.0, float(clip_dur))
    mode = str(position_mode or "smart").strip().lower()
    if mode not in ("smart", "middle", "start", "end"):
        mode = "smart"

    # 1. Безопасные границы, чтобы реклама не вылезала слишком рано (пока зритель не втянулся)
    # и не обрывала самую последнюю секунду клипа
    if pause_video:
        if clip_dur < 10.0:
            min_t = max(1.4, clip_dur * 0.22)
            max_t = min(clip_dur - 1.4, clip_dur * 0.78)
        elif clip_dur < 18.0:
            min_t = max(2.8, clip_dur * 0.22)
            max_t = min(clip_dur - 2.8, clip_dur * 0.78)
        else:
            min_t = max(4.5, clip_dur * 0.22)
            max_t = min(clip_dur - 4.5, clip_dur * 0.78)
    else:
        tail_guard = max(1.5, float(ban_dur) + 0.5)
        min_t = max(1.5, min(clip_dur * 0.22, clip_dur - tail_guard - 0.5))
        max_t = max(min_t, min(clip_dur - tail_guard, clip_dur * 0.78))

    if max_t < min_t + 0.3:
        fallback_t = max(1.0, min(clip_dur - 1.0, clip_dur * 0.50))
        return BannerPlacementResult(round(fallback_t, 3), "короткий клип (центр)")

    # 2. Восстанавливаем относительную огибающую звука и координаты кульминации (0 .. clip_dur)
    trim_start = float(trim_info.trim_start) if trim_info is not None else 0.0
    step_sec = float(trim_info.rms_step_sec) if (trim_info is not None and trim_info.rms_step_sec > 0) else 0.25
    rel_rms_arr: Optional[np.ndarray] = None
    med_rms = 1.0

    if trim_info is not None and trim_info.rms_env:
        full_env = np.asarray(trim_info.rms_env, dtype=np.float32)
        i0 = max(0, int(round(trim_start / step_sec)))
        i1 = min(full_env.size, max(i0 + 2, int(math.ceil((trim_start + clip_dur) / step_sec))))
        if i1 > i0 + 1:
            rel_rms_arr = full_env[i0:i1]
            med_rms = max(float(np.median(rel_rms_arr)), 1e-5)

    def get_rel_rms(t_rel: float) -> float:
        if rel_rms_arr is None or rel_rms_arr.size == 0:
            return 1.0
        idx = min(rel_rms_arr.size - 1, max(0, int(round(t_rel / step_sec))))
        return float(rel_rms_arr[idx]) / med_rms

    if trim_info is not None:
        peak_rel = max(0.0, min(clip_dur, float(trim_info.peak_audio_time) - trim_start))
        if trim_info.climax_onset_time >= 0.0:
            onset_rel = max(0.0, min(peak_rel, float(trim_info.climax_onset_time) - trim_start))
        else:
            onset_rel = max(0.0, peak_rel - 1.2)
    else:
        peak_rel = clip_dur * 0.62
        onset_rel = clip_dur * 0.54

    # 3. Определяем целевую точку интриги и коридор поиска
    target_onset = onset_rel
    target_peak = peak_rel
    used_secondary = False
    early_peak_fallback = False

    if mode == "start":
        ideal_t = min(max(1.8, clip_dur - 2.0), 3.0)
        search_lo = max(1.2, ideal_t - 1.8)
        search_hi = min(clip_dur - (1.5 if pause_video else ban_dur + 0.5), ideal_t + 3.0)
    elif mode == "end":
        ideal_t = max(1.8, clip_dur - (2.5 if pause_video else ban_dur + 2.0))
        search_lo = max(1.5, ideal_t - 3.2)
        search_hi = min(clip_dur - (1.0 if pause_video else ban_dur + 0.5), ideal_t + 1.8)
    elif mode == "middle":
        ideal_t = clip_dur * 0.50 if pause_video else max(min_t, clip_dur * 0.50 - ban_dur * 0.5)
        search_lo = max(min_t, ideal_t - 3.5)
        search_hi = min(max_t, ideal_t + 3.5)
    else:
        # РЕЖИМ SMART: ищем идеальную интригу строго ПЕРЕД развязкой!
        if onset_rel < min_t + 0.8:
            # Главный пик случился в самом начале клипа — ищем вторую эмоциональную волну во 2-й половине
            sec_lo = min_t + 1.6
            sec_hi = max(sec_lo + 1.0, clip_dur - 1.8)
            found_sec = False
            if rel_rms_arr is not None and rel_rms_arr.size >= 4:
                j0 = min(rel_rms_arr.size - 1, max(0, int(sec_lo / step_sec)))
                j1 = min(rel_rms_arr.size, max(j0 + 1, int(sec_hi / step_sec)))
                sub_env = rel_rms_arr[j0:j1].copy()
                if sub_env.size > 0:
                    sub_norm = sub_env / (float(np.max(rel_rms_arr)) + 1e-7)
                    for w in words:
                        ws = float(w.start)
                        if sec_lo <= ws <= sec_hi:
                            lw = str(w.word).lower()
                            if "!" in lw or any(r in lw for r in SPEECH_EMOTIONAL_ROOTS):
                                k = min(sub_norm.size - 1, max(0, int((ws - j0 * step_sec) / step_sec)))
                                sub_norm[max(0, k - 2) : min(sub_norm.size, k + 3)] += 0.35
                    best_k = int(np.argmax(sub_norm))
                    if float(sub_env[best_k]) >= med_rms * 1.30 or float(sub_norm[best_k]) >= 0.90:
                        target_peak = (j0 + best_k) * step_sec
                        target_onset = max(min_t + 0.5, target_peak - 1.0)
                        used_secondary = True
                        found_sec = True
            if not found_sec:
                # Ищем яркое эмоциональное слово во второй части
                late_emo = [
                    float(w.start)
                    for w in words
                    if sec_lo <= float(w.start) <= sec_hi
                    and ("!" in str(w.word) or any(r in str(w.word).lower() for r in SPEECH_EMOTIONAL_ROOTS))
                ]
                if late_emo:
                    target_peak = late_emo[0] + 0.5
                    target_onset = late_emo[0]
                    used_secondary = True
                else:
                    early_peak_fallback = True
                    target_onset = clip_dur * 0.50
                    target_peak = clip_dur * 0.58

        if pause_video:
            ideal_t = max(min_t, min(max_t, target_onset - 0.9))
        else:
            # Без стоп-кадра лучше запустить баннер заранее, чтобы он успел закончиться к самой развязке
            ideal_t = max(min_t, min(max_t, target_onset - ban_dur - 0.3))
        search_lo, search_hi = min_t, max_t

    if search_hi < search_lo:
        search_lo, search_hi = min_t, max_t

    # 4. Карта точных границ фраз субтитров (чтобы не замораживать «полуфразу» на экране)
    phrase_end_indices: set[int] = set()
    if phrases:
        acc = 0
        for p in phrases:
            if p:
                acc += len(p)
                phrase_end_indices.add(acc - 1)

    # 5. Оцениваем каждую естественную границу между словами Whisper (нулевой риск разрезать слово!)
    candidates: list[tuple[float, float, float, bool, str]] = []

    if len(words) >= 2:
        for i in range(len(words) - 1):
            w_cur = words[i]
            w_next = words[i + 1]
            w_cur_end = float(w_cur.end)
            w_next_start = float(w_next.start)
            gap = max(0.0, w_next_start - w_cur_end)

            # Точка разреза строго между словами (при паузе >= 0.24с — уже после исчезновения прошлого субтитра!)
            if gap >= 0.24:
                cut_t = min(w_next_start - 0.06, w_cur_end + 0.16)
            elif gap >= 0.04:
                cut_t = 0.5 * (w_cur_end + w_next_start)
            else:
                cut_t = w_next_start

            if not (search_lo <= cut_t <= search_hi):
                continue

            cur_txt = str(w_cur.word).strip()
            cur_clean = re.sub(r"[^\wа-яА-ЯёЁ]+", "", cur_txt.lower())
            ends_sentence = cur_txt.endswith((".", "!", "?", "…"))
            ends_question = cur_txt.endswith("?")
            ends_clause = cur_txt.endswith((",", ";", ":", "—", "-"))
            is_phrase_end = (i in phrase_end_indices) or ends_sentence or gap >= 0.45

            # А) Чистота речи и границ фраз
            speech_sc = min(gap, 1.0) * 2.2
            if gap >= 0.18:
                speech_sc += 0.85
            elif gap < 0.04:
                speech_sc -= 0.70

            if ends_sentence:
                speech_sc += 2.4
            elif ends_clause:
                speech_sc += 1.2
            if ends_question and mode == "smart":
                speech_sc += 0.85
            if is_phrase_end:
                speech_sc += 1.35
            if not (ends_sentence or ends_clause) and cur_clean in HANGING_TAIL_WORDS:
                speech_sc -= 2.1

            # Б) Тишина / спад громкости в точке разреза
            rel_rms = get_rel_rms(cut_t)
            audio_sc = 0.0
            if rel_rms <= 0.65:
                audio_sc = 0.85
            elif rel_rms <= 1.15:
                audio_sc = 0.25
            elif rel_rms >= 2.10:
                audio_sc = -1.60
            elif rel_rms >= 1.50:
                audio_sc = -0.60

            # В) Интрига перед кульминацией (для режима smart) или близость к целевому месту
            if mode == "smart":
                lead = target_onset - cut_t
                lead_to_peak = target_peak - cut_t
                if not pause_video:
                    # В режиме без паузы стремимся финишировать баннер перед развязкой
                    ideal_lead = ban_dur + 0.3
                    if 0.0 <= (lead - ban_dur) <= 3.0:
                        intrigue_sc = 3.2 - 0.45 * abs(lead - ideal_lead)
                    elif lead >= 0.4:
                        intrigue_sc = 1.6 - 0.20 * abs(lead - ideal_lead)
                    else:
                        intrigue_sc = -3.0
                else:
                    if 0.20 <= lead <= 3.50:
                        intrigue_sc = 3.5 - 0.42 * abs(lead - 1.10)
                    elif 3.50 < lead <= 7.50:
                        intrigue_sc = 1.8 - 0.24 * (lead - 3.50)
                    elif -0.40 <= lead < 0.20 and lead_to_peak >= 0.45:
                        intrigue_sc = 1.9
                    elif lead_to_peak < 0.35 and cut_t <= target_peak + 1.40:
                        # Внутри самого крика/смеха — жесткий штраф!
                        intrigue_sc = -4.6
                    elif cut_t > target_peak + 1.40:
                        # Развязка уже прошла — зритель свайпнет рекламу!
                        intrigue_sc = -3.3
                    else:
                        intrigue_sc = 0.4 - 0.15 * abs(cut_t - ideal_t)

                # Бонус, если сразу за паузой (в ближайших 1–3 словах) начинается эмоциональная развязка
                next_slice = words[i + 1 : min(len(words), i + 4)]
                cur_is_emo = "!" in cur_txt or any(r in cur_clean for r in SPEECH_EMOTIONAL_ROOTS)
                next_has_emo = any(
                    ("!" in str(nw.word) or any(r in str(nw.word).lower() for r in SPEECH_EMOTIONAL_ROOTS))
                    for nw in next_slice
                )
                if next_has_emo and not cur_is_emo and lead >= -0.3:
                    intrigue_sc += 0.90

                dist_sc = -0.16 * abs(cut_t - ideal_t)
                total_sc = speech_sc + audio_sc + intrigue_sc + dist_sc
            else:
                total_sc = speech_sc + audio_sc - 0.35 * abs(cut_t - ideal_t)

            if ends_question:
                kind = "после вопроса-интриги"
            elif ends_sentence:
                kind = "конец предложения"
            elif ends_clause:
                kind = "пауза по смыслу"
            elif is_phrase_end:
                kind = "граница фразы"
            else:
                kind = "граница слов"

            candidates.append((total_sc, cut_t, gap, is_phrase_end, kind))

    # Проверяем также чистые тихие зоны ДО первого слова или ПОСЛЕ последнего слова
    if words:
        first_start = float(words[0].start)
        if first_start >= search_lo + 0.5:
            pre_t = min(search_hi, max(search_lo, first_start - 0.22))
            lead = target_onset - pre_t
            sc = 2.6 + (1.5 if (mode == "smart" and 0.2 <= lead <= 4.0) else 0.0) - 0.22 * abs(pre_t - ideal_t)
            candidates.append((sc, pre_t, max(0.3, first_start - pre_t), True, "перед первой репликой"))
        last_end = float(words[-1].end)
        if last_end <= search_hi - 0.5:
            post_t = max(search_lo, min(search_hi, last_end + 0.25))
            lead = target_onset - post_t
            sc = 2.2 + (1.5 if (mode == "smart" and 0.2 <= lead <= 4.0) else (-1.5 if mode == "smart" and post_t > target_peak else 0.0)) - 0.22 * abs(post_t - ideal_t)
            candidates.append((sc, post_t, max(0.3, post_t - last_end), True, "после последней фразы"))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_t, best_gap, is_p_end, best_kind = candidates[0]
        lead_val = round(target_peak - best_t, 2)
        if mode == "smart":
            if early_peak_fallback:
                prefix = "середина (пик в начале)"
            elif used_secondary:
                prefix = f"интрига за {max(0.0, lead_val):.1f}с до 2-й волны"
            elif lead_val >= 0.3:
                prefix = f"интрига за {lead_val:.1f}с до кульминации"
            else:
                prefix = "чистая пауза в речи"
            gap_str = f", пауза {best_gap:.2f}с" if best_gap >= 0.08 else ""
            reason = f"{prefix} • {best_kind}{gap_str}"
        else:
            gap_str = f" (пауза {best_gap:.2f}с)" if best_gap >= 0.08 else ""
            reason = f"{best_kind}{gap_str}"
        return BannerPlacementResult(
            pause_time=round(best_t, 3),
            reason=reason,
            gap_sec=round(best_gap, 3),
            lead_before_climax=lead_val,
            is_phrase_boundary=is_p_end,
        )

    # 6. Фоллбек без слов (чистый геймплей / крик / музыка): ищем самую тихую точку по RMS перед кульминацией
    if rel_rms_arr is not None and rel_rms_arr.size >= 4:
        j_lo = min(rel_rms_arr.size - 1, max(0, int(math.ceil(search_lo / step_sec))))
        j_hi = min(rel_rms_arr.size - 1, max(j_lo, int(math.floor(search_hi / step_sec))))
        best_t = max(search_lo, min(search_hi, ideal_t))
        best_cost = 1e9
        for j in range(j_lo, j_hi + 1):
            t_cand = j * step_sec
            rel_rms = float(rel_rms_arr[j]) / med_rms
            cost = rel_rms * 1.4 + 0.32 * abs(t_cand - ideal_t)
            if mode == "smart" and t_cand > target_peak - 0.35:
                cost += 3.5
            if cost < best_cost:
                best_cost = cost
                best_t = t_cand
        lead_val = round(target_peak - best_t, 2)
        reason = (
            f"тихий вдох за {max(0.0, lead_val):.1f}с до пика звука"
            if mode == "smart" and lead_val >= 0.2
            else "тихая пауза по звуку"
        )
        return BannerPlacementResult(
            pause_time=round(best_t, 3),
            reason=reason,
            gap_sec=0.0,
            lead_before_climax=lead_val,
            is_phrase_boundary=True,
        )

    final_t = max(search_lo, min(search_hi, ideal_t))
    return BannerPlacementResult(
        pause_time=round(final_t, 3),
        reason="базовая позиция",
        lead_before_climax=round(target_peak - final_t, 2),
    )
