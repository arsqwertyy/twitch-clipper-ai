import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from audio_analyzer import SmartTrimResult, compute_smart_hook_trim
from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path, setup_nvidia_dlls

setup_nvidia_dlls()

from faster_whisper import WhisperModel  # noqa: E402


HALLUCINATION_PHRASES = (
    "субтитры сделал",
    "субтитры создавал",
    "субтитры подготовил",
    "субтитры добавил",
    "редактор субтитров",
    "корректор субтитров",
    "перевод выполнен",
    "dimatorzok",
    "dima torzok",
    "продолжение следует",
    "подписывайтесь на канал",
    "подпишись на канал",
    "ставь лайк",
    "ставьте лайки",
    "спасибо за просмотр",
    "спасибо за внимание",
    "до новых встреч",
    "amara.org",
    "музыка",
    "[музыка]",
    "♪",
)

DEFAULT_RU_PROMPT = (
    "Разговорная русская речь на стриме Twitch. "
    "Чат, донат, клип, игра, стрим, бро, короче."
)
DEFAULT_RU_HOTWORDS = (
    "донат донатофон стрим стример чат клип тильт имба катка чел бро кринж твич "
    "GTA ChatGPT Discord"
)

STRAY_PUNCT_RE = re.compile(r"^[.,!?;:…—\-]+$")

# Корни эмоциональных слов, триггеров и экспрессии для ярко-красной акцентной подсветки в субтитрах
EMOTIONAL_ROOTS = (
    "еба", "бля", "сук", "пизд", "аху", "оху", "ниху", "жест", "шок", "крик",
    "смерт", "убил", "капец", "треш", "дичь", "зажрав", "дурак", "псих", "страш",
    "боже", "черт", "чёрт", "втф", "wtf", "ахах", "хаха", "стой", "куда", "чего",
    "нельзя", "помоги", "спаси", "ужас", "кошмар", "гений", "легенд", "разрыв",
)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


class SubtitleGenerator:
    """
    Транскрибирует аудио с помощью Faster-Whisper на видеокарте NVIDIA (CUDA)
    и создает динамические .ass субтитры в стиле TikTok / Shorts:
    - Поп-анимация (эффект пружинки при появлении каждого слова)
    - 2 цвета акцентов (Желтый для обычных слов + Ярко-красный для эмоциональных триггеров и крика).
    """

    def __init__(self, sub_cfg: dict, video_cfg: dict):
        self.enabled: bool = bool(sub_cfg.get("enabled", True))
        self.model_size: str = str(sub_cfg.get("model_size", "large-v3-turbo"))
        self.language: Optional[str] = sub_cfg.get("language", "ru") or None
        self.device: str = str(sub_cfg.get("device", "cuda"))
        self.compute_type: str = str(sub_cfg.get("compute_type", "int8_float16"))
        self.words_per_phrase: int = max(1, int(sub_cfg.get("words_per_phrase", 3)))
        self.font_name: str = str(sub_cfg.get("font_name", "Arial"))
        self.font_size: int = int(sub_cfg.get("font_size", 72))
        self.margin_v: int = int(sub_cfg.get("margin_v", 420))
        self.smart_hook: bool = bool(sub_cfg.get("smart_hook", True))
        self.initial_prompt: Optional[str] = sub_cfg.get("initial_prompt")
        self.hotwords: Optional[str] = sub_cfg.get("hotwords")

        self.play_res_x: int = int(video_cfg.get("width", 1080))
        self.play_res_y: int = int(video_cfg.get("height", 1920))
        if video_cfg.get("mode", "tiktok") != "tiktok":
            self.play_res_x = 1920
            self.play_res_y = 1080
            self.margin_v = 80
            self.font_size = 58

        self._model: Optional[WhisperModel] = None
        self.active_device: str = self.device
        self.last_words: list[WordTiming] = []
        self.last_phrases: list[list[WordTiming]] = []

    def load_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        candidates: list[tuple[str, str]] = []
        if self.device == "cuda":
            candidates.append(("cuda", self.compute_type))
            for fallback_ct in ("int8_float16", "int8", "float16"):
                if (self.device, fallback_ct) not in candidates:
                    candidates.append(("cuda", fallback_ct))
        else:
            candidates.append((self.device, self.compute_type))
        if ("cpu", "int8") not in candidates:
            candidates.append(("cpu", "int8"))

        last_exc: Optional[Exception] = None
        for dev, ct in candidates:
            try:
                self._model = WhisperModel(self.model_size, device=dev, compute_type=ct)
                self.active_device = f"{dev} ({ct})"
                return self._model
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(f"Не удалось загрузить модель Whisper '{self.model_size}': {last_exc!r}")

    def generate_ass_for_video(self, video_path: Path, ass_output_path: Path) -> int:
        cnt, _ = self.generate_ass_and_trim(video_path, ass_output_path)
        return cnt

    def generate_ass_and_trim(
        self,
        video_path: Path,
        ass_output_path: Path,
        max_duration: Optional[float] = None,
    ) -> tuple[int, Optional[SmartTrimResult]]:
        """
        Распознает речь, рассчитывает умную подрезку краев (Smart Hook) и точку кульминации звука,
        сдвигает таймкоды субтитров и генерирует .ass файл с поп-анимацией и двухцветными акцентами.
        """
        audio = self._load_audio_16k(video_path)
        words: list[WordTiming] = []

        if self.enabled and audio.size >= 8000:
            # Для Whisper убираем саб-бас / гул взрывов (< 65 Гц), оставляя сырой звук для Smart Hook
            speech_audio = self._load_audio_16k(video_path, highpass_hz=65)
            if speech_audio.size < 8000:
                speech_audio = audio
            try:
                words = self._transcribe_audio_array(speech_audio)
            except Exception as exc:
                msg = repr(exc).lower()
                is_gpu_error = any(k in msg for k in ("cuda", "cudnn", "cublas", "out of memory", "gpu"))
                if self.active_device.startswith("cpu") or not is_gpu_error:
                    raise
                if "int8_float16" not in self.active_device and "int8" not in self.active_device:
                    try:
                        print(f"[whisper] Ошибка VRAM/GPU ({exc!r}), пробую cuda int8_float16...")
                        self._model = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")
                        self.active_device = "cuda (int8_float16)"
                        words = self._transcribe_audio_array(speech_audio)
                    except Exception:
                        print("[whisper] Переключаюсь на CPU (int8)...")
                        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                        self.active_device = "cpu (int8)"
                        words = self._transcribe_audio_array(speech_audio)
                else:
                    print(f"[whisper] Ошибка на GPU ({exc!r}), переключаюсь на CPU...")
                    self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                    self.active_device = "cpu (int8)"
                    words = self._transcribe_audio_array(speech_audio)

        trim_info: Optional[SmartTrimResult] = None
        if audio.size >= 8000:
            total_sec = audio.size / 16000.0
            if self.smart_hook:
                trim_info = compute_smart_hook_trim(audio, words, max_clip_duration=max_duration)
            else:
                # Даже если обрезка краев выключена, находим секунду пика громкости для Zoom-In эффекта
                raw_trim = compute_smart_hook_trim(audio, words, max_clip_duration=max_duration)
                t_end_lim = min(total_sec, max_duration - 0.15) if (max_duration and max_duration > 5.0) else total_sec
                trim_info = SmartTrimResult(
                    trim_start=0.0,
                    trim_end=round(t_end_lim, 2),
                    peak_audio_time=raw_trim.peak_audio_time,
                    peak_audio_db=raw_trim.peak_audio_db,
                    reason="без подрезки",
                    is_silent_afk=raw_trim.is_silent_afk,
                    climax_onset_time=raw_trim.climax_onset_time,
                    rms_env=raw_trim.rms_env,
                    rms_step_sec=raw_trim.rms_step_sec,
                )

            t_start = trim_info.trim_start
            t_end = trim_info.trim_end
            if t_start > 0.05 or t_end < (total_sec - 0.05):
                shifted: list[WordTiming] = []
                for w in words:
                    if w.end <= t_start or w.start >= t_end:
                        continue
                    ns = max(0.0, w.start - t_start)
                    ne = max(ns + 0.08, min(t_end - t_start, w.end - t_start))
                    shifted.append(WordTiming(word=w.word, start=ns, end=ne))
                words = shifted

        self.last_words = list(words)
        self.last_phrases = []

        if not self.enabled or not words:
            return 0, trim_info

        phrases = self._group_words_into_phrases(words)
        self.last_phrases = phrases
        ass_content = self._build_ass_document(phrases)
        ass_output_path.write_text(ass_content, encoding="utf-8-sig")
        return len(words), trim_info

    def write_ass_for_pause(self, ass_output_path: Path, pause_time: float) -> bool:
        """
        Пересобирает .ass для версии со стоп-кадром баннера:
        разрывает группу субтитров ровно по точке pause_time и очищает ±0.04с вокруг стоп-кадра,
        чтобы на паузе за баннером никогда не зависали слова и не спойлерилась следующая фраза.
        """
        if not self.enabled or not self.last_words:
            return False
        phrases = self._group_words_into_phrases(self.last_words, split_at_time=pause_time)
        ass_content = self._build_ass_document(phrases, pause_time=pause_time)
        ass_output_path.write_text(ass_content, encoding="utf-8-sig")
        return True

    @staticmethod
    def _load_audio_16k(video_path: Path, highpass_hz: int = 0) -> np.ndarray:
        cmd = [
            get_ffmpeg_path(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
        ]
        if highpass_hz > 0:
            cmd.extend(["-af", f"highpass=f={int(highpass_hz)}"])
        cmd.extend(["-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        res = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=WIN_NO_WINDOW)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg не смог извлечь аудио: {res.stderr.decode('utf-8', 'replace')[-300:]}")
        return np.frombuffer(res.stdout, dtype=np.float32).copy()

    def _transcribe_audio_array(self, audio: np.ndarray) -> list[WordTiming]:
        model = self.load_model()
        if audio.size < 16000 // 2:
            return []

        prompt = self.initial_prompt
        hotwords = self.hotwords
        if self.language == "ru":
            if prompt is None:
                prompt = DEFAULT_RU_PROMPT
            if hotwords is None:
                hotwords = DEFAULT_RU_HOTWORDS

        segments, _ = model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=(0.0, 0.2, 0.4, 0.6),
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.9,
            no_speech_threshold=0.55,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.35,
                min_speech_duration_ms=150,
                min_silence_duration_ms=450,
                speech_pad_ms=300,
            ),
            condition_on_previous_text=True,
            initial_prompt=prompt or None,
            hotwords=hotwords or None,
            hallucination_silence_threshold=1.5,
        )

        words: list[WordTiming] = []
        repeat_run = 0
        last_norm = ""

        for seg in segments:
            seg_text_lower = (seg.text or "").strip().lower()
            if not seg_text_lower or any(h in seg_text_lower for h in HALLUCINATION_PHRASES):
                continue
            # Отсекаем галлюцинации поверх фоновой музыки/игрового шума и зацикленные сегменты
            if getattr(seg, "no_speech_prob", 0.0) > 0.75 and getattr(seg, "avg_logprob", 0.0) < -1.0:
                continue
            if getattr(seg, "compression_ratio", 0.0) > 2.4:
                continue
            if not seg.words:
                continue

            for w in seg.words:
                cleaned = w.word.strip()
                if not cleaned:
                    continue
                if getattr(w, "probability", 1.0) < 0.05 and len(cleaned) <= 2:
                    continue

                start_t = max(0.0, float(w.start))
                end_t = max(start_t + 0.08, float(w.end))

                # Склеиваем разорванные дефисные части ('чуть' + '-чуть', 'какой' + '-то') и одиночную пунктуацию
                if words and (
                    cleaned.startswith("-")
                    or STRAY_PUNCT_RE.match(cleaned)
                    or words[-1].word.endswith("-")
                ):
                    words[-1].word = f"{words[-1].word}{cleaned}"
                    words[-1].end = max(words[-1].end, end_t)
                    continue

                norm = re.sub(r"[^\wа-яА-ЯёЁ]+", "", cleaned.lower())
                if norm and norm == last_norm:
                    repeat_run += 1
                    if repeat_run >= 3:
                        continue
                else:
                    repeat_run = 0
                    last_norm = norm

                if words and start_t < words[-1].end:
                    words[-1].end = max(words[-1].start + 0.06, start_t)

                words.append(WordTiming(word=cleaned, start=start_t, end=end_t))

        return words

    def _group_words_into_phrases(
        self,
        words: list[WordTiming],
        split_at_time: Optional[float] = None,
    ) -> list[list[WordTiming]]:
        phrases: list[list[WordTiming]] = []
        current: list[WordTiming] = []

        for w in words:
            if current:
                prev_w = current[-1]
                pause = w.start - prev_w.end
                prev_word = prev_w.word
                ends_sentence = prev_word.endswith((".", "!", "?"))
                crosses_banner = (
                    split_at_time is not None
                    and prev_w.start < split_at_time
                    and (w.start >= split_at_time - 0.02 or prev_w.end <= split_at_time <= w.start)
                )
                if len(current) >= self.words_per_phrase or pause > 0.65 or ends_sentence or crosses_banner:
                    phrases.append(current)
                    current = []
            current.append(w)

        if current:
            phrases.append(current)

        return phrases

    @staticmethod
    def _is_emotional_word(word: str) -> bool:
        lower = word.lower()
        if "!" in lower or "?" in lower:
            return True
        return any(root in lower for root in EMOTIONAL_ROOTS)

    def _build_ass_document(
        self,
        phrases: list[list[WordTiming]],
        pause_time: Optional[float] = None,
    ) -> str:
        """
        Генерирует .ass документ со стилем TikTok:
        динамические субтитры с поп-анимацией (пружинкой) и 2 цветами акцентов (Желтый и Красный).
        """
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.play_res_x}
PlayResY: {self.play_res_y}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,{self.font_name},{self.font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,5,3,2,60,60,{self.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        lines = [header.strip()]
        pause_pad = 0.04

        # Динамические субтитры с поп-анимацией (пружинкой) и 2 цветами акцентов (Желтый + Красный)
        for p_idx, phrase in enumerate(phrases):
            next_phrase_start = phrases[p_idx + 1][0].start if p_idx + 1 < len(phrases) else None
            phrase_is_before_pause = (
                pause_time is not None and phrase[0].start < (pause_time - 0.01)
            )
            phrase_is_after_pause = (
                pause_time is not None and not phrase_is_before_pause
            )

            for idx, active_word in enumerate(phrase):
                start_ts = active_word.start
                if idx + 1 < len(phrase):
                    end_ts = max(active_word.end, phrase[idx + 1].start)
                else:
                    end_ts = active_word.end + 0.14
                    if next_phrase_start is not None:
                        end_ts = min(end_ts, max(active_word.end, next_phrase_start - 0.02))

                if pause_time is not None:
                    if phrase_is_before_pause:
                        max_allowed_end = max(0.05, pause_time - pause_pad)
                        if start_ts >= max_allowed_end:
                            continue
                        end_ts = min(end_ts, max_allowed_end)
                    elif phrase_is_after_pause:
                        min_allowed_start = pause_time + pause_pad
                        if start_ts < min_allowed_start:
                            start_ts = min_allowed_start
                        end_ts = max(end_ts, start_ts + 0.08)

                if end_ts <= start_ts + 0.02:
                    continue

                start_str = self._format_ass_time(start_ts)
                end_str = self._format_ass_time(end_ts)

                rendered_words = []
                for j, w in enumerate(phrase):
                    safe_word = (
                        w.word.replace("\\", "")
                        .replace("{", "")
                        .replace("}", "")
                        .upper()
                    )
                    is_emo = self._is_emotional_word(w.word)
                    if j == idx:
                        if is_emo:
                            # Эмоциональный триггер / крик / мат: ярко-красный (&H3838FF в BGR) + мощный Pop-In!
                            rendered_words.append(
                                f"{{\\c&H3838FF&\\fscx85\\fscy85\\t(0,85,\\fscx124\\fscy124)\\t(85,175,\\fscx114\\fscy114)}}"
                                f"{safe_word}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
                            )
                        else:
                            # Обычное активное слово: ярко-желтый (&H00FFFF в BGR) + плавный Pop-In!
                            rendered_words.append(
                                f"{{\\c&H00FFFF&\\fscx88\\fscy88\\t(0,80,\\fscx116\\fscy116)\\t(80,165,\\fscx108\\fscy108)}}"
                                f"{safe_word}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
                            )
                    else:
                        rendered_words.append(safe_word)

                dialogue_text = " ".join(rendered_words)
                lines.append(
                    f"Dialogue: 1,{start_str},{end_str},TikTok,,0,0,0,,{dialogue_text}"
                )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        total_cs = int(round(max(0.0, seconds) * 100))
        hours, rem = divmod(total_cs, 360000)
        minutes, rem = divmod(rem, 6000)
        secs, centisecs = divmod(rem, 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
