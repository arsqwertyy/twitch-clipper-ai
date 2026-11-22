import asyncio
import math
import random
import re
import ssl
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from emote_manager import ChannelEmoteManager


@dataclass
class ClipEvent:
    """Событие найденного хайлайта для передачи в очередь нарезки."""
    start_ts: float
    end_ts: float
    peak_score: float
    z_score: float
    total_messages: int
    sample_messages: list[str] = field(default_factory=list)
    channel: str = ""
    moment_tag: str = "🔥 Хайп"
    reaction_ratio: float = 0.0
    reaction_z: float = 0.0
    viral_score: float = 0.0


@dataclass
class WindowStats:
    """Статистика чата за одно 10-секундное скользящее окно."""
    timestamp: float
    msg_count: int
    hype_score: float
    mean_score: float
    std_score: float
    z_score: float
    is_triggered: bool
    top_messages: list[str]
    channel: str = ""
    unique_users: int = 0
    audio_z: float = 0.0
    audio_db: float = -60.0
    reaction_ratio: float = 0.0
    reaction_z: float = 0.0
    moment_tag: str = ""
    viral_score: float = 0.0


def normalize_channel(channel: str) -> str:
    """'https://www.twitch.tv/Name/videos?x=1' / '#Name' / 'Name' -> 'name'."""
    s = channel.strip().split("?")[0].split("#")[0].rstrip("/")
    m = re.search(r"twitch\.tv/([^/]+)", s, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).lower()
        if candidate not in ("videos", "directory", "downloads", "p", "search"):
            return candidate
    if "/" in s:
        s = s.split("/")[-1]
    res = s.lstrip("#").lower()
    if res in ("videos", "clips", "about", "schedule") or res.isdigit():
        return ""
    return res


def parse_channels_list(raw: str | list[str]) -> list[str]:
    """Разбирает строку (через запятую/пробел/слипшиеся ссылки) или список каналов в уникальный список ников."""
    if isinstance(raw, str):
        # Если ссылка прилипла к нику без пробела (например: frametamer666https://www.twitch.tv/...)
        separated = re.sub(r"(?=https?://)", " ", raw.strip(), flags=re.IGNORECASE)
        parts = re.split(r"[,;\s]+", separated)
    else:
        parts = [str(x) for x in raw]
    result: list[str] = []
    for p in parts:
        ch = normalize_channel(p)
        if ch and ch not in ("", "www.twitch.tv", "twitch.tv") and ch not in result:
            result.append(ch)
    return result


def _median(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


IGNORED_BOTS: set[str] = {
    "nightbot",
    "streamelements",
    "moobot",
    "fossabot",
    "wizebot",
    "streamlabs",
    "soundalerts",
    "botrixoficial",
    "commanderroot",
    "restreambot",
    "pretzelrocks",
    "songlistbot",
    "kofistreambot",
}

LAUGH_OR_HYPE_RE = re.compile(
    r"(?:ахах|хаха|пхах|ъаъа|вхах|ыхах|ахха|хаах|пхаа|кекв|ору+|ржу+|угар\w*|kekw|omegalul|lulw|icant|\bw{3,}\b)",
    re.IGNORECASE,
)
CLIP_CALL_RE = re.compile(
    r"(?:\bклип\w*|\bclip\w*|\bнарезк\w*|в\s*тик\s*ток|в\s*шортс|\bзаливай\w*|\bвырезай\w*)",
    re.IGNORECASE,
)
LAUGH_CAT_RE = re.compile(
    r"(?:ахах|хаха|пхах|ъаъа|вхах|ыхах|ахха|хаах|пхаа|кекв|ору+|ржу+|угар\w*|разорвал\w*|разъеб\w*|kekw|omegalul|lulw|\blul\b|\bicant\b|\bxdd?\b|\){4,})",
    re.IGNORECASE,
)
SHOCK_CAT_RE = re.compile(
    r"(?:ебать|ебану\w*|ахуе\w*|охуе\w*|нихуя|пиздец|жесть|пипец|капец|охрене\w*|нихрена|ужас|кошмар|шо+к|wtf|wut|\bче+го+\b|\bчто+о+\b|\bкуда+\b|\bстой+\b|\bбля+\b|\bблять\b|\bсука\b|\bнахуй\b|\?{3,}|!{3,})",
    re.IGNORECASE,
)
HYPE_CAT_RE = re.compile(
    r"(?:сюда+|имба|легенд\w*|гений|прайм|завоз|шедевр|красав\w*|соло\b|разнес\w*|pogchamp|\bpog\b|\bez\b|\bгг\b|\bgg\b|\bw{3,}\b|огонь|кайф|топчик|мощно|жестко)",
    re.IGNORECASE,
)
CRINGE_CAT_RE = re.compile(
    r"(?:кринж\w*|осужда\w*|\bбан\b|забан\w*|руин\w*|тильт\w*|фейл\w*|слил\b|позор|monkas|notlikethis|\bbruh\b|\bбрух\b)",
    re.IGNORECASE,
)
ELONGATED_RE = re.compile(r"([аоуеяиыэю])\1{3,}|([aouey])\2{3,}", re.IGNORECASE)
PUNCT_BURST_RE = re.compile(r"(\){4,}|!{3,}|\?{3,})")
BOT_CMD_RE = re.compile(r"^![a-zA-Zа-яА-ЯёЁ0-9_]+")
SYSTEM_NOTICE_RE = re.compile(
    r"(watched\s+\d+\s+consecutive\s+streams|gifted\s+a\s+tier|gifting\s+\d+.*community|subscribed\s+(at\s+tier|with\s+prime))",
    re.IGNORECASE,
)
# Голосования / приветствия / плюсики, которые дают ложный скачок объема без интересного момента
POLL_DIGITS_RE = re.compile(r"^[+\-=0-9\s.,/]{1,6}$")
POLL_OR_GREETING_WORDS: set[str] = {
    "да", "нет", "не", "го", "гоу", "ку", "qq", "привет", "прив", "приветик",
    "здарова", "здорово", "хай", "йоу", "салют", "участвую", "плюс", "минус",
    "ок", "окей", "ага", "угу", "спс", "спасибо", "пока", "бб", "bb", "hi",
    "hello", "yes", "no", "plus", "minus", "добрый", "здравствуйте", "всем",
}

EMOTION_LABELS: dict[str, str] = {
    "clip": "✂️ Чат просит клип",
    "laugh": "😂 Смех",
    "shock": "😱 Шок / Разрыв",
    "hype": "🔥 Хайп / Клатч",
    "cringe": "😬 Фейл / Кринж",
    "custom": "⚡ Реакция чата",
    "audio": "🔊 Крик + Чат",
    "normal": "💬 Активность",
}

STEM_HYPE_KEYWORDS: set[str] = {
    "ахуе",
    "нихуя",
    "пиздец",
    "ебать",
    "сюда",
    "жесть",
    "кринж",
    "легенд",
}

DEFAULT_HYPE_KEYWORDS: list[str] = [
    "ахах",
    "хаха",
    "пхах",
    "ору",
    "кек",
    "клип",
    "clip",
    "kekw",
    "lul",
    "lulw",
    "omegalul",
    "icant",
    "pog",
    "pogchamp",
    "wtf",
    "ебать",
    "ахуе",
    "нихуя",
    "пиздец",
    "сюда",
    "жесть",
    "имба",
    "легенда",
    "кринж",
    "гг",
    "gg",
    "ez",
    "xd",
    "xdd",
    "???",
    "!!!",
]


def is_ignored_chat_message(user: str, text: str) -> bool:
    """Фильтрует ботов, !команды и системные уведомления о подписках/стриках."""
    if not text:
        return True
    if user and user.lower() in IGNORED_BOTS:
        return True
    if BOT_CMD_RE.match(text):
        return True
    if SYSTEM_NOTICE_RE.search(text):
        return True
    return False


class TwitchChatMonitor:
    """
    Анонимный читатель чата Twitch (через IRC без аккаунта и токена)
    и адаптивный детектор хайлайтов с робастной статистикой (медиана + MAD).
    """

    IRC_HOST = "irc.chat.twitch.tv"
    IRC_PORT_SSL = 6697
    MAX_MSGS_PER_USER_PER_WINDOW = 2

    def __init__(
        self,
        channel: str,
        detector_cfg: dict,
        on_clip_detected: Callable[[ClipEvent], None],
        on_window_stats: Optional[Callable[[WindowStats], None]] = None,
        learned_profile: Optional[dict] = None,
        get_audio_stats: Optional[Callable[[], tuple[float, float]]] = None,
        is_stream_live: Optional[Callable[[], bool]] = None,
    ):
        self.channel = normalize_channel(channel)
        self.cfg = detector_cfg
        self.profile = learned_profile or {}
        self.on_clip_detected = on_clip_detected
        self.on_window_stats = on_window_stats
        self.get_audio_stats = get_audio_stats
        self.is_stream_live = is_stream_live
        self.use_audio_peaks: bool = bool(detector_cfg.get("use_audio_peaks", True))

        self.window_seconds: float = float(detector_cfg.get("window_seconds", 10))
        self.history_windows: int = int(detector_cfg.get("history_windows", 30))
        self.warmup_windows: int = max(3, int(detector_cfg.get("warmup_windows", 6)))

        # Порог Z-score берем из настроек пользователя (ползунок в GUI)
        self.z_threshold: float = float(
            detector_cfg.get("z_score_threshold", self.profile.get("z_score_threshold", 2.45))
        )
        # Минимальные шумовые пороги: учитываем обученный профиль конкретного стримера,
        # чтобы средние/дневные стримы не блокировались завышенным глобальным порогом,
        # а для крупных эфиров пороги автоматически масштабируются от живой медианы чата!
        cfg_min_msgs = int(detector_cfg.get("min_messages_in_window", 10))
        prof_min_msgs = int(self.profile.get("min_messages_in_window", cfg_min_msgs))
        self.min_messages: int = max(4, min(cfg_min_msgs, prof_min_msgs))

        cfg_min_users = int(detector_cfg.get("min_unique_chatters", 8))
        prof_min_users = int(self.profile.get("min_unique_chatters", cfg_min_users))
        self.min_unique_chatters: int = max(3, min(cfg_min_users, prof_min_users))

        cfg_min_hype = float(detector_cfg.get("min_hype_score", 20.0))
        prof_min_hype = float(self.profile.get("min_hype_score", cfg_min_hype))
        self.min_hype_score: float = max(11.0, min(cfg_min_hype, prof_min_hype))

        self.profile_baseline_mean: float = float(self.profile.get("baseline_mean", 0.0))
        self.min_viral_score: float = max(0.0, float(detector_cfg.get("min_viral_score", 38.0)))

        self.pre_roll: float = float(detector_cfg.get("pre_roll_seconds", 14))
        self.post_roll: float = float(detector_cfg.get("post_roll_seconds", 6))
        self.max_clip_seconds: float = float(detector_cfg.get("max_clip_seconds", 37))
        self.cooldown_seconds: float = float(detector_cfg.get("cooldown_seconds", 120))

        # Объединяем базовые ключевые слова и индивидуальные эмоуты/сленг только этого канала
        base_kws = list(detector_cfg.get("hype_keywords", []) or DEFAULT_HYPE_KEYWORDS)
        custom_kws = list(self.profile.get("custom_keywords", []) or [])
        all_kws = base_kws + [k for k in custom_kws if k not in base_kws]

        self.word_keywords: set[str] = set()
        self.substr_keywords: list[str] = []
        for kw in all_kws:
            kw = str(kw).lower().strip()
            if not kw:
                continue
            # Все вариации смеха уже учитываются единым LAUGH_OR_HYPE_RE (без двойного начисления)
            if LAUGH_OR_HYPE_RE.fullmatch(kw):
                continue
            if not kw.isalnum() or kw in STEM_HYPE_KEYWORDS:
                if kw not in self.substr_keywords:
                    self.substr_keywords.append(kw)
            else:
                self.word_keywords.add(kw)

        # Подключаем глобальные и индивидуальные эмоуты 7TV / BTTV этого канала
        self._emote_mgr = ChannelEmoteManager(Path(__file__).resolve().parent)
        self.emote_lookup: dict[str, tuple[str, str, float, bool]] = self._emote_mgr.build_channel_emote_lookup(
            self.channel
        )

        self._current_bucket: list[tuple[float, str, str, float, str, bool]] = []
        self._rolling_msgs: deque[tuple[float, str, str, float, str, bool]] = deque()
        # Живую историю НЕ предзаполняем фейковыми значениями из старого VOD:
        # детектор калибруется по первым warmup_windows реального эфира и далее непрерывно адаптируется.
        self._score_history: deque[float] = deque(maxlen=self.history_windows)
        self._reaction_history: deque[float] = deque(maxlen=self.history_windows)
        self._reaction_cnt_history: deque[int] = deque(maxlen=self.history_windows)
        self._msg_history: deque[int] = deque(maxlen=self.history_windows)
        self._users_history: deque[int] = deque(maxlen=self.history_windows)

        self._active_highlight_start: Optional[float] = None
        self._active_last_hyped_ts: float = 0.0
        self._active_peak_score: float = 0.0
        self._active_peak_z: float = 0.0
        self._active_peak_reaction_z: float = 0.0
        self._active_peak_reaction_ratio: float = 0.0
        self._active_peak_viral: float = 0.0
        self._active_hyped_windows: int = 0
        self._active_moment_tag: str = "🔥 Хайп"
        self._active_msg_count: int = 0
        self._active_samples: list[str] = []
        self._last_clip_end_ts: float = 0.0
        self._tick_idx: int = 0

        self._bg_tasks: set[asyncio.Task] = set()
        self._running = False
        self.connected = False

    def reload_channel_emotes(self) -> int:
        """Перезагружает словарь эмоутов 7TV/BTTV канала из обновлённого кэша."""
        self._emote_mgr = ChannelEmoteManager(Path(__file__).resolve().parent)
        self.emote_lookup = self._emote_mgr.build_channel_emote_lookup(self.channel)
        return len(self.emote_lookup)

    def classify_and_score_message(self, text: str) -> tuple[float, str, bool]:
        """
        Классифицирует и оценивает сообщение чата:
        Возвращает (score, category, is_reaction).
        - Голосования ('1', '2', '+', 'да', 'нет') и приветствия ('ку', 'привет')
          получают пониженный вес 0.20 (category='noise'), чтобы не резать скучные опросы!
        - Реальные эмоциональные реакции ('laugh', 'shock', 'hype', 'cringe', 'clip')
          учитываются отдельно для расчёта reaction_ratio и reaction_z.
        """
        stripped = text.strip()
        lower = stripped.lower()
        if not lower:
            return 0.0, "noise", False

        # 1. Фильтрация голосований / плюсиков / односложных приветствий
        if POLL_DIGITS_RE.match(lower):
            return 0.20, "noise", False
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", lower)
        token_set = set(tokens)
        if tokens and len(tokens) <= 2 and all(t in POLL_OR_GREETING_WORDS for t in tokens):
            return 0.20, "noise", False

        # 1b. Проверяем эмоуты 7TV / BTTV канала и глобальные реакционные эмоуты
        emote_react_cats: list[str] = []
        emote_max_score = 0.0
        has_channel_custom_emote = False
        all_ambient_emotes = bool(tokens)

        for tok in token_set:
            em_info = self.emote_lookup.get(tok)
            if em_info is None:
                all_ambient_emotes = False
                continue
            _orig, em_cat, em_sc, em_is_react = em_info
            if em_cat == "ambient":
                continue
            all_ambient_emotes = False
            if em_is_react:
                emote_react_cats.append(em_cat)
                if em_sc > emote_max_score:
                    emote_max_score = em_sc
            elif em_cat == "channel_emote":
                has_channel_custom_emote = True

        # Если сообщение состоит ТОЛЬКО из фоновых/танцевальных эмоутов (catJAM, Waiting, Chatting),
        # понижаем его вес, чтобы музыка на паузе не вызывала ложный клип
        if all_ambient_emotes and tokens:
            return 0.45, "noise", False

        score = 1.0
        matched_cats: list[str] = []

        if CLIP_CALL_RE.search(lower):
            matched_cats.append("clip")
        if LAUGH_CAT_RE.search(lower):
            matched_cats.append("laugh")
        if SHOCK_CAT_RE.search(lower):
            matched_cats.append("shock")
        if HYPE_CAT_RE.search(lower):
            matched_cats.append("hype")
        if CRINGE_CAT_RE.search(lower):
            matched_cats.append("cringe")

        for ec in emote_react_cats:
            if ec not in matched_cats:
                matched_cats.append(ec)

        # Кастомные ключевые слова и локальные эмоуты 7TV/BTTV канала
        word_hits = sum(1 for kw in self.word_keywords if kw in token_set)
        substr_hits = sum(1 for kw in self.substr_keywords if kw in lower)
        if ((word_hits + substr_hits) > 0 or has_channel_custom_emote) and not matched_cats:
            matched_cats.append("custom")

        if "clip" in matched_cats:
            score += 1.1
        elif len(matched_cats) >= 2:
            score += 1.15
        elif len(matched_cats) == 1:
            cat_bonus = {
                "laugh": 0.85,
                "shock": 0.80,
                "hype": 0.75,
                "cringe": 0.70,
                "custom": 0.75,
            }.get(matched_cats[0], 0.70)
            score += cat_bonus

        if emote_max_score > 0.0:
            score = max(score, min(2.45, emote_max_score * 0.78))

        # Локальный 7TV-эмоут канала сам по себе даёт умеренный вес (1.75), но без флага прямой реакции
        # (если его начнут спамить >=4 зрителей одновременно, _process_window засчитает волну 7TV-эмоутов!)
        is_reaction = any(c != "custom" for c in matched_cats) or ((word_hits + substr_hits) > 0)

        if PUNCT_BURST_RE.search(lower):
            score += 0.30
            if not matched_cats:
                matched_cats.append("shock" if "?" in lower or "!" in lower else "laugh")
            is_reaction = True

        if ELONGATED_RE.search(lower):
            score += 0.25
            is_reaction = True

        letters = [ch for ch in stripped if ch.isalpha()]
        if len(letters) >= 5:
            upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
            if upper_ratio >= 0.75:
                score += 0.30
                if len(letters) >= 7:
                    is_reaction = True

        primary_cat = matched_cats[0] if matched_cats else ("hype" if is_reaction else "normal")
        return min(score, 2.6), primary_cat, is_reaction

    def score_message(self, text: str) -> float:
        """Оценивает 'хайповость' одного сообщения чата (0.2 .. 2.6)."""
        return self.classify_and_score_message(text)[0]

    async def run(self) -> None:
        self._running = True
        await asyncio.gather(self._irc_loop(), self._analyzer_loop())

    def stop(self) -> None:
        self._running = False

    async def _irc_loop(self) -> None:
        privmsg_re = re.compile(r"^:(?P<user>[^!]+)![^ ]+ PRIVMSG #\S+ :(?P<text>.*)$")

        while self._running:
            writer = None
            try:
                ssl_ctx = ssl.create_default_context()
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.IRC_HOST, self.IRC_PORT_SSL, ssl=ssl_ctx),
                    timeout=20.0,
                )
                anon_nick = f"justinfan{random.randint(10000, 99999)}"
                writer.write(b"PASS SCHMOOPIIE\r\n")
                writer.write(f"NICK {anon_nick}\r\n".encode("utf-8"))
                writer.write(f"JOIN #{self.channel}\r\n".encode("utf-8"))
                await writer.drain()
                self.connected = True

                while self._running:
                    try:
                        line_bytes = await asyncio.wait_for(reader.readline(), timeout=240.0)
                    except asyncio.TimeoutError:
                        writer.write(b"PING :tmi.twitch.tv\r\n")
                        await writer.drain()
                        line_bytes = await asyncio.wait_for(reader.readline(), timeout=30.0)

                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    if line.startswith("PING"):
                        writer.write((line.replace("PING", "PONG", 1) + "\r\n").encode("utf-8"))
                        await writer.drain()
                        continue

                    if " RECONNECT" in line and "PRIVMSG" not in line:
                        break

                    match = privmsg_re.match(line)
                    if match:
                        user = match.group("user")
                        text = match.group("text").strip()
                        if not is_ignored_chat_message(user, text):
                            sc, cat, is_react = self.classify_and_score_message(text)
                            self._current_bucket.append((time.time(), user, text, sc, cat, is_react))

            except asyncio.CancelledError:
                break
            except Exception:
                pass
            finally:
                self.connected = False
                if writer is not None:
                    try:
                        writer.close()
                        await asyncio.wait_for(writer.wait_closed(), timeout=3.0)
                    except Exception:
                        pass
            if self._running:
                await asyncio.sleep(5.0)

    def _safe_callback(self, cb: Optional[Callable], *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as exc:
            print(f"[chat_monitor:{self.channel}] Ошибка в обработчике: {exc!r}")

    def _reset_live_state(self) -> None:
        self._rolling_msgs.clear()
        self._score_history.clear()
        self._reaction_history.clear()
        self._reaction_cnt_history.clear()
        self._msg_history.clear()
        self._users_history.clear()
        self._active_highlight_start = None
        self._active_last_hyped_ts = 0.0
        self._active_peak_score = 0.0
        self._active_peak_z = 0.0
        self._active_peak_reaction_z = 0.0
        self._active_peak_reaction_ratio = 0.0
        self._active_peak_viral = 0.0
        self._active_hyped_windows = 0
        self._active_moment_tag = "🔥 Хайп"
        self._active_msg_count = 0
        self._active_samples = []
        self._tick_idx = 0

    @staticmethod
    def _robust_baseline(hist: deque[float], floor_std: float = 3.2, rel_std: float = 0.28) -> tuple[float, float]:
        """
        Возвращает (baseline, effective_std) по долгосрочной медиане + MAD.
        Не берёт max(long_med, recent_med) и исключает последнее окно при расчёте медианы,
        чтобы нарастающий 20-секундный хайлайт не завышал сам себе базовую норму!
        """
        n = len(hist)
        if n < 2:
            val = float(hist[0]) if n == 1 else 0.0
            return val, max(floor_std, val * rel_std)
        arr = list(hist)
        ref_arr = arr[:-1] if n >= 5 else arr
        base = _median(ref_arr)
        mad = _median([abs(x - base) for x in ref_arr])
        robust_std = 1.4826 * mad
        eff_std = max(robust_std, floor_std, base * rel_std)
        return base, eff_std

    def _process_window(
        self,
        bucket: list[tuple],
        now: float,
        audio_z: float,
        audio_db: float,
        update_history: bool = True,
    ) -> tuple[WindowStats, Optional[ClipEvent]]:
        """
        Мульти-сигнальный анализ 10-секундного окна живого эфира:
        1. Адаптивная робастная норма общего хайпа (chat_z).
        2. Отдельная робастная норма ЭМОЦИОНАЛЬНЫХ реакций (reaction_z + reaction_ratio) —
           отсекает скучные опросы ('1/2/+') и ловит смех/шок даже на 70k-онлайне!
        3. Прямой сигнал 'Чат просит клип' ('клип' / 'в тикток' / 'нарезку' от нескольких зрителей).
        4. Комбо-триггер 'Крик/смех стримера в голосовом диапазоне + эмоциональная реакция чата'.
        """
        window_cutoff = now - self.window_seconds - 0.5
        per_user: Counter[str] = Counter()
        counted: list[tuple[float, str, str, float, str, bool]] = []
        cat_users: dict[str, set[str]] = {}
        reaction_timestamps: list[float] = []
        emote_wave_users: dict[str, set[str]] = {}
        custom_emote_msg_indices: list[tuple[int, float, str, set[str]]] = []

        for item in bucket:
            ts = item[0]
            if ts < window_cutoff:
                continue
            user, text = item[1], item[2]
            if is_ignored_chat_message(user, text):
                continue
            if len(item) >= 6:
                sc, cat, is_react = float(item[3]), str(item[4]), bool(item[5])
            else:
                sc, cat, is_react = self.classify_and_score_message(text)

            per_user[user] += 1
            if per_user[user] <= self.MAX_MSGS_PER_USER_PER_WINDOW:
                msg_idx = len(counted)
                counted.append((ts, user, text, sc, cat, is_react))
                if self.emote_lookup:
                    msg_emotes: set[str] = set()
                    for tok in set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", text.lower())):
                        em_info = self.emote_lookup.get(tok)
                        if em_info is not None and em_info[1] != "ambient":
                            emote_wave_users.setdefault(tok, set()).add(user)
                            msg_emotes.add(tok)
                    if msg_emotes and not is_react:
                        custom_emote_msg_indices.append((msg_idx, ts, user, msg_emotes))
                if is_react:
                    cat_users.setdefault(cat, set()).add(user)
                    reaction_timestamps.append(ts)

        # Детектор волны 7TV/BTTV эмоутов: если >=4 разных зрителей в 10с окне одновременно
        # спамят один и тот же локальный 7TV-эмоут канала (например 'Kurica', 'WideSvin'),
        # засчитываем эти сообщения как подтверждённую эмоциональную волну чата!
        wave_emotes = {em for em, u_set in emote_wave_users.items() if len(u_set) >= 4}
        if wave_emotes and custom_emote_msg_indices:
            for msg_idx, ts, user, msg_emotes in custom_emote_msg_indices:
                if msg_emotes & wave_emotes:
                    old_ts, old_u, old_t, old_sc, old_cat, _ = counted[msg_idx]
                    counted[msg_idx] = (old_ts, old_u, old_t, max(old_sc, 2.05), "hype", True)
                    cat_users.setdefault("hype", set()).add(user)
                    reaction_timestamps.append(ts)

        msg_count = len(counted)
        unique_users = len(per_user)
        noise_count = sum(1 for x in counted if x[4] == "noise")
        non_noise_count = msg_count - noise_count
        non_noise_ratio = non_noise_count / max(1, msg_count)

        chat_hype_score = sum(x[3] for x in counted)
        reaction_count = len(reaction_timestamps)
        reaction_score = sum(x[3] for x in counted if x[5])
        reaction_ratio = reaction_count / max(1, msg_count)
        clip_call_users = len(cat_users.get("clip", set()))

        # Определяем доминирующую эмоцию в чате (смех / шок / хайп / кринж / запрос клипа)
        dominant_cat = "normal"
        consensus_ratio = 0.0
        if cat_users:
            ranked_cats = sorted(cat_users.items(), key=lambda kv: len(kv[1]), reverse=True)
            dominant_cat = ranked_cats[0][0]
            total_react_users = len({u for s in cat_users.values() for u in s})
            consensus_ratio = len(ranked_cats[0][1]) / max(1, total_react_users)

        n_hist = len(self._score_history)
        if n_hist >= 2:
            mean_score, effective_std = self._robust_baseline(self._score_history, floor_std=3.2, rel_std=0.28)
            # В первые 90 секунд после запуска мягко опираемся на обученную норму канала,
            # чтобы самый первый рядовой момент после старта не срабатывал ложно на пустой истории
            if n_hist < 9 and self.profile_baseline_mean > 0:
                soft_floor = min(self.profile_baseline_mean * 0.65, 22.0)
                if mean_score < soft_floor:
                    mean_score = soft_floor
                    effective_std = max(effective_std, mean_score * 0.30)
            chat_z = (chat_hype_score - mean_score) / effective_std
        elif n_hist == 1:
            mean_score = self._score_history[0]
            effective_std = max(3.5, mean_score * 0.30)
            chat_z = 0.0
        else:
            mean_score = chat_hype_score
            effective_std = max(3.5, chat_hype_score * 0.32)
            chat_z = 0.0

        # Отдельный Z-score по эмоциональным реакциям (смех / шок / хайп / кринж / клип)
        if len(self._reaction_history) >= 2:
            react_base, react_std = self._robust_baseline(self._reaction_history, floor_std=3.0, rel_std=0.32)
            reaction_z = (reaction_score - react_base) / react_std
        else:
            react_base = reaction_score
            reaction_z = 0.0

        # Ограниченный бонус за дружную единую эмоцию (консенсус толпы >= 60%) + голосовой пик стримера
        consensus_bonus = 0.0
        if reaction_count >= 8 and consensus_ratio >= 0.60 and reaction_ratio >= 0.30 and chat_z >= 1.8:
            consensus_bonus = 0.22 * effective_std

        audio_bonus = 0.0
        if (
            self.use_audio_peaks
            and audio_z >= 2.2
            and audio_db >= -18.0
            and (chat_z >= 1.5 or reaction_z >= 1.8)
            and non_noise_count >= max(6, self.min_messages // 2)
        ):
            audio_z_boost = min(audio_z - 1.8, 1.8) * 0.28
            audio_bonus = audio_z_boost * effective_std

        hype_score = chat_hype_score + audio_bonus + consensus_bonus
        z_score = (hype_score - mean_score) / effective_std if n_hist >= 2 else 0.0

        has_warmup = n_hist >= self.warmup_windows
        in_cooldown = now < (self._last_clip_end_ts + self.cooldown_seconds)

        # Адаптивные минимумы под текущий живой онлайн стрима (по долгосрочной медиане без последнего окна)
        if self._msg_history:
            msg_list = list(self._msg_history)
            live_med_msgs = _median(msg_list[:-1] if len(msg_list) >= 5 else msg_list)
        else:
            live_med_msgs = 0.0
        if self._users_history:
            usr_list = list(self._users_history)
            live_med_users = _median(usr_list[:-1] if len(usr_list) >= 5 else usr_list)
        else:
            live_med_users = 0.0
        if self._reaction_cnt_history:
            rc_list = list(self._reaction_cnt_history)
            live_med_react_cnt = _median(rc_list[:-1] if len(rc_list) >= 5 else rc_list)
        else:
            live_med_react_cnt = 0.0

        # Пороги отбора кандидатов для часового турнира «👑 Король часа»:
        # нарезаем все яркие моменты за час (5-12 кандидатов), чтобы ИИ выбрал из них №1 лучший!
        req_msgs = max(self.min_messages, int(round(live_med_msgs * 1.35)))
        req_users = max(self.min_unique_chatters, int(round(live_med_users * 1.25)))
        min_surge_ratio = max(1.75, 1.0 + 0.28 * self.z_threshold)
        req_hype = max(self.min_hype_score, mean_score * min_surge_ratio)

        # Анти-опрос / анти-приветствие: реальный момент обязан нести эмоциональную реакцию
        has_emotional_substance = (
            reaction_ratio >= 0.20
            or reaction_z >= 1.85
            or (chat_z >= (self.z_threshold + 1.2) and non_noise_ratio >= 0.65)
        )

        # Триггер 1: Взрыв чата с подтверждёнными эмоциями
        chat_spike = (
            non_noise_count >= req_msgs
            and unique_users >= req_users
            and hype_score >= req_hype
            and z_score >= self.z_threshold
            and has_emotional_substance
        )

        # Триггер 2: Массовый эмоциональный разрыв (взрыв смеха/шока/7TV эмоутов)
        req_react_cnt = max(6, self.min_messages, int(round(live_med_react_cnt * 1.90)))
        reaction_surge = (
            reaction_count >= req_react_cnt
            and unique_users >= max(5, self.min_unique_chatters, int(round(live_med_users * 1.15)))
            and reaction_ratio >= 0.35
            and reaction_z >= max(self.z_threshold + 0.15, 2.7)
            and chat_z >= max(1.3, self.z_threshold * 0.45)
            and reaction_score >= max(16.0, self.min_hype_score * 0.95, react_base * 2.1)
        )

        # Триггер 3: Зрители требуют «КЛИП / В ТИКТОК / НАРЕЗКУ» (от 3+ разных зрителей)
        req_clip_callers = max(3, min(14, int(round(live_med_users * 0.10)) + 2))
        clip_call_trigger = (
            clip_call_users >= req_clip_callers
            and non_noise_count >= max(7, self.min_messages)
            and reaction_ratio >= 0.22
            and (chat_z >= max(1.7, self.z_threshold * 0.60) or reaction_z >= max(2.0, self.z_threshold * 0.65))
        )

        # Триггер 4: Громкий крик/взрыв смеха стримера (audio_z >= 2.5 и громко >= -18 дБ) + реакция чата
        audio_chat_combo = (
            self.use_audio_peaks
            and audio_z >= 2.5
            and audio_db >= -18.0
            and chat_z >= max(1.6, self.z_threshold * 0.62)
            and reaction_z >= max(1.8, self.z_threshold * 0.65)
            and reaction_ratio >= 0.22
            and non_noise_count >= max(self.min_messages, int(round(live_med_msgs * 1.15)))
            and unique_users >= max(self.min_unique_chatters, int(round(live_med_users * 1.10)))
            and chat_hype_score >= max(15.0, self.min_hype_score * 0.95, mean_score * 1.60)
        )

        is_spike = has_warmup and not in_cooldown and (
            chat_spike or reaction_surge or clip_call_trigger or audio_chat_combo
        )

        # Определяем человекочитаемый тег причины и откалиброванный индекс виральности (0..100)
        if clip_call_trigger:
            moment_tag = EMOTION_LABELS["clip"]
        elif audio_chat_combo and not (chat_spike or reaction_surge):
            moment_tag = EMOTION_LABELS["audio"]
        else:
            moment_tag = EMOTION_LABELS.get(dominant_cat, "🔥 Хайп")

        surge_ratio = hype_score / max(mean_score, 8.0)
        viral_score = max(
            0.0,
            min(
                100.0,
                max(0.0, z_score - 1.5) * 11.0
                + max(0.0, reaction_z - 1.5) * 9.0
                + reaction_ratio * 24.0
                + min(18.0, max(0.0, surge_ratio - 1.5) * 8.0)
                + (14.0 if clip_call_trigger else min(8.0, clip_call_users * 2.0))
                + (min(10.0, max(0.0, audio_z - 2.0) * 5.0) if self.use_audio_peaks else 0.0),
            ),
        )

        sorted_msgs = sorted(counted, key=lambda x: x[3], reverse=True)
        top_messages = [f"{u}: {t}" for _, u, t, _, _, _ in sorted_msgs[:3]]

        emitted_event: Optional[ClipEvent] = None
        is_triggered_ui = False

        if self._active_highlight_start is None:
            if is_spike:
                is_triggered_ui = True
                # Привязываем старт клипа к реальному началу волны реакций внутри 10с окна (25-й перцентиль),
                # чтобы pre_roll максимально точно захватывал завязку момента!
                if len(reaction_timestamps) >= 3:
                    srt_ts = sorted(reaction_timestamps)
                    onset_ts = srt_ts[len(srt_ts) // 4]
                else:
                    onset_ts = now - self.window_seconds
                self._active_highlight_start = onset_ts - self.pre_roll
                self._active_last_hyped_ts = now
                self._active_peak_score = hype_score
                self._active_peak_z = max(z_score, reaction_z)
                self._active_peak_reaction_z = reaction_z
                self._active_peak_reaction_ratio = reaction_ratio
                self._active_peak_viral = viral_score
                self._active_hyped_windows = 1
                self._active_moment_tag = moment_tag
                self._active_msg_count = msg_count
                self._active_samples = top_messages[:3]
        else:
            next_duration = (now + self.post_roll) - self._active_highlight_start
            still_hyped = (
                non_noise_count >= self.min_messages
                and (
                    (hype_score >= mean_score * 1.65 and chat_z >= 1.6)
                    or (reaction_z >= 2.0 and reaction_ratio >= 0.30)
                )
            )

            if still_hyped and (next_duration <= self.max_clip_seconds):
                is_triggered_ui = True
                self._active_last_hyped_ts = now
                self._active_hyped_windows += 1
                self._active_peak_score = max(self._active_peak_score, hype_score)
                self._active_peak_z = max(self._active_peak_z, z_score, reaction_z)
                self._active_peak_reaction_z = max(self._active_peak_reaction_z, reaction_z)
                self._active_peak_reaction_ratio = max(self._active_peak_reaction_ratio, reaction_ratio)
                # Продолжительная волна реакций (2+ окна подряд) получает бонус к индексу виральности
                sustained_bonus = 10.0 if self._active_hyped_windows >= 2 else 0.0
                boosted_viral = min(100.0, viral_score + sustained_bonus)
                if boosted_viral > self._active_peak_viral:
                    self._active_peak_viral = boosted_viral
                    if moment_tag != EMOTION_LABELS["normal"]:
                        self._active_moment_tag = moment_tag
                self._active_msg_count += msg_count
                if len(self._active_samples) < 5:
                    self._active_samples.extend(top_messages[:2])
            else:
                end_anchor = now if still_hyped else self._active_last_hyped_ts
                clip_start = self._active_highlight_start
                clip_end = min(end_anchor + self.post_roll, clip_start + self.max_clip_seconds)
                final_viral = min(
                    100.0,
                    self._active_peak_viral + (8.0 if self._active_hyped_windows >= 2 else 0.0),
                )
                if final_viral >= self.min_viral_score:
                    emitted_event = ClipEvent(
                        start_ts=clip_start,
                        end_ts=clip_end,
                        peak_score=self._active_peak_score,
                        z_score=self._active_peak_z,
                        total_messages=self._active_msg_count,
                        sample_messages=list(self._active_samples),
                        channel=self.channel,
                        moment_tag=self._active_moment_tag,
                        reaction_ratio=round(self._active_peak_reaction_ratio, 2),
                        reaction_z=round(self._active_peak_reaction_z, 2),
                        viral_score=round(final_viral, 1),
                    )
                    self._last_clip_end_ts = clip_end
                else:
                    # Слабый единичный всплеск ниже планки качества — не тратим GPU/квоту YouTube
                    self._last_clip_end_ts = max(self._last_clip_end_ts, now - self.cooldown_seconds + 30.0)
                self._active_highlight_start = None
                self._active_last_hyped_ts = 0.0
                self._active_hyped_windows = 0

        # Обновляем скользящую статистически независимую историю (раз в 10с при шаге 5с)
        if update_history:
            self._score_history.append(chat_hype_score)
            self._reaction_history.append(reaction_score)
            self._reaction_cnt_history.append(reaction_count)
            self._msg_history.append(non_noise_count)
            self._users_history.append(unique_users)

        stats = WindowStats(
            timestamp=now,
            msg_count=msg_count,
            hype_score=hype_score,
            mean_score=mean_score,
            std_score=effective_std,
            z_score=max(z_score, reaction_z) if is_triggered_ui else z_score,
            is_triggered=is_triggered_ui,
            top_messages=top_messages,
            channel=self.channel,
            unique_users=unique_users,
            audio_z=audio_z,
            audio_db=audio_db,
            reaction_ratio=round(reaction_ratio, 2),
            reaction_z=round(reaction_z, 2),
            moment_tag=moment_tag,
            viral_score=round(viral_score, 1),
        )
        return stats, emitted_event

    async def _analyzer_loop(self) -> None:
        # Скользящее окно с шагом 5с (перекрытие 50% по 10с окну) —
        # больше не пропускает яркие всплески, попавшие на стык 10-секундных интервалов!
        step_sec = max(2.0, min(5.0, self.window_seconds / 2.0))
        ticks_per_window = max(1, int(round(self.window_seconds / step_sec)))

        while self._running:
            await asyncio.sleep(step_sec)
            if not self._running:
                break
            now = time.time()

            new_items = self._current_bucket
            self._current_bucket = []

            if self.is_stream_live is not None and not self.is_stream_live():
                if self._score_history or self._active_highlight_start is not None or self._rolling_msgs:
                    self._reset_live_state()
                continue

            for it in new_items:
                self._rolling_msgs.append(it)
            cutoff = now - self.window_seconds - 0.5
            while self._rolling_msgs and self._rolling_msgs[0][0] < cutoff:
                self._rolling_msgs.popleft()

            self._tick_idx += 1
            is_full_tick = (self._tick_idx % ticks_per_window) == 0

            audio_z, audio_db = 0.0, -60.0
            if self.use_audio_peaks and self.get_audio_stats is not None:
                try:
                    audio_z, audio_db = await asyncio.to_thread(self.get_audio_stats)
                except Exception:
                    audio_z, audio_db = 0.0, -60.0

            stats, event = self._process_window(
                list(self._rolling_msgs),
                now,
                audio_z,
                audio_db,
                update_history=is_full_tick,
            )
            # В UI и лог отдаем каждый 10с тик, а при срабатывании взрыва — мгновенно!
            if is_full_tick or stats.is_triggered or event is not None:
                self._safe_callback(self.on_window_stats, stats)

            if event is not None:
                delay = max(0.0, event.end_ts - time.time()) + 8.0
                task = asyncio.create_task(self._schedule_clip(event, delay=delay))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

    async def _schedule_clip(self, event: ClipEvent, delay: float) -> None:
        await asyncio.sleep(max(0.0, delay))
        self._safe_callback(self.on_clip_detected, event)
