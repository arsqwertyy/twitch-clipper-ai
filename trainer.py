import math
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests

from chat_monitor import (
    LAUGH_OR_HYPE_RE,
    TwitchChatMonitor,
    _median,
    is_ignored_chat_message,
    normalize_channel,
)
from face_detector import FaceWebcamDetector
from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path, get_streamlink_cmd
from video_clipper import VideoClipper

TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_PUBLIC_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

STOP_WORDS = {
    # Русские служебные, местоимения, частые разговорные слова
    "это", "как", "что", "так", "все", "всё", "еще", "ещё", "уже", "тут", "там", "вот",
    "просто", "если", "когда", "только", "даже", "меня", "тебя", "него", "было", "будет",
    "очень", "сейчас", "почему", "зачем", "какой", "какая", "какие", "кто", "где", "или", "для",
    "всем", "привет", "спасибо", "она", "оно", "они", "мне", "тебе", "нам", "вам", "них",
    "его", "ему", "ней", "ими", "вас", "нас", "иди", "был", "была", "были", "без", "про",
    "чтобы", "чтоб", "есть", "буду", "будут", "будем", "всех", "кого", "чего", "этом", "этой",
    "этого", "этим", "этих", "наши", "наш", "наша", "ваш", "ваша", "твою", "твой", "твоя",
    "мой", "моя", "мои", "свой", "свои", "себя", "себе", "раз", "два", "три", "одни", "один",
    "одна", "одно", "много", "мало", "сколько", "значит", "норм", "типа", "щас", "ток", "угу",
    "реально", "сильно", "слишком", "постоянно", "кажется", "сказать", "сделать", "сделай",
    "делает", "делать", "делай", "попробуй", "момент", "карту", "карта",
    "смотрю", "смотреть", "видела", "видел", "увидишь", "бери", "добавь",
    "место", "конечно", "здесь", "плиз", "пожалуйста", "подарите", "подари", "лучше", "хуже",
    "меньше", "больше", "вебку", "вебка", "вебке", "онлайн", "история", "экран", "звук", "микро",
    "года", "год", "день", "время", "люди", "чел", "чувак", "ребят", "пацаны", "брат",
    "новый", "новая", "новые", "этот", "эта", "эти", "того", "тоже", "либо", "нибудь",
    "можно", "нужно", "надо", "нет", "давай", "стоит", "хочу", "могу", "знаю", "думаю",
    "вообще", "кстати", "короче", "ладно", "понял", "пока", "сегодня", "вчера", "завтра",
    "стрим", "стриме", "стример", "стримить", "стримит", "чат", "чате", "канал", "видео",
    "игра", "игре", "донат", "ролик", "проект", "аудитория", "строчек", "истории", "попыток",
    "первое", "второе", "россии", "занял", "заходил", "похожи", "жалко", "факт",
    # Английские служебные и системные слова Twitch (подписки/гифты/ссылки)
    "the", "and", "you", "that", "this", "for", "with", "not", "are", "was", "have",
    "from", "what", "when", "your", "just", "like", "good", "http", "https", "www", "com",
    "twitch", "youtube", "watched", "consecutive", "streams", "sparked", "watch", "streak",
    "subscribed", "sub", "subs", "gifted", "gifting", "community", "tier", "total", "channel",
    "prime", "months", "month", "resub", "prime", "anon", "anonymous",
}


@dataclass
class TrainingReport:
    channel: str
    source: str  # "vod", "followers_estimate"
    followers: int = 0
    is_live_now: bool = False
    analyzed_vods: int = 0
    analyzed_messages: int = 0
    analyzed_windows: int = 0
    baseline_mean: float = 5.0
    baseline_std: float = 2.5
    z_score_threshold: float = 2.2
    min_messages_in_window: int = 6
    min_unique_chatters: int = 5
    min_hype_score: float = 12.0
    custom_keywords: list[str] = field(default_factory=list)
    best_vod_id: Optional[str] = None
    best_vod_title: str = ""
    best_offset_sec: int = 0
    best_peak_score: float = 0.0
    best_sample_msg: str = ""
    webcam_crop: Optional[dict] = None
    test_clip_path: Optional[Path] = None

    def to_profile_dict(self) -> dict:
        d = {
            "source": self.source,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "followers": self.followers,
            "analyzed_messages": self.analyzed_messages,
            "analyzed_windows": self.analyzed_windows,
            "baseline_mean": round(self.baseline_mean, 2),
            "baseline_std": round(self.baseline_std, 2),
            "z_score_threshold": round(self.z_score_threshold, 2),
            "min_messages_in_window": int(self.min_messages_in_window),
            "min_unique_chatters": int(self.min_unique_chatters),
            "min_hype_score": round(self.min_hype_score, 1),
            "custom_keywords": list(self.custom_keywords),
        }
        if self.webcam_crop:
            d["webcam_crop"] = self.webcam_crop
        return d


class AutoTrainer:
    """
    Глубокое ИИ-обучение (калибровка) и тестирование канала перед стримом:
    1. Скачивает тысячи реальных сообщений зрителей из последних 1-2 стримов (VOD) через GQL.
    2. Строит статистическое распределение скорости чата, подбирает индивидуальные пороги
       (Z-score, мин. сообщений, мин. зрителей, мин. балл хайпа) и выявляет взрывные эмоуты/сленг.
    3. Сканирует видеопоток VOD в нескольких точках эфира, чтобы точно определить координаты
       вебкамеры стримера (даже если в какой-то момент он выходил из-за стола).
    4. Вырезает полноценный 30-секундный тестовый TikTok-клип (9:16 + сплит вебки + субтитры Whisper).
    """

    def __init__(self, base_dir: Path, cfg: dict):
        self.base_dir = base_dir
        self.cfg = cfg
        self.detector_cfg = cfg.get("detector", {})
        self.face_detector = FaceWebcamDetector()
        self._last_detected_webcam: Optional[dict] = None

    def train_and_test_channel(
        self,
        channel: str,
        clipper: Optional[VideoClipper] = None,
        make_test_clip: bool = True,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> TrainingReport:
        channel = normalize_channel(channel)
        self._last_detected_webcam = None

        def log(msg: str) -> None:
            if log_cb:
                log_cb(f"[@{channel}] {msg}")

        log("Шаг 1/4: Запрос информации о канале и последних записях стримов (VOD)...")
        info = self._fetch_channel_info(channel)
        followers = info.get("followers", 0)
        is_live = info.get("is_live", False)
        videos = info.get("videos", [])

        report = TrainingReport(
            channel=channel,
            source="followers_estimate",
            followers=followers,
            is_live_now=is_live,
        )

        scorer = TwitchChatMonitor(channel, self.detector_cfg, lambda _e: None)

        # Собираем равномерные ПОЛНЫЕ 10-секундные окна из самого свежего длинного VOD
        # (второй VOD подключаем только если в первом не хватило точек выборки — чтобы
        # старый тихий стрим не занижал норму актуального онлайна).
        all_windows: list[tuple[str, int, int, int, float, dict[str, set[str]], str, float]] = []
        used_vods = 0

        for idx, vod in enumerate(videos[:2]):
            if used_vods >= 1 and len(all_windows) >= 16:
                break
            vod_id = vod["id"]
            length_sec = int(vod.get("lengthSeconds") or 0)
            if length_sec < 300:
                continue
            samples_cnt = 26 if idx == 0 else 14
            log(
                f"Шаг 2/4: Анализ полных 10с окон чата VOD #{vod_id} («{vod.get('title', '')[:38]}», "
                f"{length_sec // 60} мин, точек выборки: {samples_cnt})..."
            )
            w_list = self._sample_vod_chat_windows(vod_id, length_sec, scorer, num_samples=samples_cnt)
            if w_list:
                used_vods += 1
                for item in w_list:
                    all_windows.append((vod_id, *item))
                if report.best_vod_id is None:
                    report.best_vod_id = vod_id
                    report.best_vod_title = vod.get("title", "")

        if len(all_windows) >= 8:
            report.source = "vod"
            report.analyzed_vods = used_vods
            self._calibrate_from_windows(report, all_windows, scorer)
            log(
                f"📊 Статистика обучена по {report.analyzed_messages} сообщениям ({report.analyzed_windows} полных 10с окон из {used_vods} VOD)!"
            )
            log(
                f"⚙️ Ориентир VOD: Норма чата={report.baseline_mean:.1f} очк/10с (±{report.baseline_std:.1f}) | "
                f"Порог Z-score={report.z_score_threshold:.2f} | Мин. сообщ={report.min_messages_in_window} | "
                f"Мин. зрителей={report.min_unique_chatters}"
            )
            if report.custom_keywords:
                log(f"🔥 Выявлены взрывные слова/эмоуты канала: {', '.join(report.custom_keywords)}")
        else:
            self._calibrate_from_followers(report, followers)
            log(
                f"Публичных VOD с чатом не найдено (фолловеров: {followers:,}). "
                f"Установлены стартовые пороги: норма={report.baseline_mean:.1f}, Z={report.z_score_threshold:.2f}, "
                f"мин. сообщ={report.min_messages_in_window}."
            )

        # Шаг 3: Поиск вебкамеры в видеопотоке VOD (выполняется всегда, если есть VOD)
        if report.best_vod_id and report.best_offset_sec > 0:
            self._probe_vod_webcam(
                channel=channel,
                vod_id=report.best_vod_id,
                peak_offset_sec=report.best_offset_sec,
                log=log,
            )
            report.webcam_crop = self._last_detected_webcam

        # Шаг 4: Нарезка полноценного 30-секундного тестового клипа (если включена)
        if make_test_clip and clipper is not None and report.best_vod_id and report.best_offset_sec > 0:
            log(
                f"Шаг 4/4: Вырезаем 30-секундный хайлайт из VOD #{report.best_vod_id} "
                f"(таймкод {report.best_offset_sec // 60:02d}:{report.best_offset_sec % 60:02d}, "
                f"пик хайпа: {report.best_peak_score:.1f})..."
            )
            report.test_clip_path = self._cut_test_clip_from_vod(
                channel=channel,
                vod_id=report.best_vod_id,
                peak_offset_sec=report.best_offset_sec,
                peak_score=report.best_peak_score,
                clipper=clipper,
                log=log,
            )

        return report

    def _fetch_channel_info(self, channel: str) -> dict:
        query = """
        query($login: String!) {
          user(login: $login) {
            id
            login
            followers { totalCount }
            stream { id viewersCount }
            videos(first: 4, type: ARCHIVE) {
              edges {
                node { id title lengthSeconds }
              }
            }
          }
        }
        """
        try:
            r = requests.post(
                TWITCH_GQL_URL,
                headers={"Client-Id": TWITCH_PUBLIC_CLIENT_ID},
                json={"query": query, "variables": {"login": channel}},
                timeout=12,
            )
            data = r.json()
            user = (data.get("data") or {}).get("user") or {}
            followers = int(((user.get("followers") or {}).get("totalCount")) or 0)
            is_live = bool(user.get("stream"))
            edges = ((user.get("videos") or {}).get("edges")) or []
            videos = [e["node"] for e in edges if e and e.get("node")]
            return {"followers": followers, "is_live": is_live, "videos": videos}
        except Exception:
            return {"followers": 0, "is_live": False, "videos": []}

    def _fetch_single_vod_window(
        self,
        vod_id: str,
        target_off: int,
        window_sec: float,
        scorer: TwitchChatMonitor,
    ) -> Optional[tuple[int, int, int, float, dict[str, set[str]], str, float]]:
        """
        Скачивает РОВНО ОДНО честное, не обрезанное по краям окно длительностью window_sec (10с)
        вокруг таймкода target_off. Для быстрых чатов (где 1 страница GQL покрывает < 10с)
        догружает 1-3 соседних чанка по contentOffsetSeconds и объединяет их по ID сообщения.
        """
        q_comments = """
        query($videoID: ID!, $contentOffsetSeconds: Int) {
          video(id: $videoID) {
            comments(contentOffsetSeconds: $contentOffsetSeconds) {
              edges {
                node {
                  id
                  createdAt
                  contentOffsetSeconds
                  commenter { login }
                  message { fragments { text } }
                }
              }
            }
          }
        }
        """
        seen: dict[str, tuple[float, int, str, str, float]] = {}
        queried_off = int(target_off)

        session = requests.Session()
        try:
            def fetch_page(off: int) -> int:
                r = session.post(
                    TWITCH_GQL_URL,
                    headers={"Client-Id": TWITCH_PUBLIC_CLIENT_ID},
                    json={"query": q_comments, "variables": {"videoID": vod_id, "contentOffsetSeconds": int(off)}},
                    timeout=8,
                )
                edges = ((((r.json().get("data") or {}).get("video") or {}).get("comments") or {}).get("edges")) or []
                added = 0
                for idx, edge in enumerate(edges):
                    node = edge.get("node") or {}
                    c_off = int(node.get("contentOffsetSeconds") or 0)
                    commenter = ((node.get("commenter") or {}).get("login")) or "anon"
                    frags = ((node.get("message") or {}).get("fragments")) or []
                    text = "".join(str(f.get("text") or "") for f in frags).strip()
                    if is_ignored_chat_message(commenter, text):
                        continue
                    created_at = node.get("createdAt")
                    ts: Optional[float] = None
                    if created_at:
                        try:
                            ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).timestamp()
                        except ValueError:
                            ts = None
                    if ts is None:
                        ts = float(c_off)
                    msg_id = str(node.get("id") or f"{c_off}_{commenter}_{idx}_{text[:12]}")
                    if msg_id not in seen:
                        seen[msg_id] = (ts, c_off, commenter, text, scorer.score_message(text))
                        added += 1
                return added

            try:
                fetch_page(queried_off)
            except Exception:
                return None

            if not seen:
                return None

            items = sorted(seen.values(), key=lambda x: x[0])
            # Для быстрого чата (1 страница GQL = 2-8 сек) догружаем смежные чанки до полных 10с
            for _ in range(3):
                span = items[-1][0] - items[0][0]
                if span >= window_sec:
                    break
                next_off = max(items[-1][1] + 1, queried_off + 1)
                queried_off = next_off
                try:
                    added = fetch_page(queried_off)
                except Exception:
                    break
                if added == 0:
                    queried_off += 2
                    try:
                        added = fetch_page(queried_off)
                    except Exception:
                        break
                    if added == 0:
                        break
                items = sorted(seen.values(), key=lambda x: x[0])
        finally:
            session.close()

        span = items[-1][0] - items[0][0] if len(items) >= 2 else 0.0
        scale = 1.0
        if span >= window_sec:
            # Берем строго полное окно window_sec внутри [t_first, t_last] — ноль обрезки краев
            # и одинаковый статистический вес (ровно 1 окно на каждую точку выборки)!
            center = 0.5 * (items[0][0] + items[-1][0])
            w_start = center - 0.5 * window_sec
            w_end = w_start + window_sec
            win_items = [x for x in items if w_start <= x[0] < w_end]
            win_offset = int(win_items[0][1]) if win_items else int(target_off)
        elif span >= 0.5 and len(items) >= 3:
            # Сверхбыстрый чат (>~200 сообщ./10с даже после 4 страниц): честный пересчет скорости на 10с
            win_items = items
            scale = (window_sec / span) * ((len(items) - 1) / len(items))
            win_offset = int(items[0][1])
        else:
            win_items = items
            win_offset = int(items[0][1])

        per_user: Counter[str] = Counter()
        counted: list[tuple[float, int, str, str, float]] = []
        token_users: dict[str, set[str]] = {}

        for item in win_items:
            _, _, u, t, _sc = item
            per_user[u] += 1
            if per_user[u] <= scorer.MAX_MSGS_PER_USER_PER_WINDOW:
                counted.append(item)
                for tok in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,20}", t.lower()):
                    if (
                        tok not in STOP_WORDS
                        and tok != scorer.channel
                        and not tok.isdigit()
                        and not tok[0].isdigit()
                        and not LAUGH_OR_HYPE_RE.fullmatch(tok)
                    ):
                        token_users.setdefault(tok, set()).add(u)

        if not counted:
            return (win_offset, 0, 0, 0.0, {}, "", 0.0)

        raw_msg_count = len(counted)
        raw_unique = len(per_user)
        raw_hype = sum(x[4] for x in counted)
        react_ratio = sum(1 for x in counted if x[4] >= 1.5) / max(1, raw_msg_count)

        if scale > 1.0:
            msg_count = max(raw_msg_count, int(round(raw_msg_count * scale)))
            unique_users = max(raw_unique, int(round(raw_unique * scale)))
            hype_score = round(raw_hype * scale, 2)
        else:
            msg_count = raw_msg_count
            unique_users = raw_unique
            hype_score = round(raw_hype, 2)

        # Оценка интересности окна для выбора лучшего тестового момента (отдаём приоритет эмоциям, а не опросам)
        interest_rank = round(hype_score * (0.50 + 1.40 * react_ratio), 2)
        top_item = max(counted, key=lambda x: x[4])
        top_msg = f"{top_item[2]}: {top_item[3]}"
        return (win_offset, msg_count, unique_users, hype_score, token_users, top_msg, interest_rank)

    def _sample_vod_chat_windows(
        self,
        vod_id: str,
        length_sec: int,
        scorer: TwitchChatMonitor,
        num_samples: int = 26,
    ) -> list[tuple[int, int, int, float, dict[str, set[str]], str, float]]:
        window_sec = float(scorer.window_seconds)
        start_min = max(120, int(length_sec * 0.06))
        end_max = max(start_min + 60, int(length_sec * 0.92))
        step = max(45, (end_max - start_min) // max(1, num_samples))
        sample_offsets = [start_min + i * step for i in range(num_samples)]

        results: list[tuple[int, int, int, float, dict[str, set[str]], str, float]] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(self._fetch_single_vod_window, vod_id, off, window_sec, scorer)
                for off in sample_offsets
            ]
            for fut in futures:
                try:
                    res = fut.result()
                    if res is not None:
                        results.append(res)
                except Exception:
                    continue

        results.sort(key=lambda x: x[0])
        return results

    def _calibrate_from_windows(
        self,
        report: TrainingReport,
        windows: list[tuple[str, int, int, int, float, dict[str, set[str]], str, float]],
        scorer: TwitchChatMonitor,
    ) -> None:
        report.analyzed_windows = len(windows)
        report.analyzed_messages = sum(w[2] for w in windows)

        scores = [w[4] for w in windows]
        msgs = [w[2] for w in windows]
        users = [w[3] for w in windows]
        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        # Робастная статистика (медиана + MAD) — совпадает с логикой живого детектора!
        med_score = _median(scores)
        mad_score = _median([abs(x - med_score) for x in scores])
        std_val = max(3.0, 1.4826 * mad_score, med_score * 0.35)

        med_msgs = _median(msgs)
        med_users = _median(users)

        report.baseline_mean = max(2.0, med_score)
        report.baseline_std = std_val
        report.min_messages_in_window = max(6, int(round(med_msgs * 1.35)))
        report.min_unique_chatters = max(5, int(round(med_users * 1.25)))

        configured_z = float(self.detector_cfg.get("z_score_threshold", 2.6))
        p93_score = sorted_scores[min(n - 1, int(n * 0.93))]
        raw_z = (p93_score - report.baseline_mean) / report.baseline_std
        # Сохраняем адекватный, не слишком мягкий порог (2.35 .. 3.2), ориентируясь на настройку пользователя
        recommended_z = max(2.35, min(3.2, round(raw_z, 2) if raw_z >= 2.35 else configured_z))
        report.z_score_threshold = round(max(configured_z, recommended_z) if configured_z >= 2.35 else recommended_z, 2)
        report.min_hype_score = max(14.0, round(report.baseline_mean * 1.85, 1))

        # Сравнение частоты слов во взрывных окнах (топ 25%) против спокойных окон (нижние 75%),
        # с жестким требованием: слово должно встречаться минимум в 3 разных взрывах и от >= 3 уникальных зрителей!
        sorted_by_score = sorted(windows, key=lambda w: w[7] if len(w) > 7 else w[4], reverse=True)
        top_cut = max(4, n // 4)
        burst_wins = sorted_by_score[:top_cut]
        calm_wins = sorted_by_score[top_cut:]

        burst_win_counts: Counter[str] = Counter()
        burst_unique_users: dict[str, set[str]] = {}
        calm_win_counts: Counter[str] = Counter()

        for w in burst_wins:
            tok_map = w[5]
            for tok, u_set in tok_map.items():
                burst_win_counts[tok] += 1
                burst_unique_users.setdefault(tok, set()).update(u_set)
        for w in calm_wins:
            for tok in w[5].keys():
                calm_win_counts[tok] += 1

        existing_kws = scorer.word_keywords | set(scorer.substr_keywords)
        min_burst_wins = max(3, int(math.ceil(len(burst_wins) * 0.30)))
        candidates: list[tuple[float, str]] = []
        for tok, b_cnt in burst_win_counts.items():
            if b_cnt < min_burst_wins or tok in existing_kws or tok == report.channel:
                continue
            if len(burst_unique_users.get(tok, ())) < 3:
                continue
            c_cnt = calm_win_counts.get(tok, 0)
            lift = (b_cnt / max(1, len(burst_wins))) / ((c_cnt + 0.8) / max(1, len(calm_wins)))
            if lift >= 2.2:
                candidates.append((lift * b_cnt, tok))

        candidates.sort(reverse=True)
        report.custom_keywords = [tok for _, tok in candidates[:6]]

        # Выбираем самый эмоционально сильный момент из первого (самого свежего) VOD для тестовой нарезки
        first_vod_wins = [w for w in windows if w[0] == report.best_vod_id] or windows
        best_win = max(first_vod_wins, key=lambda w: w[7] if len(w) > 7 else w[4])
        report.best_vod_id = best_win[0]
        report.best_offset_sec = best_win[1]
        report.best_peak_score = best_win[4]
        report.best_sample_msg = best_win[6]

    @staticmethod
    def _calibrate_from_followers(report: TrainingReport, followers: int) -> None:
        if followers >= 500_000:
            report.baseline_mean = 24.0
            report.baseline_std = 9.0
            report.min_messages_in_window = 12
            report.min_unique_chatters = 9
            report.min_hype_score = 32.0
            report.z_score_threshold = 2.6
        elif followers >= 50_000:
            report.baseline_mean = 10.0
            report.baseline_std = 4.5
            report.min_messages_in_window = 7
            report.min_unique_chatters = 5
            report.min_hype_score = 16.0
            report.z_score_threshold = 2.5
        else:
            report.baseline_mean = 4.0
            report.baseline_std = 2.5
            report.min_messages_in_window = 6
            report.min_unique_chatters = 5
            report.min_hype_score = 14.0
            report.z_score_threshold = 2.4

    def _probe_vod_webcam(
        self,
        channel: str,
        vod_id: str,
        peak_offset_sec: int,
        log: Callable[[str], None],
    ) -> None:
        """
        Проверяет до 3 точек видеопотока VOD, чтобы гарантированно найти угловую вебкамеру стримера
        (даже если в момент пика чата стример отошел от кресла или открыл полноэкранную сцену).
        """
        log("Шаг 3/4: Сканирование видеопотока для точного определения координат вебкамеры...")
        vod_url = f"https://www.twitch.tv/videos/{vod_id}"
        quality = str(self.cfg.get("stream", {}).get("quality", "best"))
        q_arg = f"{quality},best" if quality != "best" else "best"

        probe_dir = self.base_dir / self.cfg.get("paths", {}).get("buffer_dir", "buffer") / f"_probe_{channel}"
        try:
            res = subprocess.run(
                [*get_streamlink_cmd(), "--stream-url", vod_url, q_arg],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=WIN_NO_WINDOW,
            )
            m3u8_url = res.stdout.strip()
            if res.returncode != 0 or not m3u8_url.startswith("http"):
                return

            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_ts = probe_dir / "probe.ts"

            # Проверяем сначала точку хайлайта, а если там нет угловой вебки — еще 2 точки эфира
            candidate_offsets = [
                max(60, peak_offset_sec - 10),
                max(300, int(peak_offset_sec * 0.65)),
                max(600, int(peak_offset_sec * 0.35)),
            ]

            for off in candidate_offsets:
                probe_ts.unlink(missing_ok=True)
                subprocess.run(
                    [
                        get_ffmpeg_path(),
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(off),
                        "-i",
                        m3u8_url,
                        "-t",
                        "4",
                        "-an",
                        "-c:v",
                        "copy",
                        "-f",
                        "mpegts",
                        str(probe_ts),
                    ],
                    capture_output=True,
                    timeout=25,
                    creationflags=WIN_NO_WINDOW,
                )
                if not probe_ts.exists():
                    continue
                cam = self.face_detector.detect_webcam_for_video(probe_ts, channel=channel)
                probe_ts.unlink(missing_ok=True)
                if cam and not cam.is_fullscreen_face:
                    self._last_detected_webcam = cam.to_dict()
                    log(
                        f"🎯 Вебкамера стримера найдена и сохранена ({cam.preset_name}): "
                        f"x={cam.x:.2f}, y={cam.y:.2f}, ширина={cam.w:.2f}, высота={cam.h:.2f}"
                    )
                    return

            log("ℹ️ Угловая вебкамера не обнаружена (полноэкранная камера или стрим без вебки).")
        except Exception:
            pass
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def _cut_test_clip_from_vod(
        self,
        channel: str,
        vod_id: str,
        peak_offset_sec: int,
        peak_score: float,
        clipper: VideoClipper,
        log: Callable[[str], None],
    ) -> Optional[Path]:
        """Скачивает полноценные 30 секунд реального VOD в точке взрыва чата и собирает TikTok-клип с субтитрами."""
        vod_url = f"https://www.twitch.tv/videos/{vod_id}"
        quality = str(self.cfg.get("stream", {}).get("quality", "best"))
        q_arg = f"{quality},best" if quality != "best" else "best"
        test_buf = self.base_dir / self.cfg.get("paths", {}).get("buffer_dir", "buffer") / f"_test_{channel}"

        try:
            res = subprocess.run(
                [*get_streamlink_cmd(), "--stream-url", vod_url, q_arg],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=WIN_NO_WINDOW,
            )
            m3u8_url = res.stdout.strip()
            if res.returncode != 0 or not m3u8_url.startswith("http"):
                return None

            test_buf.mkdir(parents=True, exist_ok=True)
            for old in test_buf.glob("*.ts"):
                old.unlink(missing_ok=True)

            start_sec = max(0, peak_offset_sec - 18)
            seg_pattern = str(test_buf / "seg_%04d.ts")

            ff_cmd = [
                get_ffmpeg_path(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start_sec),
                "-i",
                m3u8_url,
                "-t",
                "30",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                "5",
                "-segment_format",
                "mpegts",
                "-reset_timestamps",
                "1",
                seg_pattern,
            ]
            dl_res = subprocess.run(
                ff_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=75,
                creationflags=WIN_NO_WINDOW,
            )
            segs = sorted(test_buf.glob("seg_*.ts"))
            if dl_res.returncode != 0 or not segs:
                return None

            log("30 секунд видео скачаны, запускаем распознавание речи Faster-Whisper и рендер TikTok 9:16...")
            clip_path, words_cnt, elapsed = clipper.create_clip_from_segments(
                segs, f"test_{channel}", peak_score, fallback_webcam_crop=self._last_detected_webcam
            )
            if clip_path:
                ban_info = getattr(clipper, "last_banner_pause_info", "")
                ban_txt = f" | 📢 Баннер: {ban_info}" if ban_info else ""
                log(f"✅ Тестовый 30-секундный клип готов за {elapsed:.1f}с: {clip_path.name} (слов Whisper: {words_cnt}){ban_txt}")
            return clip_path
        except Exception as exc:
            log(f"Не удалось скачать VOD для теста ({exc!r}).")
            return None
        finally:
            shutil.rmtree(test_buf, ignore_errors=True)
