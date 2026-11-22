import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import requests


TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_PUBLIC_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

# Глобальные и самые популярные 7TV / BTTV / FFZ / Twitch эмоуты по категориям реакций
GLOBAL_EMOTE_CATEGORIES: dict[str, tuple[str, float]] = {
    # 😂 Смех
    "xdd": ("laugh", 2.8),
    "xdding": ("laugh", 2.8),
    "xdx": ("laugh", 2.7),
    "xd": ("laugh", 2.4),
    "kekw": ("laugh", 2.8),
    "kekl": ("laugh", 2.8),
    "kekwait": ("laugh", 2.7),
    "kek": ("laugh", 2.5),
    "omegalul": ("laugh", 2.9),
    "lulw": ("laugh", 2.8),
    "lul": ("laugh", 2.6),
    "pepelaugh": ("laugh", 2.8),
    "peepogiggles": ("laugh", 2.6),
    "hehehehe": ("laugh", 2.7),
    "roflanebalo": ("laugh", 2.9),
    "roflanbuldiga": ("laugh", 2.8),
    "roflanpominki": ("laugh", 2.8),
    "icant": ("laugh", 2.9),
    "porvalo": ("laugh", 2.9),
    "troll": ("laugh", 2.5),
    "tf": ("laugh", 2.4),
    "clown": ("laugh", 2.5),
    "pepeclown": ("laugh", 2.6),
    "catlaugh": ("laugh", 2.8),
    "classic": ("laugh", 2.5),
    "goodone": ("laugh", 2.6),
    "pointless": ("laugh", 2.4),
    # 😱 Шок / Разрыв
    "aintnoway": ("shock", 3.1),
    "caught": ("shock", 3.0),
    "huh": ("shock", 2.9),
    "xylihuh": ("shock", 3.0),
    "alert": ("shock", 2.9),
    "monkas": ("shock", 2.8),
    "monkaw": ("shock", 2.8),
    "monkaeyes": ("shock", 2.7),
    "monkagiga": ("shock", 2.9),
    "wutface": ("shock", 2.8),
    "wtf": ("shock", 2.8),
    "booba": ("shock", 2.8),
    "flushed": ("shock", 2.6),
    "flushedcat": ("shock", 2.7),
    "awkwardflushed": ("shock", 2.7),
    "susge": ("shock", 2.5),
    "trolldespair": ("shock", 2.7),
    "despair": ("shock", 2.6),
    "blubbers": ("shock", 2.7),
    "ppcrazy": ("shock", 2.8),
    "durka": ("shock", 2.8),
    "stopstreaming": ("shock", 2.8),
    "haram": ("shock", 2.7),
    "осуждаю": ("shock", 2.8),
    "осуждаем": ("shock", 2.8),
    # 🔥 Хайп / Клатч
    "ww": ("hype", 2.7),
    "www": ("hype", 2.9),
    "wwww": ("hype", 3.0),
    "ez": ("hype", 2.5),
    "clap": ("hype", 2.4),
    "gigachad": ("hype", 2.7),
    "based": ("hype", 2.6),
    "chadge": ("hype", 2.6),
    "banger": ("hype", 2.8),
    "bussers": ("hype", 2.7),
    "pagman": ("hype", 2.9),
    "pagchomp": ("hype", 2.9),
    "pogu": ("hype", 2.8),
    "pog": ("hype", 2.7),
    "pogchamp": ("hype", 2.8),
    "hypers": ("hype", 2.6),
    "letsgo": ("hype", 2.8),
    "goat": ("hype", 2.7),
    "cinema": ("hype", 2.8),
    "holy": ("hype", 2.7),
    "wicked": ("hype", 2.7),
    "minionsturdy": ("hype", 2.6),
    "goo": ("hype", 2.6),
    # 😬 Фейл / Кринж
    "ll": ("cringe", 2.6),
    "lll": ("cringe", 2.8),
    "pepepains": ("cringe", 2.8),
    "hahaa": ("cringe", 2.7),
    "diesofcringe": ("cringe", 2.9),
    "cringe": ("cringe", 2.8),
    "facepalm": ("cringe", 2.7),
    "failfish": ("cringe", 2.7),
    "notlikethis": ("cringe", 2.7),
    "xylipizdish": ("cringe", 2.8),
    "veryfu": ("cringe", 2.7),
    "catfu": ("cringe", 2.7),
    "чушпан": ("cringe", 2.7),
}

# Фоновые/музыкальные/приветственные эмоуты, которые НЕ должны вызывать ложную нарезку клипов
AMBIENT_IDLE_SUBSTRINGS = (
    "hello", "peeposithey", "peepoleave", "sleep", "waiting", "chatting",
    "prayge", "blessrng", "okayge", "okayeg", "catjam", "pepejam", "peped",
    "neondance", "abdulpls", "catkiss", "lovesvin", "beertime", "tasty",
    "coffee", "teatime", "afk", "brb", "goodnight", "goodmorning",
)

EMOTE_STEM_RULES: list[tuple[tuple[str, ...], str, float]] = [
    (
        ("laugh", "kek", "lul", "rofl", "xdd", "giggle", "hehe", "icant", "porval", "clown", "troll", "смех", "ору", "ржу", "рофл"),
        "laugh",
        2.7,
    ),
    (
        ("aintnoway", "caught", "huh", "monka", "shock", "alert", "wtf", "booba", "flushed", "crazy", "despair", "panic", "scream", "scared", "осужд", "дурк", "шок", "аху"),
        "shock",
        2.8,
    ),
    (
        ("pog", "pag", "hype", "chad", "based", "banger", "wicked", "clutch", "goat", "holy", "sturdy", "cinema", "имба", "сюда", "база"),
        "hype",
        2.7,
    ),
    (
        ("pains", "cringe", "fail", "pizd", "чушпан", "кринж", "тильт", "позор"),
        "cringe",
        2.7,
    ),
]


def classify_emote_name(emote_name: str) -> tuple[str, float, bool]:
    """
    Классифицирует имя эмоута 7TV/BTTV:
    возвращает (category, score, is_direct_reaction).
    - Для известных эмоциональных эмоутов ('CAUGHT', 'xdd', 'AINTNOWAY', 'RoflanEbalo' и т.д.):
      is_direct_reaction=True, score=2.4..3.1.
    - Для фоновых/танцевальных эмоутов ('catJAM', 'Chatting', 'Waiting'):
      category='ambient', score=0.6, is_direct_reaction=False.
    - Для уникальных локальных эмоутов канала ('Kurica', 'WideSvin' и т.д.):
      category='channel_emote', score=1.75, is_direct_reaction=False (но если >=4 зрителей спамят
      этот эмоут одновременно в одном 10с окне, детектор засчитывает волну эмоутов как реакцию!).
    """
    low = (emote_name or "").strip().lower()
    if not low:
        return "normal", 1.0, False

    if low in GLOBAL_EMOTE_CATEGORIES:
        cat, sc = GLOBAL_EMOTE_CATEGORIES[low]
        return cat, sc, True

    if any(sub in low for sub in AMBIENT_IDLE_SUBSTRINGS):
        return "ambient", 0.65, False

    for stems, cat, sc in EMOTE_STEM_RULES:
        if any(st in low for st in stems):
            return cat, sc, True

    return "channel_emote", 1.75, False


class ChannelEmoteManager:
    """
    Загружает и кэширует эмоуты 7TV и BetterTTV (BTTV) для каждого Twitch-канала.
    Кэш хранится в buffer/_emotes_cache.json, чтобы старт программы занимал 0.01 сек.
    """

    CACHE_TTL_SEC = 24 * 3600

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self.cache_path = self.base_dir / "buffer" / "_emotes_cache.json"
        self._lock = threading.Lock()
        self._data: dict = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except Exception:
                pass
        return {"channels": {}}

    def _save_cache(self) -> None:
        with self._lock:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def get_channel_emotes(self, channel: str) -> list[str]:
        ch = (channel or "").strip().lower()
        rec = (self._data.get("channels") or {}).get(ch)
        if isinstance(rec, dict) and isinstance(rec.get("emotes"), list):
            return [str(x) for x in rec["emotes"] if x]
        return []

    def get_channel_emote_stats(self, channel: str) -> tuple[int, int, int]:
        """Возвращает (всего_эмоутов, 7tv_count, bttv_count)."""
        ch = (channel or "").strip().lower()
        rec = (self._data.get("channels") or {}).get(ch)
        if isinstance(rec, dict):
            all_cnt = len(rec.get("emotes") or [])
            s7_cnt = int(rec.get("7tv_count") or 0)
            bt_cnt = int(rec.get("bttv_count") or 0)
            return all_cnt, s7_cnt, bt_cnt
        return 0, 0, 0

    def build_channel_emote_lookup(self, channel: str) -> dict[str, tuple[str, str, float, bool]]:
        """
        Создает быстрый словарь для детектора чата:
        token_lower -> (original_name, category, score, is_direct_reaction),
        объединяя глобальные топ-эмоуты 7TV/BTTV и индивидуальные эмоуты канала.
        """
        lookup: dict[str, tuple[str, str, float, bool]] = {}
        for em_low, (cat, sc) in GLOBAL_EMOTE_CATEGORIES.items():
            lookup[em_low] = (em_low, cat, sc, True)

        for em_name in self.get_channel_emotes(channel):
            low = em_name.strip().lower()
            if not low or len(low) < 2:
                continue
            if low in lookup:
                orig_cat, orig_sc, orig_react = lookup[low][1], lookup[low][2], lookup[low][3]
                lookup[low] = (em_name, orig_cat, orig_sc, orig_react)
            else:
                cat, sc, is_react = classify_emote_name(em_name)
                lookup[low] = (em_name, cat, sc, is_react)
        return lookup

    @staticmethod
    def _fetch_twitch_user_id(login: str) -> str:
        q = "query($login: String!) { user(login: $login) { id } }"
        try:
            r = requests.post(
                TWITCH_GQL_URL,
                headers={"Client-Id": TWITCH_PUBLIC_CLIENT_ID},
                json={"query": q, "variables": {"login": login}},
                timeout=8,
            )
            return str((((r.json().get("data") or {}).get("user") or {}).get("id")) or "")
        except Exception:
            return ""

    def fetch_and_cache_channel(self, channel: str, force: bool = False) -> tuple[int, int, int]:
        """Скачивает эмоуты 7TV + BTTV для одного канала и сохраняет в кэш."""
        ch = re.sub(r"[^a-zA-Z0-9_]", "", (channel or "").strip().lower())
        if not ch:
            return 0, 0, 0

        existing = (self._data.get("channels") or {}).get(ch)
        now = time.time()
        if (
            not force
            and isinstance(existing, dict)
            and existing.get("emotes")
            and (now - float(existing.get("updated_ts") or 0.0)) < self.CACHE_TTL_SEC
        ):
            return self.get_channel_emote_stats(ch)

        uid = str((existing or {}).get("twitch_id") or "") or self._fetch_twitch_user_id(ch)
        if not uid:
            return self.get_channel_emote_stats(ch)

        s7_names: list[str] = []
        bttv_names: list[str] = []

        # 1. Загружаем 7TV эмоуты канала
        for url in (
            f"https://api.7tv.app/v3/users/twitch/{uid}",
            f"https://7tv.io/v3/users/twitch/{uid}",
        ):
            try:
                r7 = requests.get(url, timeout=12)
                if r7.status_code == 200:
                    em_list = ((r7.json().get("emote_set") or {}).get("emotes")) or []
                    s7_names = [str(e.get("name") or "").strip() for e in em_list if e.get("name")]
                    break
                if r7.status_code == 404:
                    break
            except Exception:
                continue

        # 2. Загружаем BetterTTV (BTTV) эмоуты канала
        try:
            rb = requests.get(f"https://api.betterttv.net/3/cached/users/twitch/{uid}", timeout=8)
            if rb.status_code == 200:
                jd = rb.json()
                raw_b = (jd.get("channelEmotes") or []) + (jd.get("sharedEmotes") or [])
                bttv_names = [str(e.get("code") or "").strip() for e in raw_b if e.get("code")]
        except Exception:
            pass

        seen: set[str] = set()
        combined: list[str] = []
        for name in s7_names + bttv_names:
            if not name:
                continue
            low = name.lower()
            if low not in seen:
                seen.add(low)
                combined.append(name)

        # Если сеть временно недоступна, сохраняем старый кэш
        if not combined and isinstance(existing, dict) and existing.get("emotes"):
            return self.get_channel_emote_stats(ch)

        with self._lock:
            self._data.setdefault("channels", {})[ch] = {
                "twitch_id": uid,
                "updated_ts": now,
                "7tv_count": len(s7_names),
                "bttv_count": len(bttv_names),
                "emotes": combined,
            }
        self._save_cache()
        return len(combined), len(s7_names), len(bttv_names)

    def sync_channels(
        self,
        channels: list[str],
        force: bool = False,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> dict[str, tuple[int, int, int]]:
        """Параллельно обновляет эмоуты 7TV + BTTV для списка каналов."""
        clean_chs = [c.strip().lower() for c in channels if c and c.strip()]
        results: dict[str, tuple[int, int, int]] = {}
        if not clean_chs:
            return results

        with ThreadPoolExecutor(max_workers=min(5, len(clean_chs))) as pool:
            futs = {pool.submit(self.fetch_and_cache_channel, ch, force): ch for ch in clean_chs}
            for fut, ch in futs.items():
                try:
                    results[ch] = fut.result()
                except Exception:
                    results[ch] = self.get_channel_emote_stats(ch)

        if log_cb:
            total_em = sum(v[0] for v in results.values())
            active_chs = sum(1 for v in results.values() if v[0] > 0)
            if total_em > 0:
                log_cb(
                    f"🎭 Подгружены эмоуты 7TV/BTTV: {total_em} шт. для {active_chs}/{len(clean_chs)} каналов!"
                )
        return results
