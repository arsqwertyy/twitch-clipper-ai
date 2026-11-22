import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

STREAMER_DISPLAY_NAMES: dict[str, tuple[str, list[str]]] = {
    "frametamer666": ("Фреймтамер", ["#frametamer", "#фреймтамер", "#frametamer666"]),
    "stintik": ("Стинт", ["#stintik", "#стинт", "#стинтик"]),
    "recrent": ("Рекрент", ["#recrent", "#рекрент"]),
    "sasavot": ("Сасавот", ["#sasavot", "#сасавот"]),
    "buster": ("Бустер", ["#buster", "#бустер"]),
    "evelone2004": ("Эвелон", ["#evelone", "#эвелон", "#evelone192"]),
    "gensyxa": ("Генсуха", ["#gensyxa", "#генсуха"]),
    "bratishkinoff": ("Братишкин", ["#bratishkinoff", "#братишкин"]),
    "ravshann": ("Равшан", ["#ravshann", "#равшан"]),
    "t2x2": ("T2x2", ["#t2x2", "#т2х2"]),
    "enzzai": ("Enzzai", ["#enzzai", "#энзай"]),
    "whylollycry": ("Лолли", ["#whylollycry", "#лолли"]),
    "korya_mc": ("Коря", ["#korya_mc", "#коря", "#korya"]),
    "zubarefff": ("Зубарев", ["#zubarefff", "#зубарев"]),
    "shadowkekw": ("Шадоукек", ["#shadowkekw", "#шадоукек"]),
    "mokrivskyi": ("Мокривский", ["#mokrivskyi", "#мокривский"]),
    "deepins02": ("Дипинс", ["#deepins", "#дипинс"]),
    "mazellovvv": ("Мазеллов", ["#mazellovvv", "#мазеллов"]),
    "jesusavgn": ("Хесус", ["#jesusavgn", "#хесус"]),
}

TAG_TO_HOOK: dict[str, tuple[str, str]] = {
    "смех": ("РАЗРЫВНОЙ МОМЕНТ", "😂"),
    "шок": ("ЧТО ПРОИЗОШЛО?!", "😱"),
    "разрыв": ("ПОЛНЫЙ РАЗРЫВ", "😱"),
    "хайп": ("ЛЕГЕНДАРНЫЙ МОМЕНТ", "🔥"),
    "клатч": ("БЕЗУМНЫЙ КЛАТЧ", "🔥"),
    "кринж": ("ЭТО ЧТО БЫЛО", "😬"),
    "фейл": ("ЭПИЧНЫЙ ФЕЙЛ", "😬"),
    "клип": ("ЧАТ ВЗОРВАЛСЯ", "✂️"),
}

VALID_VISIBILITIES = ("public", "unlisted", "private")

DEFAULT_YOUTUBE_CFG = {
    "enabled": True,
    "auto_upload": True,
    "visibility": "public",
    "category_id": "22",
    "min_viral_score": 0,
    "min_interval_minutes": 60,
    "max_per_day": 24,
    "client_secrets_path": "client_secrets.json",
    "token_path": "youtube_token.json",
    "title_hashtags": "#dinodrop #динодроп",
    "auto_title": True,
    "auto_comment": True,
    "extra_hashtags": "",
    "delete_after_upload": True,
}

SETUP_HELP_TEXT = (
    "Как бесплатно получить client_secrets.json (займёт 2–3 минуты, карта не нужна):\n\n"
    "1. Откройте https://console.cloud.google.com/ и войдите в свой Google-аккаунт.\n"
    "2. Сверху нажмите «Select a project» -> «New Project» (например, TwitchShorts) -> Create.\n"
    "3. В меню слева: «APIs & Services» -> «Library» -> найдите «YouTube Data API v3» -> нажмите Enable.\n"
    "4. В меню слева: «Google Auth Platform» (или «OAuth consent screen»):\n"
    "   • Выберите «External» (Внешнее), введите любое название приложения и свой email.\n"
    "   • В разделе «Audience / Test users» нажмите «Add users» и добавьте СВОЙ Gmail (от YouTube-канала)!\n"
    "5. В разделе «Clients» (или «Credentials») -> «Create Client / OAuth client ID»:\n"
    "   • Application type: «Desktop app» (Приложение для ПК) -> Create.\n"
    "   • Нажмите «Download JSON» и выберите скачанный файл здесь кнопкой «📂 Выбрать client_secrets.json».\n"
    "6. Нажмите «🔑 Авторизовать YouTube (OAuth)» — откроется ваш обычный Chrome для подтверждения входа.\n\n"
    "Если Google предупредит «App hasn't been verified» — нажмите «Advanced / Дополнительные настройки» -> "
    "«Go to ... (unsafe)» (это ваше собственное личное приложение)."
)


@dataclass
class UploadResult:
    ok: bool
    video_url: str = ""
    video_id: str = ""
    title: str = ""
    actual_privacy: str = ""
    first_comment: str = ""
    warning: str = ""
    error: str = ""
    needs_login: bool = False
    skipped_reason: str = ""
    deleted_local: bool = False


_CHAT_BAIT_RE = re.compile(
    r"(?:"
    r"\b(?:напишите|пишите|ставьте|плюс(?:ик|ы)?|минус(?:ы)?|голосу(?:ем|йте)|спамьте)\s+(?:в\s+чат\w*|плюс)\b|"
    r"\b(?:плюс(?:ик|ы)?|минус(?:ы)?|единичк\w*|двоечк\w*|цифр\w+\s+\d)\s+в\s+чат\w*\b|"
    r"\bчат\s*,?\s*(?:напишите|пишите|ставьте|плюс(?:ик|ы)?|голосуем)\b|"
    r"\bкто\s+(?:согласен|за\s+это|тоже\s+так)\s*,?\s*(?:напишите|пишите|ставьте|плюс)\b|"
    r"\b(?:участв\w+\s+в\s+розыгрыш\w*|условия\s+розыгрыш\w*|итоги\s+розыгрыш\w*)\b|"
    r"\bспасибо\s+(?:большое\s+|огромное\s+)?за\s+(?:сабк\w*|подписк\w*|прайм|подарочн\w+|рейд)\b"
    r")",
    re.IGNORECASE,
)


def evaluate_transcript_hype(transcript_text: str, words_cnt: int = 0) -> tuple[float, str, bool]:
    """
    Оценивает качество и интересность речи стримера (по тексту Faster-Whisper) для выбора «👑 Короля часа»:
    - Возвращает (бонус_или_штраф_баллов, краткое_пояснение, это_байт_на_чат).
    - Штрафует (-45) переклички чата («напишите плюс в чат», «голосуем в чате», «спасибо за сабку»)
      и пустые по речи клипы (<6 слов -> -22).
    - Поощряет (+10..+26) насыщенную живую речь, яркую цитату, эмоции и сюжетные триггеры.
    """
    txt = re.sub(r"\s+", " ", (transcript_text or "").strip())
    tokens = [re.sub(r"[^\wа-яА-ЯёЁ]+", "", w.lower()) for w in txt.split()]
    tokens = [w for w in tokens if w]
    eff_words = max(int(words_cnt or 0), len(tokens))

    if _CHAT_BAIT_RE.search(txt):
        return -45.0, "⚠️ байт на чат / голосование (-45)", True

    meaningful = [w for w in tokens if len(w) >= 3 and w not in _BORING_WORDS and not _LAUGH_ONLY_RE.match(w)]

    if eff_words < 5 or len(meaningful) < 3:
        return -22.0, "⚠️ почти нет речи (-22)", False
    if eff_words < 10 or len(meaningful) < 6:
        return -8.0, "мало речи (-8)", False

    bonus = 0.0
    notes: list[str] = []

    if 16 <= eff_words <= 95 and len(meaningful) >= 10:
        bonus += 10.0
        notes.append("живая речь")
    elif eff_words >= 10:
        bonus += 5.0

    quote = _extract_catchy_quote(txt, max_len=46)
    if quote:
        bonus += 7.0
        notes.append("яркая цитата")

    low = txt.lower()
    story_hits = sum(1 for r in _INTERESTING_TITLE_ROOTS if r in low)
    if story_hits >= 2:
        bonus += 6.0
        notes.append("сюжет")
    elif story_hits == 1:
        bonus += 3.5
        notes.append("сюжет")

    if "!" in txt or "?" in txt:
        bonus += 3.0
        notes.append("эмоции")

    bonus = round(min(26.0, bonus), 1)
    summary = ", ".join(notes) if notes else "речь в норме"
    sign = "+" if bonus >= 0 else ""
    return bonus, f"{summary} ({sign}{bonus:.0f})", False


def compute_clip_hype_rank(
    peak_score: float,
    z_score: float,
    viral_score: float,
    transcript_text: str = "",
    words_cnt: int = 0,
) -> float:
    """
    Вычисляет единый рейтинг хайпа клипа для часового отбора («👑 Король часа»):
    сочетает силу взрыва чата (viral_score + z_score + log1p(peak_score))
    и ИИ-оценку смысла речи стримера по расшифровке Faster-Whisper.
    """
    chat_rank = (
        max(0.0, float(viral_score)) * 1.0
        + max(0.0, float(z_score)) * 8.0
        + math.log1p(max(0.0, float(peak_score))) * 12.0
    )
    speech_delta, _reason, _is_bait = evaluate_transcript_hype(transcript_text, words_cnt=words_cnt)
    return round(max(1.0, chat_rank + speech_delta), 1)


def parse_channel_from_filename(filename: str) -> str:
    """Извлекает логин стримера из имени файла клипа (с_баннером_frametamer666_2026...mp4)."""
    stem = Path(filename).stem
    for prefix in ("с_баннером_", "без_баннера_", "клип_", "test_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix) :]
    if stem.lower().startswith("test_"):
        stem = stem[5:]
    m = re.match(r"^([a-zA-Z0-9_]+?)_\d{8}_\d{6}", stem)
    if m:
        return m.group(1).lower()
    parts = stem.split("_")
    return parts[0].lower() if parts else "twitch"


_TITLE_SWEAR_RE = re.compile(
    r"\b(?:бля\w*|пизд\w*|еба\w*|ёба\w*|ебл\w*|ёбл\w*|хуй\w*|хуя\w*|хуе\w*|хуё\w*|наху\w*|"
    r"сук\w*|сучк\w*|ахуе\w*|охуе\w*|ниху\w*|долбо\w*|уеб\w*|уёб\w*|пидор\w*|пидр\w*|"
    r"залуп\w*|шлюх\w*|гандон\w*|мудак\w*|ебуч\w*|заеб\w*|заёб\w*|выёб\w*|выеб\w*)\b",
    re.IGNORECASE,
)

_LEADING_FILLERS = {
    "ну", "короче", "типа", "вот", "блин", "да", "а", "и", "но", "эй",
    "алло", "ага", "угу", "слушай", "слушайте", "погоди", "ладно", "так",
}

_BORING_WORDS = {
    "да", "нет", "ну", "вот", "типа", "короче", "блин", "ладно", "ага", "угу",
    "ой", "ай", "эй", "ээ", "мм", "че", "чё", "а", "и", "но", "в", "на", "с",
    "это", "тут", "там", "просто", "вообще", "реально", "короч", "привет", "пока",
}

_LAUGH_ONLY_RE = re.compile(r"^(?:[ахпхвкх]{3,}|ха+|аха+|пха+|лол|кек|ору+)[!?.…]*$", re.IGNORECASE)

_INTERESTING_TITLE_ROOTS = (
    "сломал", "выпал", "убил", "упал", "попал", "выиграл", "проиграл", "слил",
    "задонат", "донат", "испугал", "скример", "шок", "жесть", "легенд", "имба",
    "капец", "треш", "дичь", "не верю", "серьезн", "серьёзн", "видели", "посмотр",
    "как так", "что это", "откуда", "зачем", "почему", "разнес", "разнёс", "снёс",
    "снес", "забанил", "кикнул", "взорвал", "машин", "полиц", "клатч", "хэдшот",
    "миллион", "тысяч", "девушк", "мама", "папа", "друг", "тайна", "секрет",
)


def _clean_phrase_for_title(raw: str) -> tuple[str, bool]:
    """
    Очищает фразу для заголовка YouTube: убирает нецензурные слова-паразиты и лишнюю пунктуацию,
    возвращает (очищенная_фраза, был_ли_мат).
    """
    had_swear = bool(_TITLE_SWEAR_RE.search(raw))
    cleaned = _TITLE_SWEAR_RE.sub(" ", raw) if had_swear else raw
    cleaned = re.sub(r"\s+([,!?.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:—\-])\s*([,;:—\-])+", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:—-\"'«»()")

    # Снимаем вялые вводные слова в самом начале («Ну, короче, ...»)
    tokens = cleaned.split()
    while len(tokens) > 2:
        first_norm = re.sub(r"[^\wа-яА-ЯёЁ]+", "", tokens[0].lower())
        if first_norm in _LEADING_FILLERS:
            tokens.pop(0)
        else:
            break
    cleaned = " ".join(tokens).strip(" ,;:—-\"'«»()")
    return cleaned, had_swear


def _trim_on_word_boundary(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    words = text.split()
    out: list[str] = []
    for w in words:
        cand = (" ".join(out + [w])).strip()
        if len(cand) > max_len - 1:
            break
        out.append(w)
    if not out:
        return text[: max(1, max_len - 1)].rstrip(" .,!?:;-") + "…"
    res = " ".join(out).rstrip(" ,;:—-")
    if not res.endswith(("!", "?", ".", "…")):
        res += "…"
    return res


def _extract_catchy_quote(transcript_text: str, max_len: int = 46) -> str:
    """
    Выбирает из расшифровки Whisper самую живую, понятную и подходящую под видео фразу (2–8 слов)
    без мата (чтобы YouTube не резал охваты и монетизацию).
    """
    txt = re.sub(r"\s+", " ", (transcript_text or "").strip())
    if len(txt) < 8:
        return ""

    # Разбиваем на предложения; если Whisper дал сплошной поток без точек — режем также по запятым/фрагментам
    raw_chunks = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", txt) if s.strip()]
    candidates: list[str] = list(raw_chunks)
    if len(raw_chunks) <= 2 and len(txt) > max_len:
        for ch in raw_chunks:
            sub_parts = [p.strip() for p in re.split(r",\s+", ch) if len(p.strip()) >= 10]
            if len(sub_parts) >= 2:
                candidates.extend(sub_parts)

    best = ""
    best_sc = -1e9

    for raw_s in candidates:
        s_clean, had_swear = _clean_phrase_for_title(raw_s)
        if not s_clean or len(s_clean) < 10:
            continue

        words = s_clean.split()
        if len(words) < 2:
            continue

        # Отбрасываем фразы, состоящие только из смеха или пустых междометий («Да да, ну вот»)
        meaningful = [
            re.sub(r"[^\wа-яА-ЯёЁ]+", "", w.lower())
            for w in words
        ]
        meaningful = [w for w in meaningful if len(w) >= 3 and w not in _BORING_WORDS and not _LAUGH_ONLY_RE.match(w)]
        if len(meaningful) < 2:
            continue

        if len(s_clean) > max_len:
            if len(words) > 10:
                continue
            s_final = _trim_on_word_boundary(s_clean, max_len)
        else:
            s_final = s_clean

        if len(s_final) < 10 or len(s_final) > max_len:
            continue

        low = s_final.lower()
        sc = 2.0
        # Сильный приоритет естественно чистым фразам без мата
        sc += 3.0 if not had_swear else -1.2
        if "!" in s_final:
            sc += 2.2
        if "?" in s_final:
            sc += 2.0
        if 16 <= len(s_final) <= min(42, max_len):
            sc += 1.4
        if len(meaningful) >= 3:
            sc += 1.0
        if any(r in low for r in _INTERESTING_TITLE_ROOTS):
            sc += 1.8

        if sc > best_sc:
            best_sc = sc
            best = s_final

    if best:
        # Убираем одиночную точку в конце заголовка (в Shorts смотрится живее без точки, но сохраняем ! и ?)
        if best.endswith(".") and not best.endswith("..."):
            best = best[:-1].rstrip()
        return best[0].upper() + best[1:]
    return ""


def _detect_topic_fallback(disp_name: str, moment_tag: str, transcript_text: str) -> tuple[str, str]:
    """Подбирает разнообразный тематический заголовок по смыслу речи и типу реакции, если нет готовой цитаты."""
    low = (transcript_text or "").lower()
    tag_low = (moment_tag or "").lower()

    topic_rules: list[tuple[tuple[str, ...], list[str], str]] = [
        (
            ("донат", "рублей", "тысяч", "миллион", "косар", "сабк", "подписк"),
            ["в шоке от такого доната", "не поверил донату на стриме", "офигел от дикого доната"],
            "😱",
        ),
        (
            ("скример", "страшн", "испугал", "боюсь", "призрак", "монстр", "хоррор"),
            ["жёстко испугался на стриме", "поймал скример в прямом эфире", "чуть не упал со стула"],
            "😱",
        ),
        (
            ("тачк", "машин", "дрифт", "полиц", "мент", "разбил", "влетел", "авари", "гта", "gta"),
            ["устроил полный хаос на дороге", "выдал безумный момент на скорости", "врезался так, что все в шоке"],
            "🔥",
        ),
        (
            ("убил", "хэдшот", "клатч", "катк", "снайп", "гранат", "выстрел", "урон", "килл", "попал", "раунд"),
            ["выдал нереальный момент в катке", "развалил в самый нужный момент", "затащил безумный клатч"],
            "🔥",
        ),
        (
            ("слил", "упал", "сдох", "сломал", "проиграл", "вылетел", "забанил", "кринж", "позор"),
            ["словил легендарный фейл", "не ожидал такого провала", "словил жесткий фейл на стриме"],
            "😬",
        ),
    ]

    variants: Optional[list[str]] = None
    emoji = "🔥"

    for roots, phrases, em in topic_rules:
        if any(r in low for r in roots):
            variants, emoji = phrases, em
            break

    if variants is None:
        if "смех" in tag_low or any(r in low for r in ("ахах", "хаха", "угар", "смешн", "ржу")):
            variants = [
                "порвался со смеху на стриме",
                "не выдержал и взорвался смехом",
                "до слёз рассмешил весь чат",
            ]
            emoji = "😂"
        elif "шок" in tag_low or "разрыв" in tag_low:
            variants = [
                "в полном шоке от происходящего",
                "не поверил своим глазам на стриме",
                "увидел такое, что дар речи пропал",
            ]
            emoji = "😱"
        elif "кринж" in tag_low or "фейл" in tag_low:
            variants = [
                "словил лютый фейл на стриме",
                "попал в максимально неловкий момент",
                "просто в шоке с этого кринжа",
            ]
            emoji = "😬"
        elif "клип" in tag_low:
            variants = [
                "взорвал весь чат этим моментом",
                "выдал момент, который точно в клипы",
                "подарил лучший момент стрима",
            ]
            emoji = "✂️"
        else:
            variants = [
                "выдал легендарный момент со стрима",
                "устроил момент, который нельзя пропустить",
                "выдал поворот, которого никто не ждал",
            ]
            emoji = "🔥"

    seed = abs(hash((disp_name, low[:80], moment_tag, int(time.time() // 30))))
    chosen = variants[seed % len(variants)]
    return f"{disp_name} {chosen}", emoji


def build_shorts_metadata(
    channel: str,
    moment_tag: str = "",
    transcript_text: str = "",
    extra_hashtags: str = "",
    title_hashtags: str = "#dinodrop #динодроп",
    auto_title: bool = True,
) -> tuple[str, str]:
    """
    Генерирует подходящий под сюжет видео заголовок (<=100 символов) с ОБЯЗАТЕЛЬНЫМИ хэштегами
    (по умолчанию '#dinodrop #динодроп') и чистое описание для YouTube Shorts.
    """
    req_tags = re.sub(r"\s+", " ", (title_hashtags or "#dinodrop #динодроп").strip())
    desc_text = (extra_hashtags or "").strip()

    if not auto_title and req_tags:
        return req_tags[:100], desc_text

    ch_low = (channel or "twitch").strip().lower()
    if ch_low in STREAMER_DISPLAY_NAMES:
        disp_name, ch_tags = STREAMER_DISPLAY_NAMES[ch_low]
    else:
        disp_name = ch_low.capitalize()
        ch_tags = [f"#{ch_low}"] if re.fullmatch(r"[a-zA-Z0-9_]+", ch_low) else []

    emoji = "🔥"
    tag_low = (moment_tag or "").lower()
    for key, (_lbl, em) in TAG_TO_HOOK.items():
        if key in tag_low:
            emoji = em
            break

    suffix = req_tags if req_tags else "#dinodrop #динодроп"
    # Оставляем точный запас под обязательные хэштеги, чтобы они НИКОГДА не обрезались лимитом 100 символов!
    max_prefix = max(14, 99 - len(suffix) - 1)

    quote_budget = max(14, max_prefix - len(disp_name) - 4)
    quote = _extract_catchy_quote(transcript_text, max_len=quote_budget)
    if quote:
        prefix = f"{disp_name}: {quote} {emoji}"
    else:
        fallback_line, fb_emoji = _detect_topic_fallback(disp_name, moment_tag, transcript_text)
        prefix = f"{fallback_line} {fb_emoji}"

    if len(prefix) > max_prefix:
        prefix = _trim_on_word_boundary(prefix, max_prefix)

    title = f"{prefix} {suffix}".strip()
    if len(title) > 100:
        avail = max(1, 100 - len(suffix) - 1)
        title = f"{_trim_on_word_boundary(prefix, avail)} {suffix}".strip()[:100]

    if desc_text:
        description = desc_text
    else:
        raw_tags = re.findall(r"#[\wа-яА-ЯёЁ]+", suffix) + list(ch_tags) + ["#shorts", "#twitch", "#твич"]
        seen_t: set[str] = set()
        uniq_tags: list[str] = []
        for t in raw_tags:
            tl = t.lower()
            if tl not in seen_t:
                seen_t.add(tl)
                uniq_tags.append(t)
        description = " ".join(uniq_tags[:15])

    return title, description


def build_shorts_first_comment(
    channel: str,
    moment_tag: str = "",
    transcript_text: str = "",
) -> str:
    """
    Генерирует живой первый комментарий от автора канала под сюжет ролика для буста комментариев в YouTube Shorts.
    """
    ch_low = (channel or "twitch").strip().lower()
    disp_name = STREAMER_DISPLAY_NAMES[ch_low][0] if ch_low in STREAMER_DISPLAY_NAMES else ch_low.capitalize()
    quote = _extract_catchy_quote(transcript_text, max_len=52)
    low = (transcript_text or "").lower()
    tag_low = (moment_tag or "").lower()

    if quote:
        if any(r in low for r in ("убил", "клатч", "хэдшот", "катк", "снайп", "раунд", "попал")):
            return f"«{quote}» 🔥 На сколько из 10 оцените этот момент со стрима ({disp_name})? Пишите в комменты 👇"
        if "смех" in tag_low or any(r in low for r in ("ахах", "хаха", "ржу", "угар")):
            return f"«{quote}» 😂 Кто тоже орнул на этом моменте ({disp_name})? Пишите в комменты 👇"
        if "шок" in tag_low or "разрыв" in tag_low or any(r in low for r in ("скример", "донат", "шок", "испугал")):
            return f"«{quote}» 😱 Как вам такая реакция на стриме ({disp_name})? Что думаете? 👇"
        if "фейл" in tag_low or "кринж" in tag_low or any(r in low for r in ("слил", "упал", "сломал", "проиграл")):
            return f"«{quote}» 😬 Это главный фейл эфира или бывало жёстче? Пишите в комменты 👇"
        return f"«{quote}» 🔥 Как вам этот момент со стрима ({disp_name})? Чьи нарезки делать чаще — пишите 👇"

    templates = [
        f"Как вам этот момент со стрима ({disp_name})? 🔥 Оцените от 1 до 10 в комментах 👇",
        f"{disp_name} снова выдал эмоции на стриме 😂 Как вам момент? Пишите своё мнение 👇",
        f"Кого из стримеров смотрите чаще всего? Пишите в комменты, чьи моменты нарезать ещё! 👇🔥",
    ]
    seed = abs(hash((disp_name, low[:60], moment_tag, int(time.time() // 30))))
    return templates[seed % len(templates)]


def _extract_api_tags(title: str, description: str, channel: str) -> list[str]:
    """Извлекает теги без '#' для поля snippet.tags YouTube Data API v3."""
    found = re.findall(r"#([\wа-яА-ЯёЁ]+)", f"{title} {description}")
    if "shorts" not in [t.lower() for t in found]:
        found.append("shorts")
    ch_low = (channel or "").strip().lower()
    if ch_low and ch_low != "twitch" and ch_low not in [t.lower() for t in found]:
        found.append(ch_low)
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
    return out[:20]


class YouTubeShortsUploader:
    """
    Официальный загрузчик вертикальных клипов в YouTube Shorts через YouTube Data API v3 (OAuth 2.0).
    - Авторизация проходит в обычном системном браузере пользователя (без ошибок безопасности Google).
    - После первого входа токен сохраняется в youtube_token.json и обновляется автоматически.
    - Загрузка идёт тихо в фоне с поддержкой докачки (resumable upload), учётом дневного лимита и паузы.
    """

    def __init__(self, base_dir: Path, yt_cfg: Optional[dict] = None):
        self.base_dir = Path(base_dir).resolve()
        self.cfg: dict = dict(DEFAULT_YOUTUBE_CFG)
        if yt_cfg:
            self.cfg.update(yt_cfg)
        self.history_path = self.base_dir / "youtube_uploads.json"
        self._auth_meta_path = self.base_dir / ".yt_api_channel.json"
        self._clip_meta_path = self.base_dir / "buffer" / "_clip_meta.json"
        self._api_lock = threading.Lock()
        self._history_lock = threading.Lock()

    def save_clip_meta(self, filename: str, channel: str, moment_tag: str, transcript_text: str) -> None:
        """Сохраняет краткие метаданные клипа (эмоция + текст Whisper), чтобы даже ручная заливка знала сюжет видео."""
        with self._history_lock:
            data: dict = {}
            if self._clip_meta_path.exists():
                try:
                    loaded = json.loads(self._clip_meta_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        data = loaded
                except Exception:
                    data = {}
            data[Path(filename).name] = {
                "channel": channel,
                "moment_tag": moment_tag,
                "transcript": (transcript_text or "")[:1200],
                "ts": time.time(),
            }
            if len(data) > 200:
                # Оставляем последние 150 записей
                items = sorted(data.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0.0)), reverse=True)[:150]
                data = dict(items)
            try:
                self._clip_meta_path.parent.mkdir(parents=True, exist_ok=True)
                self._clip_meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def delete_clip_and_siblings(self, video_path: Path, cleanup_all_other_clips: bool = False) -> int:
        """
        Полностью удаляет с ПК сам видеофайл, превью из buffer/_thumbs/
        и при cleanup_all_other_clips=True — все остальные .mp4 в папке клипов.
        Возвращает количество удалённых .mp4 файлов.
        """
        video_path = Path(video_path).resolve()
        clips_dir = video_path.parent
        stem_name = video_path.name
        for prefix in ("с_баннером_", "без_баннера_", "клип_"):
            if stem_name.startswith(prefix):
                stem_name = stem_name[len(prefix) :]
                break

        targets: set[Path] = {video_path}
        if clips_dir.exists():
            targets.add(clips_dir / f"с_баннером_{stem_name}")
            targets.add(clips_dir / f"без_баннера_{stem_name}")
            targets.add(clips_dir / f"клип_{stem_name}")
            if cleanup_all_other_clips:
                for mp4 in clips_dir.glob("*.mp4"):
                    targets.add(mp4.resolve())

        thumbs_dir = self.base_dir / "buffer" / "_thumbs"
        deleted_cnt = 0
        for t_path in targets:
            for _attempt in range(4):
                try:
                    if t_path.exists():
                        t_path.unlink()
                        deleted_cnt += 1
                    break
                except OSError:
                    time.sleep(0.20)
            if thumbs_dir.exists():
                for th in thumbs_dir.glob(f"{t_path.name}_*.jpg"):
                    try:
                        th.unlink(missing_ok=True)
                    except OSError:
                        pass
        return deleted_cnt

    def get_clip_meta(self, filename: str) -> dict:
        with self._history_lock:
            if not self._clip_meta_path.exists():
                return {}
            try:
                loaded = json.loads(self._clip_meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    rec = loaded.get(Path(filename).name)
                    if isinstance(rec, dict):
                        return rec
            except Exception:
                pass
            return {}

    def update_cfg(self, yt_cfg: Optional[dict]) -> None:
        merged = dict(DEFAULT_YOUTUBE_CFG)
        if yt_cfg:
            merged.update(yt_cfg)
        self.cfg = merged

    @property
    def client_secrets_path(self) -> Path:
        raw = str(self.cfg.get("client_secrets_path") or "client_secrets.json").strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return p

    @property
    def token_path(self) -> Path:
        raw = str(self.cfg.get("token_path") or "youtube_token.json").strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return p

    @staticmethod
    def is_api_available() -> bool:
        try:
            import googleapiclient.discovery  # noqa: F401
            import google_auth_oauthlib.flow  # noqa: F401
            return True
        except Exception:
            return False

    def has_client_secrets(self) -> bool:
        p = self.client_secrets_path
        return p.exists() and p.is_file()

    def get_auth_info(self) -> dict:
        if not self.token_path.exists():
            return {"logged_in": False}
        if self._auth_meta_path.exists():
            try:
                data = json.loads(self._auth_meta_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("logged_in"):
                    return data
            except Exception:
                pass
        return {"logged_in": True, "channel_name": "", "channel_id": ""}

    def is_logged_in_cached(self) -> bool:
        return self.token_path.exists() and bool(self.get_auth_info().get("logged_in"))

    def _save_auth_info(self, logged_in: bool, channel_id: str = "", channel_name: str = "") -> None:
        payload = {
            "logged_in": bool(logged_in),
            "channel_id": channel_id,
            "channel_name": channel_name,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self._auth_meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # =================== ИСТОРИЯ ЗАГРУЗОК И ЛИМИТЫ ===================

    def _load_history(self) -> dict:
        with self._history_lock:
            if not self.history_path.exists():
                return {"uploads": {}}
            try:
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("uploads"), dict):
                    return data
            except Exception:
                pass
            return {"uploads": {}}

    def _save_history(self, data: dict) -> None:
        with self._history_lock:
            self.history_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_uploaded_record(self, filename: str) -> Optional[dict]:
        hist = self._load_history()
        return hist["uploads"].get(Path(filename).name)

    def _record_upload(self, video_path: Path, video_url: str, title: str, visibility: str, auto: bool) -> None:
        hist = self._load_history()
        now_ts = time.time()
        hist["uploads"][video_path.name] = {
            "filename": video_path.name,
            "url": video_url,
            "title": title,
            "visibility": visibility,
            "auto": bool(auto),
            "timestamp": now_ts,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_history(hist)

    def get_today_upload_stats(self) -> tuple[int, float]:
        """Возвращает (кол-во заливок сегодня, timestamp последней заливки)."""
        hist = self._load_history()
        today = datetime.now().strftime("%Y-%m-%d")
        today_cnt = 0
        last_ts = 0.0
        for rec in hist["uploads"].values():
            if not isinstance(rec, dict):
                continue
            ts = float(rec.get("timestamp") or 0.0)
            if rec.get("date") == today:
                today_cnt += 1
            if ts > last_ts:
                last_ts = ts
        return today_cnt, last_ts

    def is_daily_limit_reached(self) -> tuple[bool, int, int]:
        """Возвращает (лимит_достигнут, залито_сегодня, макс_в_день)."""
        today_cnt, _ = self.get_today_upload_stats()
        max_per_day = max(1, int(self.cfg.get("max_per_day", 12)))
        return (today_cnt >= max_per_day), today_cnt, max_per_day

    def seconds_until_next_auto_upload(self) -> float:
        """Сколько секунд осталось ждать до следующей авто-заливки (0.0 = можно заливать сразу)."""
        min_interval_sec = max(0.0, float(self.cfg.get("min_interval_minutes", 0)) * 60.0)
        if min_interval_sec <= 0.0:
            return 0.0
        _, last_ts = self.get_today_upload_stats()
        if last_ts <= 0.0:
            return 0.0
        return max(0.0, min_interval_sec - (time.time() - last_ts))

    def check_auto_upload_eligibility(
        self,
        video_path: Path,
        viral_score: float,
        ignore_interval: bool = False,
    ) -> tuple[bool, str]:
        """Проверяет, можно ли автоматически залить этот клип (по фильтрам и дневному лимиту)."""
        if not bool(self.cfg.get("enabled", False)) or not bool(self.cfg.get("auto_upload", True)):
            return False, "авто-заливка отключена"
        if not self.is_logged_in_cached():
            return False, "сначала авторизуйте YouTube (OAuth) в Настройках"
        if self.get_uploaded_record(video_path.name) is not None:
            return False, "клип уже был загружен ранее"

        min_viral = float(self.cfg.get("min_viral_score", 0))
        if min_viral > 0 and viral_score < min_viral:
            return False, f"индекс виральности {viral_score:.0f} ниже порога ({min_viral:.0f})"

        today_cnt, last_ts = self.get_today_upload_stats()
        max_per_day = max(1, int(self.cfg.get("max_per_day", 50)))
        if today_cnt >= max_per_day:
            return False, f"достигнут дневной лимит загрузок ({today_cnt}/{max_per_day})"

        if not ignore_interval:
            min_interval_sec = max(0.0, float(self.cfg.get("min_interval_minutes", 0)) * 60.0)
            if min_interval_sec > 0.0 and last_ts > 0 and (time.time() - last_ts) < min_interval_sec:
                rem_min = max(1, int(round((min_interval_sec - (time.time() - last_ts)) / 60.0)))
                return False, f"пауза между публикациями (осталось ~{rem_min} мин)"

        return True, "ok"

    # =================== OAUTH 2.0 АВТОРИЗАЦИЯ И КЛИЕНТ API ===================

    def _load_valid_credentials(self):
        """Загружает и при необходимости тихо обновляет OAuth-токен из youtube_token.json."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self.token_path.exists():
            return None
        try:
            # Не переопределяем scopes при чтении, чтобы токены, выданные до добавления youtube.force-ssl,
            # продолжали безошибочно обновляться и загружать видео
            creds = Credentials.from_authorized_user_file(str(self.token_path))
        except Exception:
            return None

        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json(), encoding="utf-8")
                return creds
            except Exception:
                return None
        return None

    def has_comment_scope(self) -> bool:
        """Проверяет, есть ли в текущем youtube_token.json право на публикацию комментариев (youtube.force-ssl)."""
        if not self.token_path.exists():
            return False
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
            scopes = data.get("scopes") or []
            return any(
                s in scopes
                for s in (
                    "https://www.googleapis.com/auth/youtube.force-ssl",
                    "https://www.googleapis.com/auth/youtube",
                )
            )
        except Exception:
            return False

    def _post_first_comment(
        self,
        youtube,
        video_id: str,
        comment_text: str,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, str]:
        """
        Публикует первый вовлекающий комментарий автора под загруженным YouTube Short.
        Возвращает (опубликованный_текст, предупреждение_если_не_удалось).
        """
        if not video_id or not comment_text.strip():
            return "", ""
        body = {
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": comment_text.strip()[:500],
                    }
                },
            }
        }
        try:
            youtube.commentThreads().insert(part="snippet", body=body).execute()
            if log_cb:
                log_cb(f"💬 Оставлен первый комментарий автора под Short: «{comment_text}»")
            return comment_text, ""
        except Exception as exc:
            err_str = str(exc)
            if "insufficientPermissions" in err_str or "403" in err_str:
                return (
                    "",
                    "Для авто-комментариев под Shorts один раз нажмите «🔑 Авторизовать YouTube (OAuth)» в Настройках "
                    "(добавлено разрешение на публикацию комментариев).",
                )
            return "", ""

    def _build_youtube_service(self, creds):
        from googleapiclient.discovery import build

        return build("youtube", "v3", credentials=creds, cache_discovery=False)

    @staticmethod
    def _fetch_my_channel_info(youtube) -> tuple[str, str]:
        try:
            resp = youtube.channels().list(part="snippet", mine=True).execute()
            items = resp.get("items") or []
            if items:
                ch_id = str(items[0].get("id") or "")
                ch_title = str((items[0].get("snippet") or {}).get("title") or "")
                return ch_id, ch_title
        except Exception:
            pass
        return "", ""

    def authorize_oauth(
        self,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> tuple[bool, str]:
        """
        Запускает официальный OAuth 2.0 вход в обычном браузере пользователя по client_secrets.json.
        Сохраняет обновляемый токен в youtube_token.json.
        """
        if not self.is_api_available():
            return False, "Не установлены библиотеки google-api-python-client / google-auth-oauthlib."
        secrets_p = self.client_secrets_path
        if not secrets_p.exists():
            return (
                False,
                f"Файл ключей не найден: {secrets_p.name}\n\n"
                "Сначала скачайте Desktop OAuth JSON в Google Cloud Console и выберите его кнопкой "
                "«📂 Выбрать client_secrets.json» (подробная инструкция по кнопке «❓ Инструкция»).",
            )

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        from google_auth_oauthlib.flow import InstalledAppFlow

        with self._api_lock:
            log("🌐 Открываю ваш обычный браузер для официального входа Google OAuth 2.0...")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(secrets_p), YOUTUBE_SCOPES)
                creds = flow.run_local_server(
                    port=0,
                    access_type="offline",
                    prompt="select_account consent",
                    open_browser=True,
                    success_message="Авторизация YouTube успешно завершена! Можете закрыть эту вкладку и вернуться в Twitch Clipper.",
                )
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                self.token_path.write_text(creds.to_json(), encoding="utf-8")

                yt = self._build_youtube_service(creds)
                ch_id, ch_name = self._fetch_my_channel_info(yt)
                self._save_auth_info(True, channel_id=ch_id, channel_name=ch_name)
                name_part = f" («{ch_name}»)" if ch_name else ""
                return True, f"YouTube Data API v3 успешно подключён{name_part}!"
            except Exception as exc:
                return False, f"Ошибка OAuth-авторизации: {exc!r}"

    def verify_session(self) -> tuple[bool, str]:
        """Проверяет актуальность OAuth-токена и возвращает название подключённого канала."""
        if not self.is_api_available():
            return False, "Библиотеки Google API не установлены."
        with self._api_lock:
            creds = self._load_valid_credentials()
            if creds is None:
                self._save_auth_info(False)
                return False, "Токен отсутствует или истёк — нажмите «🔑 Авторизовать YouTube (OAuth)»."
            try:
                yt = self._build_youtube_service(creds)
                ch_id, ch_name = self._fetch_my_channel_info(yt)
                self._save_auth_info(True, channel_id=ch_id, channel_name=ch_name)
                if ch_name:
                    return True, f"Подключён канал «{ch_name}» (ID: {ch_id})"
                return True, "OAuth-токен активен и готов к загрузке Shorts."
            except Exception as exc:
                return False, f"Ошибка проверки API: {exc!r}"

    # =================== ЗАГРУЗКА SHORTS ЧЕРЕЗ YOUTUBE DATA API v3 ===================

    def upload_video(
        self,
        video_path: Path,
        title: Optional[str] = None,
        description: Optional[str] = None,
        channel: str = "",
        moment_tag: str = "",
        transcript_text: str = "",
        visibility: Optional[str] = None,
        is_auto: bool = False,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> UploadResult:
        """
        Загружает вертикальное видео (Shorts) на YouTube-канал тихо в фоне через официальный YouTube Data API v3.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            return UploadResult(ok=False, error=f"Файл не найден: {video_path}")
        if not self.is_api_available():
            return UploadResult(ok=False, error="Библиотеки Google YouTube API не установлены.")

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        ch_name = channel or parse_channel_from_filename(video_path.name)
        auto_title_str, auto_desc = build_shorts_metadata(
            channel=ch_name,
            moment_tag=moment_tag,
            transcript_text=transcript_text,
            extra_hashtags=str(self.cfg.get("extra_hashtags", DEFAULT_YOUTUBE_CFG["extra_hashtags"])),
            title_hashtags=str(self.cfg.get("title_hashtags", DEFAULT_YOUTUBE_CFG["title_hashtags"])),
            auto_title=bool(self.cfg.get("auto_title", DEFAULT_YOUTUBE_CFG["auto_title"])),
        )
        final_title = (title or auto_title_str).strip()[:100]
        final_desc = (description if description is not None else auto_desc).strip()[:4900]
        vis_code = str(visibility or self.cfg.get("visibility", "public")).strip().lower()
        if vis_code not in VALID_VISIBILITIES:
            vis_code = "public"
        category_id = str(self.cfg.get("category_id", "22") or "22")
        api_tags = _extract_api_tags(final_title, final_desc, ch_name)

        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        with self._api_lock:
            creds = self._load_valid_credentials()
            if creds is None:
                self._save_auth_info(False)
                return UploadResult(
                    ok=False,
                    needs_login=True,
                    error="Нет активного OAuth-токена — нажмите «🔑 Авторизовать YouTube (OAuth)» в Настройках.",
                )

            try:
                youtube = self._build_youtube_service(creds)
                body = {
                    "snippet": {
                        "title": final_title,
                        "description": final_desc,
                        "tags": api_tags,
                        "categoryId": category_id,
                        "defaultLanguage": "ru",
                        "defaultAudioLanguage": "ru",
                    },
                    "status": {
                        "privacyStatus": vis_code,
                        "selfDeclaredMadeForKids": False,
                    },
                }

                media = MediaFileUpload(
                    str(video_path),
                    mimetype="video/mp4",
                    chunksize=4 * 1024 * 1024,
                    resumable=True,
                )
                try:
                    request = youtube.videos().insert(
                        part="snippet,status",
                        body=body,
                        media_body=media,
                    )

                    log(f"⬆️ Загружаю в YouTube Shorts через API v3: «{final_title}» ({video_path.name})...")

                    response = None
                    retries = 0
                    last_pct = -1
                    while response is None:
                        try:
                            status, response = request.next_chunk()
                            if status is not None:
                                pct = int(status.progress() * 100)
                                if pct >= last_pct + 25:
                                    log(f"⬆️ Загрузка на YouTube: {pct}%...")
                                    last_pct = pct
                        except HttpError as he:
                            code = getattr(he.resp, "status", 0)
                            if code in (500, 502, 503, 504) and retries < 4:
                                retries += 1
                                time.sleep(2.0 * retries)
                                continue
                            err_text = he.content.decode("utf-8", errors="replace") if getattr(he, "content", None) else str(he)
                            if "quotaExceeded" in err_text:
                                return UploadResult(
                                    ok=False,
                                    error="Исчерпана дневная квота YouTube Data API v3 (обновится завтра по тихоокеанскому времени).",
                                )
                            if "uploadLimitExceeded" in err_text:
                                return UploadResult(
                                    ok=False,
                                    error="YouTube временно ограничил число загрузок на канал за сутки (uploadLimitExceeded).",
                                )
                            if code == 401:
                                self._save_auth_info(False)
                                return UploadResult(
                                    ok=False,
                                    needs_login=True,
                                    error="Токен YouTube устарел — нажмите «🔑 Авторизовать YouTube (OAuth)» заново.",
                                )
                            return UploadResult(ok=False, error=f"Ошибка YouTube API ({code}): {err_text[:320]}")
                        except Exception as exc:
                            if retries < 3:
                                retries += 1
                                time.sleep(2.0 * retries)
                                continue
                            return UploadResult(ok=False, error=f"Сетевая ошибка при загрузке: {exc!r}")
                finally:
                    # Обязательно закрываем файловый дескриптор MediaFileUpload перед удалением на Windows!
                    try:
                        st = media.stream()
                        if st is not None:
                            st.close()
                    except Exception:
                        pass

                video_id = str(response.get("id") or "")
                short_url = f"https://youtube.com/shorts/{video_id}" if video_id else ""
                actual_privacy = str((response.get("status") or {}).get("privacyStatus") or vis_code)

                warning = ""
                if vis_code == "public" and actual_privacy == "private":
                    warning = (
                        "YouTube ограничил доступ до Private, потому что проект Google Cloud ещё не прошёл "
                        "YouTube API Audit (заявка бесплатная в Google Cloud / YouTube API Services)."
                    )

                posted_comment = ""
                if video_id and bool(self.cfg.get("auto_comment", True)) and actual_privacy in ("public", "unlisted"):
                    comment_txt = build_shorts_first_comment(
                        channel=ch_name,
                        moment_tag=moment_tag,
                        transcript_text=transcript_text,
                    )
                    posted_comment, comment_warn = self._post_first_comment(
                        youtube=youtube,
                        video_id=video_id,
                        comment_text=comment_txt,
                        log_cb=log,
                    )
                    if comment_warn and not warning:
                        warning = comment_warn

                self._record_upload(video_path, short_url, final_title, actual_privacy, auto=is_auto)

                deleted_local = False
                if video_id and bool(self.cfg.get("delete_after_upload", True)):
                    self.delete_clip_and_siblings(video_path, cleanup_all_other_clips=bool(is_auto))
                    deleted_local = not video_path.exists()

                return UploadResult(
                    ok=True,
                    video_url=short_url,
                    video_id=video_id,
                    title=final_title,
                    actual_privacy=actual_privacy,
                    first_comment=posted_comment,
                    warning=warning,
                    deleted_local=deleted_local,
                )
            except Exception as exc:
                return UploadResult(ok=False, error=f"Ошибка загрузки через YouTube Data API v3: {exc!r}")
