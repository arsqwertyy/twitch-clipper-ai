import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from audio_analyzer import AudioPeakTracker
from chat_monitor import normalize_channel
from ffmpeg_utils import WIN_NO_WINDOW, get_ffmpeg_path, get_streamlink_cmd

TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_PUBLIC_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"


def _seg_times(path: Path, fallback_duration: float) -> Optional[tuple[float, float, int]]:
    """
    Возвращает (start, end, size) сегмента по времени ФС:
    start = время создания файла (st_birthtime на Windows), end = время последней записи.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    end = st.st_mtime
    start = getattr(st, "st_birthtime", None)
    if start is None or start > end or (end - start) > fallback_duration * 6:
        start = end - fallback_duration
    return start, end, st.st_size


class StreamRecorder:
    """
    Отслеживает онлайн стрима через Streamlink и непрерывно пишет эфир
    в кольцевой буфер коротких .ts сегментов (без перекодирования, -c copy).
    """

    def __init__(
        self,
        channel: str,
        buffer_dir: Path,
        stream_cfg: dict,
        on_status_change: Optional[Callable[[bool, str], None]] = None,
    ):
        self.channel = normalize_channel(channel)
        self.url = f"https://www.twitch.tv/{self.channel}"
        self.buffer_dir = buffer_dir.resolve()
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        self.quality: str = str(stream_cfg.get("quality", "best"))
        self.segment_duration: int = int(
            stream_cfg.get("segment_duration", stream_cfg.get("segment_Duration", 5))
        )
        self.buffer_keep_seconds: float = float(stream_cfg.get("buffer_keep_seconds", 150))
        check_val = float(stream_cfg.get("check_interval", 300))
        self.check_interval: float = 300.0 if check_val < 60 else check_val

        self.on_status_change = on_status_change
        self.is_live = False
        self._running = False
        self._active_locks = 0

        self._streamlink_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._session_id = 0
        self._run_id = int(time.time())
        self.audio_tracker = AudioPeakTracker()

    def get_recent_audio_stats(self) -> tuple[float, float]:
        """Возвращает (audio_z_score, peak_db) за последние 10 секунд записанного эфира."""
        if not self.is_live:
            return 0.0, -60.0
        now = time.time()
        segs = self.get_segments_for_range(now - 14.0, now)
        return self.audio_tracker.analyze_recent_segments(segs)

    @property
    def _quality_arg(self) -> str:
        return f"{self.quality},best" if self.quality != "best" else "best"

    def _notify(self, is_live: bool, msg: str) -> None:
        if self.on_status_change:
            try:
                self.on_status_change(is_live, msg)
            except Exception:
                pass

    def lock_buffer(self) -> None:
        """Блокирует удаление сегментов на время сборки клипа."""
        self._active_locks += 1

    def unlock_buffer(self) -> None:
        self._active_locks = max(0, self._active_locks - 1)

    def clear_old_files_on_startup(self) -> None:
        """Очищает старые сегменты и временные файлы от прошлых запусков."""
        for pattern in ("seg_*.ts", "concat_*.txt", "raw_*.mp4", "subs_*.ass"):
            for f in self.buffer_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

    def _list_segments(self) -> list[tuple[Path, float, float, int]]:
        """Все сегменты буфера, отсортированные по времени окончания."""
        items = []
        for p in self.buffer_dir.glob("seg_*.ts"):
            t = _seg_times(p, self.segment_duration)
            if t is not None:
                items.append((p, *t))
        items.sort(key=lambda x: x[2])
        return items

    def get_segments_for_range(
        self,
        start_ts: float,
        end_ts: float,
        min_segments: int = 0,
    ) -> list[Path]:
        """
        Упорядоченный список ЗАВЕРШЁННЫХ .ts сегментов, пересекающих интервал [start_ts, end_ts].
        Если задан min_segments (например, 5 сегментов = 25 сек для клипа), при нехватке сегментов
        добирает соседние непрерывные сегменты из буфера, а если в буфере меньше 4 сегментов (<20с) —
        возвращает пустой список, чтобы никогда не создавать короткие 5-7 секундные обрезки.
        """
        items = self._list_segments()
        if not items:
            return []

        # Последний сегмент прямо сейчас дописывается ffmpeg'ом — его не берём
        if self.is_live and self._ffmpeg_proc is not None and self._ffmpeg_proc.poll() is None:
            items = items[:-1]

        valid_items = [it for it in items if it[3] >= 1024]
        if not valid_items:
            return []

        selected_idx: list[int] = []
        fallback_idx: list[int] = []
        for idx, (_path, seg_start, seg_end, _size) in enumerate(valid_items):
            if seg_end > start_ts and seg_start < end_ts:
                fallback_idx.append(idx)
                overlap = min(seg_end, end_ts) - max(seg_start, start_ts)
                seg_len = max(0.1, seg_end - seg_start)
                if overlap >= min(2.0, seg_len * 0.4):
                    selected_idx.append(idx)

        chosen_idx = selected_idx if selected_idx else fallback_idx
        if not chosen_idx:
            return []

        if min_segments > 0:
            lo = chosen_idx[0]
            hi = chosen_idx[-1]
            # Добираем соседние сегменты до min_segments (обычно 5-6 сегментов = 25-30 сек),
            # проверяя, что между соседними сегментами нет разрыва сессии (> 15 сек)
            while (hi - lo + 1) < min_segments:
                can_left = lo > 0 and (valid_items[lo][1] - valid_items[lo - 1][2]) <= 15.0
                can_right = (hi + 1) < len(valid_items) and (valid_items[hi + 1][1] - valid_items[hi][2]) <= 15.0
                if can_left:
                    lo -= 1
                elif can_right:
                    hi += 1
                else:
                    break
            chosen_idx = list(range(lo, hi + 1))
            if len(chosen_idx) < max(4, min_segments - 1):
                return []

        return [valid_items[i][0] for i in chosen_idx]

    async def run(self) -> None:
        """Главный цикл мониторинга онлайна и записи сегментов."""
        self._running = True
        self.clear_old_files_on_startup()
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        try:
            while self._running:
                online = await asyncio.to_thread(self._check_stream_online)
                if not self._running:
                    break
                if not online:
                    if self.is_live:
                        self.is_live = False
                        if self._active_locks == 0:
                            self.clear_old_files_on_startup()
                        self._notify(False, f"Стрим {self.channel} завершился. Ожидание нового эфира...")
                    else:
                        wait_txt = (
                            f"{int(self.check_interval // 60)} мин"
                            if self.check_interval >= 60
                            else f"{int(self.check_interval)} сек"
                        )
                        self._notify(
                            False,
                            f"Канал {self.channel} сейчас оффлайн. Повторная проверка через {wait_txt}...",
                        )
                    await asyncio.sleep(self.check_interval)
                    continue

                self._session_id += 1
                self.is_live = True
                self._notify(True, f"Стрим {self.channel} ОНЛАЙН! Идет запись буфера ({self.quality})...")

                await self._record_stream_session(self._session_id)

                if self._running:
                    await asyncio.sleep(3.0)
        finally:
            cleanup_task.cancel()
            self._terminate_processes()

    def stop(self) -> None:
        self._running = False
        self._terminate_processes()
        if self._active_locks == 0:
            self.clear_old_files_on_startup()

    def _check_stream_online(self) -> bool:
        """
        Мгновенно (~0.15с) проверяет онлайн через Twitch GQL без запуска тяжёлых подпроцессов
        для оффлайн-каналов, а при необходимости подтверждает поток через Streamlink.
        """
        gql_query = "query($login:String!){user(login:$login){stream{id type}}}"
        gql_ok = False
        gql_live = False
        try:
            r = requests.post(
                TWITCH_GQL_URL,
                headers={"Client-Id": TWITCH_PUBLIC_CLIENT_ID},
                json={"query": gql_query, "variables": {"login": self.channel}},
                timeout=6,
            )
            if r.status_code == 200:
                data = r.json()
                user = (data.get("data") or {}).get("user")
                if isinstance(user, dict):
                    gql_ok = True
                    st = user.get("stream")
                    gql_live = bool(isinstance(st, dict) and st.get("id"))
        except Exception:
            gql_ok = False

        # Если GQL точно ответил, что стрима сейчас нет — не тратим ресурсы на запуск streamlink.exe
        if gql_ok and not gql_live:
            return False

        cmd = [*get_streamlink_cmd(), "--stream-url", self.url, self._quality_arg]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                creationflags=WIN_NO_WINDOW,
            )
            out = res.stdout.strip()
            return res.returncode == 0 and out.startswith("http")
        except Exception:
            return gql_live

    async def _record_stream_session(self, session_id: int) -> None:
        """Запускает связку streamlink (stdout) -> ffmpeg (сегменты .ts)."""
        ffmpeg_exe = get_ffmpeg_path()
        seg_pattern = str(self.buffer_dir / f"seg_{self._run_id}_{session_id:02d}_%06d.ts")

        streamlink_cmd = [
            *get_streamlink_cmd(),
            "--twitch-low-latency",  # меньше задержка -> видео лучше совпадает по времени с реакцией чата
            "--stdout",
            "--loglevel",
            "error",
            self.url,
            self._quality_arg,
        ]

        ffmpeg_cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_duration),
            "-segment_format",
            "mpegts",
            "-reset_timestamps",
            "1",
            seg_pattern,
        ]

        try:
            self._streamlink_proc = subprocess.Popen(
                streamlink_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**6,
                creationflags=WIN_NO_WINDOW,
            )
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self._streamlink_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=WIN_NO_WINDOW,
            )
            if self._streamlink_proc.stdout:
                self._streamlink_proc.stdout.close()

            session_start_ts = time.time()
            while self._running:
                if self._streamlink_proc.poll() is not None or self._ffmpeg_proc.poll() is not None:
                    break
                # Watchdog: если поток реально завис и новые .ts сегменты не пишутся дольше 45 секунд — перезапускаем
                now_ts = time.time()
                if (now_ts - session_start_ts) > 35.0:
                    segs = self._list_segments()
                    latest_mtime = segs[-1][2] if segs else session_start_ts
                    if (now_ts - latest_mtime) > 45.0:
                        break
                await asyncio.sleep(1.0)
        finally:
            self._terminate_processes()

    def _terminate_processes(self) -> None:
        # Сначала streamlink (источник), затем даём ffmpeg корректно дописать последний сегмент
        for proc in (self._streamlink_proc, self._ffmpeg_proc):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._streamlink_proc = None
        self._ffmpeg_proc = None

    async def _cleanup_loop(self) -> None:
        """Удаляет сегменты старше buffer_keep_seconds (кольцевой буфер)."""
        while True:
            await asyncio.sleep(10.0)
            if self._active_locks > 0:
                continue
            cutoff = time.time() - self.buffer_keep_seconds
            items = self._list_segments()
            for path, _start, seg_end, _size in items[:-3]:
                if seg_end < cutoff:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
