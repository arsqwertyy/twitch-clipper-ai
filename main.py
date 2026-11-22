import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from chat_monitor import ClipEvent, TwitchChatMonitor, WindowStats, parse_channels_list
from emote_manager import ChannelEmoteManager
from ffmpeg_utils import setup_nvidia_dlls
from stream_recorder import StreamRecorder
from subtitle_burner import SubtitleGenerator
from trainer import AutoTrainer
from video_clipper import VideoClipper
from youtube_uploader import YouTubeShortsUploader, compute_clip_hype_rank

setup_nvidia_dlls()

console = Console()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


async def main_async(args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / args.config
    cfg = load_config(cfg_path)

    yt_uploader = YouTubeShortsUploader(base_dir, cfg.get("youtube"))
    if args.youtube_login:
        console.print("[bold red]📺 Открываю ваш обычный браузер для официального OAuth 2.0 входа YouTube Data API v3...[/bold red]")
        ok, msg = await asyncio.to_thread(
            yt_uploader.authorize_oauth,
            lambda m: console.print(f"[cyan]{escape(m)}[/cyan]"),
        )
        console.print(f"[{'bold green' if ok else 'bold yellow'}]{'✅' if ok else '⚠️'} {escape(msg)}[/{'bold green' if ok else 'bold yellow'}]")
        return

    raw_channels = args.channel or cfg.get("channels") or cfg.get("channel") or ""
    channels = parse_channels_list(raw_channels)
    if not channels:
        console.print("[bold yellow]Укажите один или несколько каналов Twitch (через запятую)![/bold yellow]")
        entered = console.input("[bold cyan]Введите ники стримеров: [/bold cyan]").strip()
        channels = parse_channels_list(entered)
        if not channels:
            console.print("[bold red]Каналы не указаны. Выход.[/bold red]")
            return

    paths_cfg = cfg.get("paths", {})
    base_buffer_dir = base_dir / paths_cfg.get("buffer_dir", "buffer")
    clips_dir = base_dir / paths_cfg.get("clips_dir", "clips")

    stream_cfg = cfg.get("stream", {})
    detector_cfg = cfg.get("detector", {})
    video_cfg = cfg.get("video", {})
    sub_cfg = cfg.get("subtitles", {})
    profiles = cfg.setdefault("channel_profiles", {})

    emote_mgr = ChannelEmoteManager(base_dir)
    await asyncio.to_thread(
        emote_mgr.sync_channels,
        channels,
        False,
        lambda m: console.print(f"[cyan]{escape(m)}[/cyan]"),
    )

    console.print("[dim]Инициализация нейросети Faster-Whisper и проверка GPU...[/dim]")
    sub_generator = SubtitleGenerator(sub_cfg=sub_cfg, video_cfg=video_cfg)
    if sub_generator.enabled:
        await asyncio.to_thread(sub_generator.load_model)

    clipper = VideoClipper(
        buffer_dir=base_buffer_dir,
        clips_dir=clips_dir,
        video_cfg=video_cfg,
        sub_generator=sub_generator,
    )

    if args.train:
        console.print("[bold magenta]🧠 Запуск пред-стримового обучения и тестирования для каналов:[/bold magenta]", ", ".join(channels))
        trainer = AutoTrainer(base_dir, cfg)
        for ch in channels:
            rep = await asyncio.to_thread(
                trainer.train_and_test_channel,
                ch,
                clipper,
                True,
                lambda msg: console.print(f"[cyan]{escape(msg)}[/cyan]"),
            )
            prev_cam = (profiles.get(ch) or {}).get("webcam_crop")
            prof_dict = rep.to_profile_dict()
            if "webcam_crop" not in prof_dict and prev_cam:
                prof_dict["webcam_crop"] = prev_cam
            profiles[ch] = prof_dict
        save_config(cfg_path, cfg)
        console.print("[bold green]✅ Обучение и тест завершены! Профили сохранены в config.yaml.[/bold green]")
        return

    clip_queue: asyncio.Queue[tuple[StreamRecorder, ClipEvent]] = asyncio.Queue()
    recorders: dict[str, StreamRecorder] = {}
    monitors: dict[str, TwitchChatMonitor] = {}

    for ch in channels:
        ch_buf = base_buffer_dir / ch

        def make_status_cb(c_name: str):
            def _cb(is_live: bool, msg: str) -> None:
                now_str = datetime.now().strftime("%H:%M:%S")
                color = "bold green" if is_live else "yellow"
                console.print(f"[{now_str}] [{color}]🎥 [{escape(c_name)}] {escape(msg)}[/{color}]")
            return _cb

        rec = StreamRecorder(
            channel=ch,
            buffer_dir=ch_buf,
            stream_cfg=stream_cfg,
            on_status_change=make_status_cb(ch),
        )
        recorders[ch] = rec

        def on_window_stats(stats: WindowStats) -> None:
            r = recorders.get(stats.channel)
            if not r or not r.is_live:
                return
            now_str = datetime.now().strftime("%H:%M:%S")
            sample_str = f" | Топ: {escape(stats.top_messages[0][:45])}" if stats.top_messages else ""
            react_pct = int(round(stats.reaction_ratio * 100))
            if stats.is_triggered:
                console.print(
                    f"[{now_str}] [bold red]🔥 [{escape(stats.channel)}] {escape(stats.moment_tag or 'ВЗРЫВ ЧАТА')}![/bold red] "
                    f"Сообщ: [bold]{stats.msg_count}[/bold] ({react_pct}% эмоц.) | Хайп: [bold]{stats.hype_score:.1f}[/bold] "
                    f"(норма: {stats.mean_score:.1f}, Z: {stats.z_score:+.2f}, Z_эмоц: {stats.reaction_z:+.2f}){sample_str}"
                )
            else:
                console.print(
                    f"[{now_str}] [cyan]💬 [{escape(stats.channel)}] Чат (10с):[/cyan] {stats.msg_count:2d} сообщ. ({react_pct:2d}% эмоц.) | "
                    f"Хайп: {stats.hype_score:5.1f} (ср: {stats.mean_score:4.1f}, Z: {stats.z_score:+.2f})"
                    f"[dim]{sample_str}[/dim]"
                )

        def make_clip_cb(r_obj: StreamRecorder):
            def _cb(event: ClipEvent) -> None:
                if r_obj.is_live:
                    clip_queue.put_nowait((r_obj, event))
            return _cb

        mon = TwitchChatMonitor(
            channel=ch,
            detector_cfg=detector_cfg,
            on_clip_detected=make_clip_cb(rec),
            on_window_stats=on_window_stats,
            learned_profile=profiles.get(ch),
            get_audio_stats=rec.get_recent_audio_stats,
            is_stream_live=lambda r=rec: r.is_live,
        )
        monitors[ch] = mon

    nvenc_str = "Да (NVIDIA h264_nvenc)" if clipper.nvenc_available else "Нет (CPU libx264)"
    whisper_str = (
        f"{sub_generator.model_size} на {sub_generator.active_device}"
        if sub_generator.enabled
        else "Выключены"
    )

    console.print(
        Panel.fit(
            f"[bold magenta]Мульти-нарезчик Twitch -> TikTok (9:16)[/bold magenta]\n"
            f"• Стримеры ({len(channels)}): [bold cyan]{', '.join(channels)}[/bold cyan]\n"
            f"• Обученных профилей: [bold green]{sum(1 for c in channels if c in profiles)}/{len(channels)}[/bold green]\n"
            f"• Кодировщик видео: [bold green]{nvenc_str}[/bold green]\n"
            f"• Субтитры Whisper: [bold green]{whisper_str}[/bold green]\n"
            f"• Папка готовых клипов: [underline]{clips_dir}[/underline]",
            title="🚀 Запущено",
            border_style="bright_blue",
        )
    )

    last_global_clip_ts = 0.0

    async def clip_worker() -> None:
        nonlocal last_global_clip_ts
        while True:
            rec, event = await clip_queue.get()
            try:
                rec.lock_buffer()
                is_test = bool(event.sample_messages and "--test-clip" in event.sample_messages[0])
                segments = rec.get_segments_for_range(
                    event.start_ts,
                    event.end_ts,
                    min_segments=0 if is_test else 5,
                )
                if not segments:
                    console.print(
                        f"[yellow]⚠️ [{escape(rec.channel)}] Недостаточно сегментов для полноценного клипа (<20 сек в буфере).[/yellow]"
                    )
                    continue

                approx_sec = len(segments) * rec.segment_duration
                m_tag = getattr(event, "moment_tag", "🔥 Хайп")
                v_sc = float(getattr(event, "viral_score", 0.0))
                if not is_test:
                    if bool(yt_uploader.cfg.get("enabled", False)):
                        reached, cnt, mx = yt_uploader.is_daily_limit_reached()
                        if reached:
                            console.print(f"[dim]⏭ [{escape(rec.channel)}] Дневной лимит YouTube ({cnt}/{mx}) достигнут.[/dim]")
                            continue
                    g_cool = float(detector_cfg.get("global_cooldown_seconds", 30.0))
                    if g_cool > 0 and last_global_clip_ts > 0 and (time.time() - last_global_clip_ts) < g_cool and v_sc < 88.0:
                        console.print(f"[dim]⏭ [{escape(rec.channel)}] Общая пауза между каналами (индекс {v_sc:.0f}/100).[/dim]")
                        continue
                console.print(
                    f"\n[bold yellow]✂️ [{escape(rec.channel)}] Сборка клипа ({escape(m_tag)}, ~{approx_sec} сек, пик: {event.peak_score:.1f}, индекс: {v_sc:.0f}/100)...[/bold yellow]"
                )
                target_dur = max(25.0, float(detector_cfg.get("max_clip_seconds", 37.0)))
                out_path, words_cnt, elapsed = await asyncio.to_thread(
                    clipper.create_clip_from_segments,
                    segments,
                    rec.channel,
                    event.peak_score,
                    (profiles.get(rec.channel) or {}).get("webcam_crop"),
                    target_dur,
                    not is_test,
                )
                if out_path:
                    if not is_test:
                        last_global_clip_ts = time.time()
                    ban_line = (
                        f"   📢 Баннер: [bold yellow]{escape(clipper.last_banner_pause_info)}[/bold yellow]\n"
                        if clipper.last_banner_pause_info
                        else ""
                    )
                    console.print(
                        f"[bold green]✅ [{escape(rec.channel)}] Готов клип за {elapsed:.1f} сек![/bold green]\n"
                        f"   📁 Файл: [bold white]{escape(str(out_path))}[/bold white]\n"
                        f"{ban_line}"
                        f"   📝 Распознано слов (Whisper): [bold cyan]{words_cnt}[/bold cyan]\n"
                    )
                    if not is_test and bool(yt_uploader.cfg.get("enabled", False)):
                        v_sc = float(getattr(event, "viral_score", 0.0))
                        can_up, why_not = yt_uploader.check_auto_upload_eligibility(out_path, v_sc)
                        if can_up:
                            transcript = clipper.last_transcript_text
                            async def _bg_yt_upload(vp=out_path, ch=rec.channel, tag=m_tag, tr=transcript):
                                res = await asyncio.to_thread(
                                    yt_uploader.upload_video,
                                    vp,
                                    None,
                                    None,
                                    ch,
                                    tag,
                                    tr,
                                    None,
                                    True,
                                    lambda m: console.print(f"[cyan]{escape(m)}[/cyan]"),
                                )
                                if res.ok:
                                    del_part = " (удалено с ПК)" if res.deleted_local else ""
                                    console.print(
                                        f"[bold green]✅📺 Опубликовано в YouTube Shorts [{escape(res.actual_privacy)}]: {escape(res.title)} {escape(res.video_url)}{del_part}[/bold green]"
                                    )
                                    if res.warning:
                                        console.print(f"[bold yellow]⚠️📺 {escape(res.warning)}[/bold yellow]")
                                else:
                                    console.print(f"[bold red]❌📺 Ошибка YouTube: {escape(res.error)}[/bold red]")
                            asyncio.create_task(_bg_yt_upload())
                        else:
                            console.print(f"[dim]📺 [{escape(rec.channel)}] Авто-заливка YouTube пропущена: {escape(why_not)}[/dim]")
                elif clipper.last_skip_reason:
                    console.print(f"[yellow]⏭ [{escape(rec.channel)}] {escape(clipper.last_skip_reason)}.[/yellow]")
                else:
                    console.print(f"[bold red]❌ [{escape(rec.channel)}] Не удалось собрать клип.[/bold red]")
            except Exception as exc:
                console.print(f"[bold red]❌ [{escape(rec.channel)}] Ошибка: {escape(repr(exc))}[/bold red]")
            finally:
                rec.unlock_buffer()
                clip_queue.task_done()

    async def test_clip_trigger() -> None:
        if not args.test_clip:
            return
        console.print("[bold magenta]🧪 Включен режим --test-clip: ждем 35 секунд записи любого онлайн-стрима...[/bold magenta]")
        live_rec = None
        while live_rec is None:
            for r in recorders.values():
                if r.is_live:
                    live_rec = r
                    break
            await asyncio.sleep(1.0)
        await asyncio.sleep(35.0)
        now = time.time()
        await clip_queue.put(
            (
                live_rec,
                ClipEvent(
                    start_ts=now - 30.0,
                    end_ts=now,
                    peak_score=99.0,
                    z_score=5.0,
                    total_messages=0,
                    sample_messages=["Тестовый клип (--test-clip)"],
                    channel=live_rec.channel,
                ),
            )
        )

    worker_task = asyncio.create_task(clip_worker())
    test_task = asyncio.create_task(test_clip_trigger())

    all_tasks = [r.run() for r in recorders.values()] + [m.run() for m in monitors.values()]
    try:
        await asyncio.gather(*all_tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for r in recorders.values():
            r.stop()
        for m in monitors.values():
            m.stop()
        worker_task.cancel()
        test_task.cancel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Мульти-нарезчик стримов Twitch в формат TikTok с авто-обучением и субтитрами Whisper."
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=str,
        default="",
        help="Один или несколько ников стримеров через запятую (например: bratishkinoff,evelone192)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу настроек (по умолчанию config.yaml)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Провести предварительное обучение по прошлым стримам (VOD) и сделать тестовый клип",
    )
    parser.add_argument(
        "--test-clip",
        action="store_true",
        help="Сделать пробный клип через 35 секунд после начала записи эфира",
    )
    parser.add_argument(
        "--youtube-login",
        action="store_true",
        help="Открыть браузер для одноразового входа в YouTube Studio для авто-заливки Shorts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Остановлено пользователем (Ctrl+C).[/bold yellow]")
        sys.exit(0)
