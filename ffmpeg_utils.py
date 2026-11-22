import os
import shutil
import subprocess
from pathlib import Path

WIN_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def setup_nvidia_dlls() -> None:
    """
    Добавляет пути к DLL библиотекам CUDA (cublas, cudnn), установленным внутри .venv,
    чтобы faster-whisper / ctranslate2 работали на видеокарте NVIDIA без ручной установки CUDA Toolkit.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    venv_base = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "nvidia"
    if not venv_base.exists():
        return

    dll_dirs = []
    for bin_dir in venv_base.glob("*/bin"):
        if bin_dir.is_dir():
            bin_str = str(bin_dir.resolve())
            dll_dirs.append(bin_str)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_str)
                except OSError:
                    pass

    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")


def get_ffmpeg_path() -> str:
    """Находит исполняемый файл ffmpeg (в системе, локальной папке или через imageio-ffmpeg)."""
    local_ffmpeg = Path(__file__).resolve().parent / "bin" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "Не найден ffmpeg! Установите imageio-ffmpeg или добавьте ffmpeg в PATH."
        ) from exc


def get_streamlink_cmd() -> list[str]:
    """Возвращает команду для запуска streamlink из виртуального окружения или PATH."""
    venv_streamlink = Path(__file__).resolve().parent / ".venv" / "Scripts" / "streamlink.exe"
    if venv_streamlink.exists():
        return [str(venv_streamlink)]

    sys_streamlink = shutil.which("streamlink")
    if sys_streamlink:
        return [sys_streamlink]

    return ["python", "-m", "streamlink"]


def check_nvenc_support(ffmpeg_exe: str) -> bool:
    """Проверяет, поддерживает ли видеокарта аппаратное кодирование h264_nvenc."""
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=8, creationflags=WIN_NO_WINDOW)
        return res.returncode == 0
    except Exception:
        return False
