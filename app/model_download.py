"""دانلود مدل «شنوا کوچیک v1.0» (sherpa-onnx export) — هسته مشترک.

هم download_model.py (CLI) و هم دیالوگ اولین اجرای اپ از این ماژول
استفاده می‌کنند. ترتیب منابع:
  1. GitHub Releases ریپوی پروژه
  2. HuggingFace: Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx (منبع اصلی)
"""
from __future__ import annotations


import urllib.error
import urllib.request
from pathlib import Path

# مخزن گیت‌هاب پروژه — assetهای مدل از Release همین تگ دانلود می‌شوند
GITHUB_REPO = "nimaone/persian-whisper-flow"
GITHUB_RELEASE_TAG = "v1.1.0"
HF_BASE = "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/resolve/main"

FILE_NAMES = ("model.onnx", "tokens.txt")
EXPECTED_MIN = {"model.onnx": 1_000_000, "tokens.txt": 10_000}


def model_files(model_dir: Path) -> list[Path]:
    return [model_dir / n for n in FILE_NAMES]


def is_complete(model_dir: Path) -> bool:
    return all(p.exists() for p in model_files(model_dir))


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, progress_cb=None) -> bool:
    """دانلود با نوشتن در فایل موقت کنار مقصد و جای‌گذاری اتمیک.

    نکته: فایل موقت باید در همان درایو مقصد باشد — os.replace بین دو
    درایو OSError می‌دهد (باگ واقعی اولین تست: Temp در C:، مقصد در J:).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dikteyar-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total and total < EXPECTED_MIN.get(dest.name, 1):
                return False
            done = 0
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total or done)
            tmp.replace(dest)
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def download_model(model_dir: Path, progress_cb=None) -> tuple[bool, str]:
    """دانلود فایل‌های ناقص مدل. خروجی: (موفق، پیام)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in FILE_NAMES:
        dest = model_dir / name
        if dest.exists():
            continue

        def cb(done: int, total: int, _name=name):
            if progress_cb:
                progress_cb(_name, done, total)

        url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{name}"
        if not _download(url, dest, cb):
            url = f"{HF_BASE}/{name}"
            if not _download(url, dest, cb):
                failed.append(name)

    if failed:
        return False, "دانلود ناموفق برای: " + ", ".join(failed)
    return True, "مدل کامل است"
