"""دانلود فایل‌های مدل تشخیص گفتار — هسته مشترک دانلود چندمدلی.

هم `download_model.py` (CLI) و هم صفحه‌ی «اولین اجرا» و تنظیمات از این
ماژول استفاده می‌کنند. هر مدل از کاتالوگ `app.models_catalog` گرفته می‌شود؛
ترتیب منابع هر فایل:
  1. GitHub Releases ریپوی پروژه (اگر فایل روی Release باشد)
  2. HuggingFace (منبع اصلی)
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from app.models_catalog import (
    DEFAULT_MODEL_ID,
    ModelSpec,
    default_model,
    github_url,
)

__all__ = [
    "DEFAULT_MODEL_ID", "FILE_NAMES", "EXPECTED_MIN",
    "model_files", "is_complete", "download_model", "download_for",
    "model_dir_for", "fmt_size",
]

_default = default_model()

# سازگاری با نسخه‌ی تک‌مدلی: نام‌ها/کف سایز فایل‌های مدل پیش‌فرض
FILE_NAMES = tuple(f.name for f in _default.files)
# کف سایز — فقط برای وقتی سرور Content-Length نمی‌دهد. مدل واقعی ~۴۴۰MB
# است؛ کف بالاتر از هر قطعیِ پروکسی نگه داشته شده تا فایل ناقص کامل به
# نظر نرسد (قطعیِ واقعی: پاسخ در ۴.۲MB تمام شد و بی‌صدا رد شد).
EXPECTED_MIN = {f.name: f.min_size for f in _default.files}


def model_dir_for(base: Path, spec: ModelSpec | None = None) -> Path:
    """پوشه‌ی واقعی مدل (پیش‌فرض: مدل پیش‌فرض) زیر پوشه‌ی مدل فعال."""
    return (spec or _default).resolve_dir(Path(base))


def model_files(model_dir: Path, spec: ModelSpec | None = None) -> list[Path]:
    spec = spec or _default
    d = spec.resolve_dir(Path(model_dir))
    return [d / f.name for f in spec.files]


def is_complete(model_dir: Path, spec: ModelSpec | None = None) -> bool:
    return all(p.exists() for p in model_files(model_dir, spec))


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, progress_cb=None,
              min_size: int | None = None) -> bool:
    """دانلود با نوشتن در فایل موقت کنار مقصد و جای‌گذاری اتمیک.

    نکته: فایل موقت باید در همان درایو مقصد باشد — os.replace بین دو
    درایو OSError می‌دهد (باگ واقعی اولین تست: Temp در C:، مقصد در J:).

    کف سایز/کامل‌بودن: پروکسی یا ISP گاهی پاسخ را میان‌راه قطع می‌کند و
    تراکنش بی‌خطا تمام به نظر می‌رسد. با Content-Length، حجم دریافتی با
    اعلام‌شده مقایسه می‌شود؛ بدون آن، کف سایز نگهبان است. در هر دو حالت
    فایل ناقص جای‌گذاری نمی‌شود تا اپ با مدل ناقص لود نشود.
    """
    floor = EXPECTED_MIN.get(dest.name, 1) if min_size is None else min_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dikteyar-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total and total < floor:
                return False
            done = 0
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total or done)
            if (total and done != total) or done < floor:
                # پاسخ ناقص — فایل موقت دور ریخته می‌شود تا جای‌گذاری نشود
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(dest)
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def download_for(model_dir: Path, spec: ModelSpec | None = None,
                 progress_cb=None) -> tuple[bool, str]:
    """دانلود فایل‌های ناقص یک مدل. خروجی: (موفق، پیام)."""
    spec = spec or _default
    target = spec.resolve_dir(Path(model_dir))
    target.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for mf in spec.files:
        dest = target / mf.name
        if dest.exists():
            continue

        def cb(done: int, total: int, _name=mf.name):
            if progress_cb:
                progress_cb(_name, done, total)

        ok = False
        if mf.on_github:
            ok = _download(github_url(spec, mf), dest, cb, mf.min_size)
        if not ok:
            ok = _download(mf.hf_url, dest, cb, mf.min_size)
        if not ok:
            failed.append(mf.name)

    if failed:
        return False, "دانلود ناموفق برای: " + ", ".join(failed)
    return True, "مدل کامل است"


def download_model(model_dir: Path, progress_cb=None) -> tuple[bool, str]:
    """سازگاری با نسخه‌ی تک‌مدلی: دانلود مدل پیش‌فرض."""
    return download_for(model_dir, _default, progress_cb)
