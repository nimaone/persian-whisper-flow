"""دانلود مدل ASR «شنوا کوچیک v1.0» (sherpa-onnx export) برای دیکته‌یار.

ترتیب منابع:
  1. GitHub Releases همین ریپو (اگر asset آپلود شده باشد)
  2. HuggingFace: Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx (منبع اصلی)

استفاده:
    python download_model.py

خروجی: model/model.onnx و model/tokens.txt
"""
from __future__ import annotations

import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "model"
MODEL_FILE = MODEL_DIR / "model.onnx"
TOKENS_FILE = MODEL_DIR / "tokens.txt"

# مخزن گیت‌هاب این پروژه — asset های مدل از Release همین تگ دانلود می‌شوند
GITHUB_REPO = "nimaone/persian-whisper-flow"
GITHUB_RELEASE_TAG = "model-v1.0"

HF_BASE = "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/resolve/main"
FILES = {
    "model.onnx": MODEL_FILE,
    "tokens.txt": TOKENS_FILE,
}


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, expected_min_bytes: int = 1_000_000) -> bool:
    """دانلود با نمایش پیشرفت؛ فایل ابتدا در temp نوشته می‌شود (download به downloads/)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dikteyar-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total and total < expected_min_bytes:
                print(f"  ✗ پاسخ خیلی کوچک است ({_fmt_size(total)}) — احتمالاً خطای 404/HTML؛ رد شد")
                return False
            tmp = Path(tempfile.gettempdir()) / f"dikteyar_{dest.name}.part"
            done = 0
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        sys.stdout.write(f"\r  {_fmt_size(done)} / {_fmt_size(total)} ({pct}%)")
                        sys.stdout.flush()
            print()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(dest)
            print(f"  ✓ ذخیره شد: {dest}")
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"  ✗ خطا: {e}")
        return False


def try_github(name: str, dest: Path) -> bool:
    if not GITHUB_REPO:
        return False
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{name}"
    print(f"[GitHub Release] {name}")
    return _download(url, dest)


def try_huggingface(name: str, dest: Path) -> bool:
    url = f"{HF_BASE}/{name}"
    print(f"[HuggingFace] {name}")
    return _download(url, dest, expected_min_bytes=10_000)  # tokens.txt ~12KB


def main() -> int:
    print("=" * 60)
    print("دانلود مدل شنوا کوچیک v1.0 (sherpa-onnx) — دیکته‌یار")
    print("=" * 60)

    if MODEL_FILE.exists() and TOKENS_FILE.exists():
        print(f"✓ مدل از قبل موجود است: {MODEL_FILE} ({_fmt_size(MODEL_FILE.stat().st_size)})")
        print("  برای دانلود مجدد، فایل‌ها را حذف کنید.")
        return 0

    MODEL_DIR.mkdir(exist_ok=True)
    failed = []
    for name, dest in FILES.items():
        if dest.exists():
            print(f"✓ {name} از قبل موجود است")
            continue
        ok = try_github(name, dest) or try_huggingface(name, dest)
        if not ok:
            failed.append(name)

    if failed:
        print("\n✗ دانلود ناموفق برای:", ", ".join(failed))
        print("راه دستی: از https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx")
        print("فایل‌های model.onnx و tokens.txt را بگیرید و در پوشه model/ بگذارید.")
        print("(در ایران ممکن است به VPN/پروکسی نیاز باشد)")
        return 1

    print("\n✓ همه فایل‌ها آماده‌اند — اپ را با `python app/main.py` اجرا کنید.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
