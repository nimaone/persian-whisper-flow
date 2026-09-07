"""دانلود مدل ASR «شنوا کوچیک v1.0» (sherpa-onnx export) برای دیکته‌یار.

CLI همین منطق را صدا می‌زند که اپ هم در دیالوگ اولین اجرا استفاده می‌کند:
    python download_model.py

خروجی: model/model.onnx و model/tokens.txt
(تگ Release که اسکریپت از آن می‌خواند در app/model_download.py است)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from app import model_download  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("دانلود مدل شنوا کوچیک v1.0 (sherpa-onnx) — دیکته‌یار")
    print("=" * 60)

    md = REPO_ROOT / "model"
    if model_download.is_complete(md):
        size = (md / "model.onnx").stat().st_size
        print(f"✓ مدل از قبل موجود است: {md / 'model.onnx'} ({model_download.fmt_size(size)})")
        print("  برای دانلود مجدد، فایل‌ها را حذف کنید.")
        return 0

    last: dict = {}

    def cb(name: str, done: int, total: int):
        if total:
            sys.stdout.write(f"\r  {name}: {model_download.fmt_size(done)} / "
                             f"{model_download.fmt_size(total)} ({done * 100 // total}%)   ")
            sys.stdout.flush()

    ok, msg = model_download.download_model(md, cb)
    print()
    if not ok:
        print(f"✗ {msg}")
        print("راه دستی: از https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx")
        print("فایل‌های model.onnx و tokens.txt را بگیرید و در پوشه model/ بگذارید.")
        print("(در ایران ممکن است به VPN/پروکسی نیاز باشد)")
        return 1

    print("✓ همه فایل‌ها آماده‌اند — اپ را با `python app/main.py` اجرا کنید.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
