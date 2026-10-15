"""دانلود مدل ASR (sherpa-onnx export) برای دیکته‌یار.

CLI همین منطق را صدا می‌زند که اپ هم در دیالوگ اولین اجرا استفاده می‌کند:

    python download_model.py                 # مدل پیش‌فرض (شنوا کوچیک)
    python download_model.py nemotron-farsi  # مدل دقیق‌تر (نیموترون)
    python download_model.py --list          # فهرست مدل‌ها

خروجی برای شنوا: model/model.onnx و model/tokens.txt
خروجی برای نیموترون: model/nemotron/{encoder,decoder,joiner}.int8.onnx + tokens.txt
(فهرست/فایل‌های هر مدل در app/models_catalog.py است)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# کنسول ویندوز به‌طور پیش‌فرض cp1252 است — متن فارسی را UTF-8 کن
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import model_download, models_catalog  # noqa: E402


def _list_models() -> int:
    print("مدل‌های موجود:")
    for s in models_catalog.all_models():
        mark = " *پیش‌فرض" if s.id == models_catalog.DEFAULT_MODEL_ID else ""
        print(f"  - {s.id}  [{s.name}] {s.approx_size}{mark}")
        print(f"      {s.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--list", "-l", "list"):
        return _list_models()

    model_id = args[0] if args else models_catalog.DEFAULT_MODEL_ID
    spec = models_catalog.get(model_id)
    if spec.id != model_id:
        print(f"✗ مدل ناشناخته: {model_id}")
        return _list_models()

    base = REPO_ROOT / "model"
    target = spec.dir(base)

    print("=" * 60)
    print(f"دانلود مدل {spec.name} ({spec.id}) — دیکته‌یار")
    print("=" * 60)

    if model_download.is_complete(base, spec):
        first = target / spec.files[0].name
        print(f"✓ مدل از قبل موجود است: {first}")
        print("  برای دانلود مجدد، فایل‌ها را حذف کنید.")
        return 0

    def cb(name: str, done: int, total: int):
        if total:
            sys.stdout.write(f"\r  {name}: {model_download.fmt_size(done)} / "
                             f"{model_download.fmt_size(total)} ({done * 100 // total}%)   ")
            sys.stdout.flush()

    ok, msg = model_download.download_for(base, spec, cb)
    print()
    if not ok:
        print(f"✗ {msg}")
        hf_dir = spec.files[0].hf_url.rsplit("/", 1)[0]
        print(f"راه دستی: از {hf_dir}")
        names = "، ".join(f.name for f in spec.files)
        print(f"این فایل‌ها را بگیرید و در پوشه {target} بگذارید: {names}")
        print("(در ایران ممکن است به VPN/پروکسی نیاز باشد)")
        return 1

    print("✓ همه فایل‌ها آماده‌اند — اپ را با `python app/main.py` اجرا کنید.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
