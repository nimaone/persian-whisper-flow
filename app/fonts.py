"""فونت وزیرمتن — باندل‌شده در assets/fonts، بارگذاری در Tkinter."""
from __future__ import annotations

from pathlib import Path

import tkinter.font as tkfont

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

_regular: str | None = None
_family = "Vazirmatn"


def _font_path(weight: str) -> Path:
    return _FONT_DIR / f"Vazirmatn-{weight}.ttf"


def ensure_registered() -> str:
    """فونت را یک‌بار برای این پروسه نصب می‌کند و نام خانواده را برمی‌گرداند.

    نصب فرآیندی (AddFontResourceEx) به Tk ربطی ندارد؛ در هر لحظه‌ای از
    عمر پروسه قابل انجام است. اگر شکست خورد به Tahoma برمی‌گردد.
    """
    global _regular
    if _regular:
        return _regular
    # نصب همه‌ی وزن‌ها — Tk با نام خانواده انتخاب می‌کند
    any_ok = False
    for w in ("Regular", "Bold", "Medium", "Light"):
        p = _font_path(w)
        if p.exists() and _install_windows(str(p.resolve())):
            any_ok = True
    if any_ok:
        _regular = _family
        return _family
    return "Tahoma"


def _install_windows(font_path: str):
    """نصب فونت فقط برای پروسه‌ی فعلی (بدون کپی در C:\\Windows\\Fonts)."""
    try:
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.windll.gdi32
        gdi32.AddFontResourceExW.restype = ctypes.c_int
        gdi32.AddFontResourceExW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
        FR_PRIVATE = 0x10  # فقط این پروسه، بدون ثبت سیستم
        res = gdi32.AddFontResourceExW(font_path, FR_PRIVATE, None)
        # FR_PRIVATE با پاک‌شدن پس از خروج پروسه
        return res > 0
    except Exception:
        return False


def get(family_backup: str = "Tahoma") -> str:
    """نام خانواده‌ی فونت فارسی قابل استفاده در Tk."""
    return ensure_registered() or family_backup
