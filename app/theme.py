"""پالت و تایپوگرافی مشترک UI — تیره‌ی خنثی با یک accent.

قاعده: سلسله‌مراتب با وزن/اندازه‌ی فونت ساخته می‌شود، نه رنگ. فقط دو
رنگ عملکردی مجاز است: سبز accent (هماهنگ با آیکون tray) و قرمز ضبط.
"""
from __future__ import annotations

BG = "#1e1e1e"          # پس‌زمینه‌ی پنجره‌ها
SURFACE = "#2a2a2a"     # کارت overlay
SURFACE_2 = "#333333"   # ورودی/دکمه‌ی ثانویه
SURFACE_3 = "#3c3c3c"   # hover ثانویه
BORDER = "#3f3f3f"
DEEP = "#181818"        # نواحی گرافیکی (اسپیکتر/موج)

FG = "#f3f3f3"
FG_DIM = "#9b9b9b"

ACCENT = "#22c55e"
ACCENT_HOVER = "#1eb457"
ON_ACCENT = "#081409"

DANGER = "#e5484d"
DANGER_HOVER = "#cd2d31"
ON_DANGER = "#ffffff"

WARN = "#d9a13c"

_fam: str | None = None


def family() -> str:
    """نام خانواده‌ی فونت فارسی — بار اول فونت را برای پروسه رجیستر می‌کند."""
    global _fam
    if _fam is None:
        from app.fonts import get
        _fam = get()
    return _fam


def asset_path(name: str):
    """مسیر یک فایل در assets/ — اگر نباشد None."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "assets" / name
    return p if p.exists() else None


def apply_icon(win) -> None:
    """آیکون نوار عنوان/تسک‌بار ویندوز از assets/logo.ico — بی‌صدا رد می‌شود."""
    ico = asset_path("logo.ico")
    if ico is None:
        return
    try:
        win.iconbitmap(str(ico))
    except Exception:
        pass
