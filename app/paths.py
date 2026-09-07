"""مسیرهای منابع در دو حالت اجرا — سورس و بسته PyInstaller.

در سورس، منابع نسبت به ریشه پروژه‌اند (پدر پوشه app).
در exe (PyInstaller onedir)، منابع داخل پوشه نصب کنار exe هستند:
  DikteYar.exe
  _internal/            ← sys._MEIPASS (assets باندل‌شده)
  model/                ← کنار exe (دانلود در اولین اجرا یا دستی)

مسیر مدل frozen در config.model_dir از قبل همین‌طور است.
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def bundle_root() -> Path:
    """ریشه منابع باندل‌شده (assets) — در frozen داخل _internal، در سورس ریشه پروژه."""
    if is_frozen():
        return Path(sys._MEIPASS)  # noqa: SLF001 — قرارداد PyInstaller
    return Path(__file__).resolve().parent.parent


def install_root() -> Path:
    """پوشه نصب (کنار exe) در frozen؛ در سورس همان ریشه پروژه."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def asset_path(name: str) -> Path | None:
    """مسیر یک فایل در assets/ — اگر وجود نداشت None."""
    p = bundle_root() / "assets" / name
    return p if p.exists() else None
