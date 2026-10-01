"""لانچر نسخه‌ی فلت دیکته‌یار — برای «اجرای خودکار با ورود به ویندوز».

رجیستری (کلید Run) به یک فایل اجرایی نیاز دارد؛ پس این فایل کوچک در
ریشه می‌ماند و مسیرش با pythonw ثبت می‌شود — بدون پنجره‌ی کنسول.
(قرینه‌ی «pythonw app/main.py» در _autostart_command نسخه CTk)
"""
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # مثل اجرای «python -m flet_ui.run» از ریشه‌ی پروژه

from flet_ui.run import main  # noqa: E402

asyncio.run(main("app"))
