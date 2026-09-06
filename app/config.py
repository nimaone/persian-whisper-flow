"""تنظیمات برنامه — ذخیره در %APPDATA%\\WhisperFlowFarsi\\settings.json"""
from __future__ import annotations

import json
import os
import sys
import winreg
from pathlib import Path

APP_NAME = "WhisperFlowFarsi"  # فقط مسیر %APPDATA% و کلید رجیستری autostart — تغییرش مهاجرت تنظیمات می‌خواهد
APP_TITLE = "دیکته‌یار"  # نام نمایشی کوتاه — نوار عنوان، تسک‌بار و tray
APP_TITLE_FULL = "دیکته‌یار (ویسپر فلوی فارسی)"  # فرم کامل — هدر صفحه اصلی و راهنما
APP_VERSION = "1.1.0"

DEFAULTS = {
    "hotkey": "ctrl+shift+space",
    "paste_method": "clipboard",  # clipboard | type
    "voice_commands": True,
    "persian_itn": True,         # تبدیل اعداد حروفی به رقم («بیست و سه» → ۲۳)
    "input_device": None,  # None = تشخیص خودکار
    "autostart": False,
    "overlay_enabled": True,
    "num_threads": 4,
    "sound_feedback": True,      # بوق کوتاه شروع/پایان ضبط
    "overlay_font_size": 15,     # اندازه فونت متن پنجره زنده
    "restore_clipboard": True,   # بازیابی محتوای قبلی کلیپ‌بورد بعد از درج
    "auto_stop_sec": 0,          # توقف خودکار پس از N ثانیه سکوت (0 = خاموش)
}


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    d = root / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def model_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "model"
    return Path(__file__).resolve().parent.parent / "model"


class Config:
    def __init__(self, data: dict):
        self.data = {**DEFAULTS, **data}

    @classmethod
    def load(cls) -> "Config":
        p = settings_path()
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return cls(json.load(f))
            except Exception:
                pass
        return cls(dict(DEFAULTS))

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def save(self):
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


def _autostart_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    py = pythonw if pythonw.exists() else Path(sys.executable)
    main_py = Path(__file__).resolve().parent / "main.py"
    return f'"{py}" "{main_py}"'


def set_autostart(enable: bool):
    """کلید Run رجیستری برای اجرای خودکار با ورود به ویندوز."""
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            if enable:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, _autostart_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
    except OSError:
        pass
