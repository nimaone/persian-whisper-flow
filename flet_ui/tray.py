"""آیکون سینی برای اپ فلت — قرینه‌ی _start_tray/_tray_set_state در app/main.py.

pystray روی ترد خودش لوپ پیام ویندوز را می‌چرخاند؛ پس کالکب‌های منو از
آن ترد صدا زده می‌شوند و هر چیزی که به UI فلت دست می‌زند باید در run.py
به ایونت‌لوپ برگردد (page.run_task/run_thread). خودِ toggle_recording و
open_settings در بک‌اند ترد-سِیف‌اند.

اگر pystray نصب/قابل‌اجرا نباشد، اپ بدون سینی هم باید کار کند —
هیچ خطایی از این ماژول بیرون نمی‌زند.
"""
from __future__ import annotations

import threading

from app.config import APP_TITLE
from app import paths

# عنوان‌های tooltip به‌ازای هر حالت — عیناً مثل app/main.py:_ui_set_state
TITLES = {
    "loading": f"{APP_TITLE} (در حال بارگذاری)",
    "idle": f"{APP_TITLE} — آماده",
    "recording": f"{APP_TITLE} — در حال ضبط",
    "transcribing": f"{APP_TITLE} — در حال تشخیص",
    "starting": APP_TITLE,
}


def tray_image():
    """لوگوی برنامه — در نبود asset، همان شکل ساده‌ی fallback نسخه CTk."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    logo = paths.asset_path("logo.png")
    if logo:
        try:
            return Image.open(logo)
        except Exception:
            pass
    img = Image.new("RGB", (64, 64), (30, 30, 40))
    d = ImageDraw.Draw(img)
    d.ellipse((18, 14, 46, 42), fill=(34, 197, 94))
    d.rectangle((29, 38, 35, 54), fill=(34, 197, 94))
    return img


class Tray:
    """آیکون سینی با منوی «نمایش پنجره/ضبط/تنظیمات/خروج»."""

    def __init__(self, on_show, on_toggle, on_settings, on_quit,
                 pystray_mod=None, image=None):
        self._on_show = on_show
        self._on_toggle = on_toggle
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._pystray = pystray_mod
        self._image = image
        self._icon = None
        self._thread = None
        self.started = False

    # ---------- ساخت و اجرا ----------
    def _safe(self, fn):
        """کالکب منو نباید استثنا بیرون بدهد — pystray ترد را می‌خواباند."""
        def wrapper(*_a):
            try:
                fn()
            except Exception:
                pass
        return wrapper

    def build(self):
        """آیکون و منو را می‌سازد (بدون اجرا) — جدا شده برای تست."""
        pystray = self._pystray
        if pystray is None:
            try:
                import pystray as pystray_mod
                pystray = pystray_mod
            except Exception:
                return None
        image = self._image if self._image is not None else tray_image()
        if image is None:
            return None
        menu = pystray.Menu(
            pystray.MenuItem("نمایش پنجره", self._safe(self._on_show), default=True),
            pystray.MenuItem("شروع/توقف ضبط", self._safe(self._on_toggle)),
            pystray.MenuItem("تنظیمات", self._safe(self._on_settings)),
            pystray.MenuItem("خروج", self._safe(self._on_quit)),
        )
        self._icon = pystray.Icon("WhisperFlowFarsi", image,
                                  TITLES["loading"], menu)
        return self._icon

    def start(self) -> bool:
        """اجرای سینی روی ترد daemon — False یعنی سینی در دسترس نبود."""
        if self.started:
            return True
        try:
            if self.build() is None:
                return False
            self._thread = threading.Thread(target=self._icon.run, daemon=True)
            self._thread.start()
            self.started = True
            return True
        except Exception:
            self._icon = None
            return False

    # ---------- رابط عمومی (قرینه‌ی _tray_set_state/_notify) ----------
    def set_title(self, text: str):
        if self._icon is None:
            return
        try:
            self._icon.title = str(text)[:63]  # سقف ویندوز: ۶۴ کاراکتر
        except Exception:
            pass

    def set_state(self, state: str):
        self.set_title(TITLES.get(state, APP_TITLE))

    def notify(self, text: str):
        if self._icon is None:
            return
        try:
            self._icon.notify(text, APP_TITLE)
        except Exception:
            pass

    def stop(self):
        if self._icon is None:
            return
        try:
            self._icon.stop()
        except Exception:
            pass
        self.started = False
