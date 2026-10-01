"""پنجره کنترل دیکته‌یار با Flet — قرینه‌ی app/control_window.py.

وضعیت (بارگذاری/آماده/ضبط/تشخیص) + دکمه اصلی + تنظیمات + نسخه.
همان تم تیره، همان چیدمان، همان متن‌ها.
"""
from __future__ import annotations

import flet as ft

from app.config import APP_TITLE, APP_TITLE_FULL, APP_VERSION
from flet_ui import theme as t


class ControlWindow:
    """همان رابط عمومی ControlWindow در control_window.py — هم‌نامِ هم‌کارکرد."""

    def __init__(self, page: ft.Page):
        self.page = page
        t.install_fonts(page)
        t.apply_icon(page)
        page.title = APP_TITLE
        page.bgcolor = t.BG
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 360
        page.window.height = 250
        page.padding = 20
        page.rtl = True

        # ---------- هدر: نام اپ (راست) + وضعیت (چپ) ----------
        self.state_text = ft.Text("در حال بارگذاری", style=t.fam("Regular", 10), color=t.FG_DIM)
        head = ft.Row([ft.Text(APP_TITLE_FULL, style=t.fam("bold", 14), color=t.FG),
                       self.state_text],
                      alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        self.status_text = ft.Text(
            "مدل در حال بارگذاری است…",
            style=t.fam("Regular", 10), color=t.FG_DIM,
            text_align=ft.TextAlign.RIGHT,
        )

        # ---------- دکمه اصلی ضبط (تمام‌عرض مثل fill="x" در CTk) ----------
        self.rec_btn = ft.Button(
            content="شروع ضبط", disabled=True, on_click=self._fire_toggle, height=44,
            bgcolor=t.SURFACE_2, color=t.FG_DIM,
            style=t.btn_style(radius=10, size=12, weight="bold", hpad=0),
        )
        st_btn = ft.Button(
            content="تنظیمات", on_click=self._fire_settings, height=36,
            bgcolor=t.SURFACE_2, color=t.FG,
            style=t.btn_style(radius=8, size=12, weight="bold", hpad=0),
        )

        self.hint_text = ft.Text("", style=t.fam("Regular", 9), color=t.FG_DIM,
                                 text_align=ft.TextAlign.RIGHT)

        # نسخه برنامه — ریز، پایین-چپ
        ver = ft.Text(f"v{APP_VERSION}", style=t.fam("Regular", 8), color=t.FG_DIM)

        self.rec_btn.expand = True   # داخل Row → تمام‌عرض مثل fill="x"
        st_btn.expand = True

        page.add(
            head,
            self.status_text,
            ft.Row([self.rec_btn]),
            ft.Row([st_btn]),
            self.hint_text,
            ver,
        )

        self.on_toggle = None
        self.on_settings = None
        self._state = "loading"

    # ---------- رویدادها ----------
    def _fire_toggle(self, e=None):
        if self.on_toggle:
            self.on_toggle()

    def _fire_settings(self, e=None):
        if self.on_settings:
            self.on_settings()

    # ---------- رابط عمومی (قرینه‌ی set_state) ----------
    def set_state(self, state: str, hotkey: str):
        """state: loading | idle | recording | transcribing"""
        self._state = state
        if state == "loading":
            self.state_text.value, self.state_text.color = "در حال بارگذاری", t.FG_DIM
            self.rec_btn.content= "شروع ضبط"
            self._set_btn_enabled(False)
            self.status_text.value = "مدل در حال بارگذاری است…"
            self.hint_text.value = ""
        elif state == "idle":
            self.state_text.value, self.state_text.color = "آماده", t.ACCENT
            self.rec_btn.content= "شروع ضبط"
            self._set_btn_enabled(True, t.ACCENT, t.ACCENT_HOVER, t.ON_ACCENT)
            self.status_text.value = f"کلید میانبر: {hotkey}"
            self.hint_text.value = "در هر برنامه‌ای کلید را بزن و صحبت کن"
        elif state == "recording":
            self.state_text.value, self.state_text.color = "در حال ضبط", t.DANGER
            self.rec_btn.content= "توقف و درج متن"
            self._set_btn_enabled(True, t.DANGER, t.DANGER_HOVER, t.ON_DANGER)
            self.status_text.value = "در حال شنیدن…"
            self.hint_text.value = "برای پایان، دوباره کلید میانبر را بزن"
        elif state == "transcribing":
            self.state_text.value, self.state_text.color = "تشخیص", t.FG_DIM
            self.rec_btn.content= "در حال تشخیص"
            self._set_btn_enabled(False)
            self.status_text.value = "متن را می‌نویسد…"
            self.hint_text.value = ""
        self._schedule_update()

    def _set_btn_enabled(self, enabled: bool, fg=None, hover=None, txt=None):
        self.rec_btn.disabled = not enabled
        self.rec_btn.bgcolor = fg if enabled else t.SURFACE_2
        self.rec_btn.color = txt if enabled else t.FG_DIM
        # Flet hover = overlay_color روشن‌تر روی bgcolor
        self.rec_btn.style.overlay_color = hover if enabled else t.SURFACE_2

    def set_status(self, msg: str):
        self.status_text.value = msg
        self._schedule_update()

    def set_error(self, msg: str):
        self.status_text.value = f"خطا: {msg[:60]}"
        self.status_text.color = t.DANGER
        self._schedule_update()

    # ---------- ارسال آپدیت از ایونت‌لوپ ----------
    def _schedule_update(self):
        """ارسال پچ‌ها باید از ترد ایونت‌لوپ فلت انجام شود — داکیومنت رسمی:
        page.run_task کوروتین را به‌عنوان Task روی ایونت‌لوپ اجرا می‌کند.
        put_nowait روی صف asyncio از ترد فرعی (هات‌کی/worker ضبط) تسکِ
        ارسال را بیدار نمی‌کند و آپدیت هرگز رندر نمی‌شود. تغییر خودِ
        props از هر تردی امن است؛ فقط ارسال باید روی لوپ برود.
        """
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            run_task(self._flush_update)
        else:
            try:
                self.page.update()   # MockPage (تست‌ها) — بدون ایونت‌لوپ
            except Exception:
                pass  # پنجره بسته شده

    async def _flush_update(self):
        try:
            self.page.update()
        except Exception:
            pass  # پنجره بسته شده
