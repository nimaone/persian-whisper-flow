"""اجراکننده‌ی UI فلت دیکته‌یار — با منطق کامل ضبط/ترنسکرایب.

usage:
    python -m flet_ui.run            ← برنامه کامل (پیش‌فرض)
    python -m flet_ui.run demo       ← فقط نمایش UI (بدون مدل/میکروفون)
    python -m flet_ui.run settings   ← فقط پنجره تنظیمات (از بک‌اند صدا زده می‌شود)

چرخه‌ی عمر مثل نسخه‌ی CTk است:
    بدون مدل → صفحه‌ی اولین اجرا (دانلود/انتخاب دستی) → اپ اصلی
    مینیمایز → رفتن به سینی    |    بستن پنجره (X) → خروج کامل
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import flet as ft

from flet_ui import theme as t


def window_event_kind(e) -> str:
    """نوع رویداد پنجره به‌صورت رشته — flet بسته به نسخه رشته یا
    WindowEventType می‌دهد؛ هر دو را یکسان می‌کنیم."""
    kind = getattr(e, "type", "")
    return str(getattr(kind, "value", kind)).lower()


async def main(target: str):
    # پنجره اول hidden بالا می‌آید تا اندازه‌ی درست پیش از اولین فریم ست شود —
    # وگرنه کلاینت فلت با اندازه‌ی پیش‌فرض بزرگش دیده می‌شود و بعد کوچک می‌شود
    hidden = True
    holder: dict = {}

    async def build(page: ft.Page):
        t.install_fonts(page)
        page.bgcolor = t.BG
        page.theme_mode = ft.ThemeMode.DARK

        # ---------- خروج تمیز (دکمه‌ی بستن پنجره / منوی سینی) ----------
        async def _graceful_close():
            # بستن پنجره، کلاینت فلت (flet.exe) را هم می‌بندد — با os._exit
            # تنها، کلاینت سرگردان می‌ماند (باگ واقعی: پروسه‌ی ۱۲۰MB باقی می‌ماند)
            try:
                await page.window.destroy()
            except Exception:
                pass

        def _quit_all(e=None):
            app = holder.get("app")
            if app is not None:
                try:
                    app.quit()
                except Exception:
                    pass
            tray = holder.get("tray")
            if tray is not None:
                try:
                    tray.stop()
                except Exception:
                    pass
            # جان‌پناه: اگر destroy پنجره/سشن را نبست، حداکثر ۳ ثانیه بعد می‌رویم
            threading.Thread(target=lambda: (time.sleep(3), os._exit(0)),
                             daemon=True).start()
            page.run_task(_graceful_close)

        # ---------- پنجره: مخفی در سینی و برگشت ----------
        async def _show_window():
            # بازگشت از سینی: هم visible و هم minimized باید برگردند
            try:
                page.window.visible = True
                page.window.minimized = False
                page.update()
                await page.window.to_front()
            except Exception:
                pass  # پنجره بسته شده

        async def _hide_to_tray():
            try:
                page.window.visible = False
                page.update()
            except Exception:
                pass

        def _setup_tray(app):
            """آیکون سینی — اگر pystray در دسترس نباشد اپ بدون سینی ادامه می‌دهد."""
            from flet_ui.tray import Tray
            tray = Tray(
                on_show=lambda: page.run_task(_show_window),
                on_toggle=app.toggle_recording,
                on_settings=app.open_settings,
                on_quit=_quit_all,
            )
            if tray.start():
                holder["tray"] = tray
                app.tray = tray  # backend عنوان/بالون‌ها را از همین می‌فرستد

        def _start_dictation():
            from flet_ui.backend import DictationApp
            from flet_ui.control_window import ControlWindow
            try:
                page.controls.clear()  # صفحه‌ی اولین اجرا باید کنار برود
            except Exception:
                pass
            win = ControlWindow(page)
            app = DictationApp(win)
            win.on_toggle = app.toggle_recording
            win.on_settings = app.open_settings
            holder["app"], holder["win"] = app, win
            _setup_tray(app)  # پیش از start تا tooltip «در حال بارگذاری» بیاید
            app.start()
            page.update()

        if target == "settings":
            from flet_ui.settings_window import SettingsWindow
            win = SettingsWindow(page)
            sel = int(os.environ.get("TAB", "0"))
            if sel:
                win.tabs.selected_index = sel
                win.page.update()
            page.update()

        elif target == "demo":
            # نمایش حالت‌ها با کلیک روی دکمه — برای تست GUI بدون مدل
            from flet_ui.control_window import ControlWindow
            win = ControlWindow(page)
            states = [("idle", "ctrl+shift+space"), ("recording", "ctrl+shift+space"),
                      ("transcribing", "ctrl+shift+space")]

            def on_toggle():
                cur = next((i for i, s in enumerate(states) if s[0] == win._state), -1)
                nxt = states[(cur + 1) % len(states)]
                win.set_state(*nxt)

            win.on_toggle = on_toggle
            win.set_state("idle", "ctrl+shift+space")
            page.update()

        else:  # برنامه کامل
            from flet_ui.first_run_window import FirstRunWindow, model_ready
            if model_ready():
                _start_dictation()
            else:
                # بدون مدل، همان صفحه‌ی اولین اجرا جای پنجره‌ی کنترل می‌نشیند
                FirstRunWindow(page, on_ready=_start_dictation, on_quit=_quit_all)

        # همه‌چیز چیدمان شد — حالا پنجره را نشان بده
        if hidden:
            try:
                await asyncio.wait_for(page.window.wait_until_ready_to_show(), 5)
            except Exception:
                pass  # آماده‌سازی طول کشید — با همین نشان بده
            page.window.visible = True
            page.update()
            await asyncio.sleep(0.5)
            # اولین set:visible گاهی نزد کلاینت hide می‌شود — دوباره بفرست
            page.window.visible = True
            page.update()

        # بستن پنجره با دکمه‌ی نوار عنوان → خروج تمیز؛ مینیمایز → سینی
        # (visible قابل اعتماد نیست چون با شروع hidden False می‌ماند)
        def _on_win_event(e):
            kind = window_event_kind(e)
            if kind == "close":
                _quit_all()
            elif kind == "minimize" and holder.get("tray") is not None:
                # بدون سینیِ فعال پنجره را قایم نمی‌کنیم — راه برگشتی نمی‌ماند
                page.run_task(_hide_to_tray)

        page.window.on_event = _on_win_event

        async def wait_close():
            # فقط نگه‌داشتن برنامه؛ خروج با رویداد close پنجره انجام می‌شود
            while True:
                await asyncio.sleep(3600)

        asyncio.create_task(wait_close())

    await ft.run_async(build, view=ft.AppView.FLET_APP_HIDDEN if hidden
                       else ft.AppView.FLET_APP)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "app"
    asyncio.run(main(target))
