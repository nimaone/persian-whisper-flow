"""اجراکننده‌ی UI فلت دیکته‌یار — با منطق کامل ضبط/ترنسکرایب.

usage:
    python -m flet_ui.run            ← برنامه کامل (پیش‌فرض)
    python -m flet_ui.run demo       ← فقط نمایش UI (بدون مدل/میکروفون)
    python -m flet_ui.run settings   ← فقط پنجره تنظیمات (از بک‌اند صدا زده می‌شود)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import flet as ft

from flet_ui import theme as t


async def main(target: str):
    # پنجره اول hidden بالا می‌آید تا اندازه‌ی درست پیش از اولین فریم ست شود —
    # وگرنه کلاینت فلت با اندازه‌ی پیش‌فرض بزرگش دیده می‌شود و بعد کوچک می‌شود
    hidden = True
    async def build(page: ft.Page):
        t.install_fonts(page)
        page.bgcolor = t.BG
        page.theme_mode = ft.ThemeMode.DARK

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
            from flet_ui.backend import DictationApp
            from flet_ui.control_window import ControlWindow
            win = ControlWindow(page)
            app = DictationApp(win)
            win.on_toggle = app.toggle_recording
            win.on_settings = app.open_settings
            app.start()
            page.update()

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

        # بستن پنجره با دکمه‌ی نوار عنوان → خروج تمیز (visible قابل
        # اعتماد نیست چون با شروع hidden False می‌ماند)
        def _on_win_event(e):
            if getattr(e, "type", "") == "close":
                if target != "settings":
                    try:
                        app.quit()
                    except Exception:
                        pass
                page.run_thread(_exit)

        def _exit():
            time.sleep(0.2)
            os._exit(0)

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
