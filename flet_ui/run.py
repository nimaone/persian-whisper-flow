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

import flet as ft

from flet_ui import theme as t


async def main(target: str):
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

        async def wait_close():
            # visible پس از destroy شدن پنجره False می‌شود
            while getattr(page.window, "visible", True):
                await asyncio.sleep(0.5)
            if target != "settings":
                try:
                    app.quit()
                except Exception:
                    pass
            sys.exit(0)

        asyncio.create_task(wait_close())

    await ft.run_async(build)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "app"
    asyncio.run(main(target))
