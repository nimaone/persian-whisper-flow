"""اجراکننده‌ی UI آزمایشی Flet — هر دو پنجره با تم دیکته‌یار.

usage: python -m flet_ui.run [control|settings]
پیش‌فرض: control.  این ماژول فقط نمایش است؛ منطق (ضبط/ASR/هات‌کی)
به نسخه‌ی CTk در app/ حلق نمی‌شود.
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
            # نمایش حالت‌های مختلف برای اسکرین‌شات
            win.var_overlay.value = True
            win.var_sound.value = True
            sel = int(os.environ.get("TAB", "0"))
            if sel:
                win.tabs.selected_index = sel
                win.page.update()
            page.update()
        else:
            from flet_ui.control_window import ControlWindow
            win = ControlWindow(page)
            # چرخه‌ی حالت‌ها با کلیک روی دکمه‌ی اصلی — برای تست GUI
            states = [("idle", "ctrl+shift+space"), ("recording", "ctrl+shift+space"),
                      ("transcribing", "ctrl+shift+space")]

            def on_toggle():
                cur = next((i for i, s in enumerate(states) if s[0] == win._state), -1)
                nxt = states[(cur + 1) % len(states)]
                win.set_state(*nxt)

            win.on_toggle = on_toggle
            win.set_state("idle", "ctrl+shift+space")

        async def wait_close():
            # visible پس از destroy شدن پنجره False می‌شود
            while getattr(page.window, "visible", True):
                await asyncio.sleep(0.5)
            sys.exit(0)

        asyncio.create_task(wait_close())

    await ft.run_async(build)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "control"
    asyncio.run(main(target))
