"""تولید اسکرین‌شات‌های README — بدون نیاز به موتور ASR (استاب).

اجرا:
    python spikes/make_screenshots.py control    # پنجره اصلی (آماده)
    python spikes/make_screenshots.py overlay    # پنجره زنده با متن نمونه
    python spikes/make_screenshots.py settings   # پنج تب تنظیمات

خروجی‌ها در docs/screenshots/ ذخیره می‌شوند (برای embed در README).
نکته: در حین اجرا نباید پنجره دیگری روی پنجره‌های اپ باشد.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

# استاب سخت‌افزار/وین۳۲ — پنجره‌ها بدون tray و DWM هم رندر می‌شوند
sys.modules["app.recorder"] = types.SimpleNamespace(detect_best_device=lambda: None)
sys.modules["app.win32"] = types.SimpleNamespace(
    style_toplevel=lambda w: None,
    smooth_show=lambda w: w.deiconify(),
    disable_min_max=lambda w: None,
)

import tkinter as tk  # noqa: E402

import customtkinter as ctk  # noqa: E402
from PIL import ImageGrab  # noqa: E402


def grab(win: tk.Wm, path: Path) -> None:
    """اسکرین‌شات دقیق bbox پنجره + چک سالم‌بودن (خالی/سیاه نباشد)."""
    win.update_idletasks()
    win.update()
    time.sleep(0.25)  # فرصت رندر نهایی
    win.update()
    x, y = win.winfo_rootx(), win.winfo_rooty()
    w, h = win.winfo_width(), win.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    g = img.convert("L")
    lo, hi = g.getextrema()
    if hi - lo < 30:
        raise SystemExit(f"تصویر تقریباً یکدست است (احتمالاً پنجره روی صفحه نیست): {path}")
    img.save(path)
    print(f"✓ {path.name}  {w}x{h}")


def shoot_control() -> None:
    from app.control_window import ControlWindow

    root = tk.Tk()
    root.withdraw()
    cw = ControlWindow(root)
    cw.win.attributes("-topmost", True)
    cw.set_state("idle", "Ctrl+Shift+Space")
    cw.win.after(600, lambda: grab(cw.win, OUT / "main.png") or root.destroy())
    root.mainloop()


def shoot_overlay() -> None:
    from app.overlay import Overlay

    ov = Overlay()
    ov.update_text("سلام، این پیام با دیکته‌یار نوشته شده است؛ کاملاً آفلاین و بدون اینترنت.")
    ov.root.attributes("-topmost", True)
    ov.show()
    ov.root.geometry("+200+200")
    for lvl in (0.6, 0.35, 0.5):
        ov.update_level(lvl)
        ov.tick()
    ov.root.after(700, lambda: grab(ov.root, OUT / "overlay.png") or ov.root.destroy())
    ov.root.mainloop()


def shoot_settings() -> None:
    from app import settings_ui

    root = tk.Tk()
    root.withdraw()
    settings_ui.open_settings(root)
    win = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    win.attributes("-topmost", True)

    tvs: list[ctk.CTkTabview] = []

    def walk(w):
        for ch in w.winfo_children():
            if isinstance(ch, ctk.CTkTabview):
                tvs.append(ch)
            walk(ch)

    walk(win)
    tv = tvs[0]
    tabs = {"عمومی": "settings_general", "میکروفون": "settings_mic",
            "درج متن": "settings_insert", "پیشرفته": "settings_advanced",
            "راهنما": "settings_help"}

    def step(names: list[str]):
        if not names:
            root.destroy()
            return
        fa, en = names[0]
        tv.set(fa)
        win.update_idletasks()
        win.after(500, lambda: _shot_then(step, win, en, names[1:]))

    def _shot_then(next_fn, win, en, rest):
        try:
            grab(win, OUT / f"{en}.png")
        except SystemExit as e:
            print(e)
        next_fn(rest)

    win.after(400, lambda: step(list(tabs.items())))
    root.mainloop()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"control": shoot_control, "overlay": shoot_overlay, "settings": shoot_settings}.get(
        what, lambda: print("usage: make_screenshots.py control|overlay|settings"))()
