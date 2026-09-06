"""تماس‌های بومی ویندوز برای ظاهر پنجره‌ها — عنوان تیره، گوشه گرد، نمایش بدون فلش."""
from __future__ import annotations

import ctypes
import sys
import tkinter as tk


def _apply_dwm(hwnd: int) -> None:
    """ست اتریبیوت‌های تیره/گرد روی یک HWND مشخص."""
    set_attr = ctypes.windll.dwmapi.DwmSetWindowAttribute
    dark = ctypes.c_int(1)  # DWMWA_USE_IMMERSIVE_DARK_MODE
    for attr in (20, 19):  # بیلدهای مختلف ویندوز
        if set_attr(hwnd, attr, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
            break
    pref = ctypes.c_int(2)  # DWMWCP_ROUND
    set_attr(hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref))


def style_toplevel(win) -> None:
    """عنوان پنجره را تیره و گوشه‌ها را گرد می‌کند (وین ۱۰/۱۱) — بی‌صدا رد می‌شود.

    پنجره را withdraw → realize می‌کند و اتریبیوت DWM را قبل از اولین
    فریم ست می‌کند؛ پنجره را مخفی نگه می‌دارد — نمایش با smooth_show.
    """
    if sys.platform != "win32":
        return
    try:
        if not win.winfo_ismapped():
            win.withdraw()
            win.update_idletasks()  # realize بدون مپ
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        if hwnd:
            _apply_dwm(hwnd)
    except Exception:
        pass


def disable_min_max(win) -> None:
    """غیرفعال‌کردن دکمه‌های مینیمایز/ماکسیمایز (برای دیالوگ‌های ثابت).

    دکمه مینیمایز در Toplevelهای Tk روی ویندوز درست کار نمی‌کند؛ به‌جای
    یک دکمه بی‌اثر، خاکستری‌اش می‌کنیم (بسته‌شدن با دکمه بستن سر جایش است).
    """
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        GWL_STYLE = -16
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        user32.SetWindowLongW(hwnd, GWL_STYLE,
                              style & ~(WS_MINIMIZEBOX | WS_MAXIMIZEBOX))
        # بازترسیم ناحیه عنوان تا دکمه‌ها بلافاصله خاکستری شوند
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x1 | 0x2 | 0x4 | 0x20)
    except Exception:
        pass


def smooth_show(win, duration_ms: int = 140) -> None:
    """نمایش نرم پنجره بدون فلش سفید.

    دلیل فلش: ویندوز قبل از اولین ترسیمِ Tk ناحیه client را با براش سفیدِ
    کلاس پنجره می‌کشد. اینجا کل پنجره (شامل نوار عنوان) تا اولین ترسیم
    کامل نامرئی (alpha=0) می‌ماند و بعد fade-in می‌شود — کاربر هیچ
    فریم سفیدی نمی‌بیند. اگر پنجره از قبل نمایان باشد کاری نمی‌کند.
    """
    if sys.platform != "win32":
        try:
            win.deiconify()
        except Exception:
            pass
        return
    try:
        if win.winfo_ismapped():
            return
        try:
            win.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        win.deiconify()
        try:
            win.update_idletasks()
            win.update()  # اولین ترسیم کامل در حالت نامرئی
            win.update()
        except tk.TclError:
            return
        steps = max(2, duration_ms // 16)
        interval = max(8, duration_ms // steps)

        def _step(i=0):
            try:
                win.attributes("-alpha", (i + 1) / steps)
            except tk.TclError:
                return
            if i + 1 < steps:
                win.after(interval, lambda: _step(i + 1))

        win.after(10, _step)
    except Exception:
        pass
