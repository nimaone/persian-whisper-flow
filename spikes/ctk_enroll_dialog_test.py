"""تست دیالوگ ثبت واژه — فریز/نامرئی‌بودن دیالوگ را بازتولید و تأیید می‌کند.

بدون تعامل: پنجره تنظیمات باز می‌شود، دکمه «+ ثبت واژه جدید» invoke
می‌شود و چک می‌کنیم دیالوگ mapped/viewable شود و grab داشته باشد.
خروجی سالم: «DONE — no freeze» در آخر.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk

import customtkinter as ctk

from app import settings_ui


def find_button(widget, needle: str):
    for c in widget.winfo_children():
        try:
            txt = c.cget("text")
        except Exception:
            txt = None
        if isinstance(c, ctk.CTkButton) and txt and needle in txt:
            return c
        r = find_button(c, needle)
        if r is not None:
            return r
    return None


def main():
    root = ctk.CTk()
    root.withdraw()

    def run():
        settings_ui.open_settings(root, app=None)
        root.update()
        win = [w for w in root.winfo_children()
               if isinstance(w, tk.Toplevel)][0]
        btn = find_button(win, "ثبت واژه جدید")
        print("دکمه پیدا شد:", btn is not None)
        btn.invoke()
        for _ in range(30):
            root.update()
            time.sleep(0.03)
        toplevels = [w for w in win.winfo_children()
                     if isinstance(w, tk.Toplevel)]
        print("تعداد دیالوگ:", len(toplevels))
        if toplevels:
            d = toplevels[0]
            print("mapped:", d.winfo_ismapped(),
                  "| viewable:", d.winfo_viewable(),
                  "| grab_current == dlg:", d.grab_current() is d)
            # تنظیمات هنوز به کلیک جواب می‌دهد؟ (grab اشتباه نباشد)
            print("تنظیمات mapped:", win.winfo_ismapped())
            d.destroy()
        win.destroy()
        root.destroy()

    root.after(150, run)
    root.mainloop()
    print("DONE — no freeze")


if __name__ == "__main__":
    main()
