"""پنجره کنترل — وضعیت + دکمه ضبط + میان‌بر تنظیمات (Fluent تیره).

این پنجره در تسک‌بار دیده می‌شود (برخلاف overlay) پس مینیمایز/سوییچ ممکن است.
"""
from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from app import theme
from app.win32 import style_toplevel, smooth_show
from app.theme import apply_icon


class ControlWindow:
    def __init__(self, root: tk.Tk):
        self.fam = theme.family()
        self.win = tk.Toplevel(root)
        self.win.title("وِیس‌فلو فارسی")
        self.win.geometry("360x250")
        self.win.minsize(320, 230)
        self.win.configure(bg=theme.BG)
        style_toplevel(self.win)  # عنوان تیره + گوشه‌ی گرد
        apply_icon(self.win)      # لوگو در نوار عنوان/تسک‌بار

        c = ctk.CTkFrame(self.win, fg_color="transparent", corner_radius=0)
        c.pack(fill="both", expand=True, padx=20, pady=16)

        # هدر: نام اپ (راست) + وضعیت (چپ)
        head = ctk.CTkFrame(c, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="وِیس‌فلو فارسی", font=(self.fam, 14, "bold"),
                     text_color=theme.FG).pack(side="right")
        self.state_var = tk.StringVar(value="در حال بارگذاری")
        self.state_lbl = ctk.CTkLabel(head, textvariable=self.state_var,
                                      font=(self.fam, 10), text_color=theme.ACCENT)
        self.state_lbl.pack(side="left")

        self.status_var = tk.StringVar(value="مدل در حال بارگذاری است…")
        self.status_lbl = ctk.CTkLabel(c, textvariable=self.status_var,
                                       font=(self.fam, 10), text_color=theme.FG_DIM,
                                       anchor="e", justify="right")
        self.status_lbl.pack(fill="x", pady=(8, 0))

        # دکمه اصلی ضبط
        self.rec_btn_var = tk.StringVar(value="شروع ضبط")
        self.on_toggle = None
        self.rec_btn = ctk.CTkButton(
            c, textvariable=self.rec_btn_var, font=(self.fam, 12, "bold"),
            height=44, corner_radius=10,
            fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
            text_color=theme.FG_DIM, command=self._fire_toggle, state="disabled",
        )
        self.rec_btn.pack(fill="x", pady=(16, 8))

        st_btn = ctk.CTkButton(
            c, text="تنظیمات", font=(self.fam, 12, "bold"), height=36, corner_radius=8,
            fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
            text_color=theme.FG, command=self._fire_settings,
        )
        st_btn.pack(fill="x")

        self.hint_var = tk.StringVar(value="")
        ctk.CTkLabel(c, textvariable=self.hint_var, font=(self.fam, 9),
                     text_color=theme.FG_DIM, anchor="e").pack(fill="x", pady=(10, 0))

        self._state = "loading"
        smooth_show(self.win)  # نمایش نرم — بدون فریم سفید

        # مینیمایز → سینی: چون Toplevel دکمه‌ی تسک‌بار ندارد، پنجره‌ی مینیمایز
        # شده off-screen گم می‌شود؛ به‌جایش withdraw می‌کنیم تا اپ فقط در
        # سینی بماند (بازگشت با دابل‌کلیک آیکون سینی یا منوی «نمایش پنجره»)
        self.win.bind("<Unmap>", self._on_unmap)

    def _on_unmap(self, event):
        if event.widget is not self.win:
            return  # رویدادهای بچه‌ها هم به bindtags این پنجره می‌رسند
        if self.win.state() == "iconic":
            self.win.after(10, self._minimize_to_tray)

    def _minimize_to_tray(self):
        if self.win.state() == "iconic":
            self.win.withdraw()

    def show_window(self):
        """بازگردانی پنجره از سینی — فقط از thread اصلی."""
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def _fire_toggle(self):
        if self.on_toggle:
            self.on_toggle()

    def _fire_settings(self):
        if self.on_settings:
            self.on_settings()

    # ---------- رابط (فقط از thread اصلی) ----------
    def set_state(self, state: str, hotkey: str):
        """state: loading | idle | recording | transcribing"""
        self._state = state
        if state == "loading":
            self.state_var.set("در حال بارگذاری")
            self.state_lbl.configure(text_color=theme.FG_DIM)
            self.rec_btn_var.set("شروع ضبط")
            self._set_btn_enabled(False)
            self.status_var.set("مدل در حال بارگذاری است…")
            self.hint_var.set("")
        elif state == "idle":
            self.state_var.set("آماده")
            self.state_lbl.configure(text_color=theme.ACCENT)
            self.rec_btn_var.set("شروع ضبط")
            self._set_btn_enabled(True, theme.ACCENT, theme.ACCENT_HOVER, theme.ON_ACCENT)
            self.status_var.set(f"کلید میانبر: {hotkey}")
            self.hint_var.set("در هر برنامه‌ای کلید را بزن و صحبت کن")
        elif state == "recording":
            self.state_var.set("در حال ضبط")
            self.state_lbl.configure(text_color=theme.DANGER)
            self.rec_btn_var.set("توقف و درج متن")
            self._set_btn_enabled(True, theme.DANGER, theme.DANGER_HOVER, theme.ON_DANGER)
            self.status_var.set("در حال شنیدن…")
            self.hint_var.set("برای پایان، دوباره کلید میانبر را بزن")
        elif state == "transcribing":
            self.state_var.set("تشخیص")
            self.state_lbl.configure(text_color=theme.FG_DIM)
            self.rec_btn_var.set("در حال تشخیص")
            self._set_btn_enabled(False)
            self.status_var.set("متن را می‌نویسد…")
            self.hint_var.set("")

    def _set_btn_enabled(self, enabled: bool, fg=None, hover=None, txt=None):
        if enabled:
            self.rec_btn.configure(state="normal", fg_color=fg,
                                   hover_color=hover, text_color=txt)
        else:
            self.rec_btn.configure(state="disabled", fg_color=theme.SURFACE_2,
                                   hover_color=theme.SURFACE_2, text_color=theme.FG_DIM)

    def set_error(self, msg: str):
        self.status_var.set(f"خطا: {msg[:60]}")
        self.status_lbl.configure(text_color=theme.DANGER)
