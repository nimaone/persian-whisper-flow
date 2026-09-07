"""پنجره overlay زنده — کارت گوشه‌گرد مینیمال نزدیک نشانگر موس.

فقط دو چیز را نشان می‌دهد: متن در حال تشخیص و یک موج ظریف متصل به سطح
واقعی میکروفون. هدر (نقطه‌ی ضبط + میان‌بر) فقط هنگام ضبط نمایان است و
در حالت «در حال تشخیص» کنار می‌رود تا متن نفَس بگیرد. ارتفاع کارت با
تعداد خطوط متن رشد می‌کند و پایین نمی‌زند.
"""
from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from app import paths, theme

PAD = 18
WAVE_BARS = 32
WAVE_BAR_W = 3
WAVE_GAP = 3
WAVE_H = 18
BODY_W = 440


class Overlay:
    def __init__(self):
        self.fam = theme.family()
        ctk.set_appearance_mode("dark")
        # رنگ ریشه = کلید شفافیت؛ تا مستطیلِ پس‌زمینه دور کارت دیده نشود
        self.root = ctk.CTk(fg_color="#010203")
        self.root.title("")
        self.root.overrideredirect(True)  # بدون قاب/عنوان
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", "#010203")
            self._transparent_ok = True
        except tk.TclError:
            self._transparent_ok = False
        self.root.configure(bg="#010203" if self._transparent_ok else theme.SURFACE)
        self.root.withdraw()  # شروع مخفی

        # لوگو به‌عنوان آیکون پیش‌فرض همه‌ی پنجره‌های Toplevel
        self._logo_ref = None
        logo = paths.asset_path("logo.png")
        if logo:
            try:
                from PIL import ImageTk
                self._logo_ref = ImageTk.PhotoImage(file=str(logo))
                self.root.iconphoto(True, self._logo_ref)
            except Exception:
                pass

        self.card = ctk.CTkFrame(
            self.root, fg_color=theme.SURFACE, bg_color="#010203",
            border_color=theme.BORDER, border_width=1, corner_radius=14,
        )
        self.card.pack(fill="both", expand=True)

        # هدر — فقط هنگام ضبط
        self.header = ctk.CTkFrame(self.card, fg_color="transparent")
        self.header.pack(fill="x", padx=PAD, pady=(10, 0))
        hrow = ctk.CTkFrame(self.header, fg_color="transparent")
        hrow.pack(fill="x")
        self.dot = tk.Canvas(hrow, width=8, height=8, bg=theme.SURFACE,
                             highlightthickness=0)
        self.dot.pack(side="right")
        ctk.CTkLabel(hrow, text="در حال شنیدن", font=(self.fam, 12, "bold"),
                     text_color=theme.ACCENT).pack(side="right", padx=(6, 0))
        self.hint = ctk.CTkLabel(hrow, text="همان کلید را دوباره بزن تا درج شود",
                                 font=(self.fam, 11), text_color=theme.FG_DIM)
        self.hint.pack(side="left")

        # متن زنده
        self.label = ctk.CTkLabel(
            self.card, text=" در حال شنیدن… ",
            font=(self.fam, 15), text_color=theme.FG,
            justify="right", anchor="e", wraplength=BODY_W,
        )
        self.label.pack(fill="x", padx=PAD, pady=(8, 10))

        # موج
        self.wave = tk.Canvas(self.card, width=BODY_W, height=WAVE_H + 4,
                              bg=theme.SURFACE, highlightthickness=0)
        self.wave.pack(fill="x", padx=PAD, pady=(0, 12))

        self._level = 0.0
        self._bars = [0.05] * WAVE_BARS
        self._phase = 0.0
        self._visible = False
        self._recording = False

    # ---------- مکان‌یابی ----------
    def _reflow(self):
        """اندازه‌ی کارت را با محتوای فعلی هم‌تراز می‌کند."""
        self.root.update_idletasks()
        w = self.card.winfo_reqwidth() + 2
        h = self.card.winfo_reqheight() + 2
        if (w, h) != (self.root.winfo_width(), self.root.winfo_height()):
            self.root.geometry(f"{w}x{h}")

    def place_near_mouse(self):
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
        except tk.TclError:
            x, y = 200, 200
        self._reflow()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        pos_x = min(max(8, x - int(w * 0.3)), self.root.winfo_screenwidth() - w - 8)
        pos_y = min(max(8, y + 26), self.root.winfo_screenheight() - h - 8)
        self.root.geometry(f"+{pos_x}+{pos_y}")

    # ---------- رابط عمومی ----------
    def show(self):
        if not self._visible:
            self._bars = [0.05] * WAVE_BARS
            self._recording = True
            self.dot.pack(side="right")
            self.hint.configure(text="همان کلید را دوباره بزن تا درج شود")
            self.header.pack(fill="x", padx=PAD, pady=(10, 0))
            self._set_text(" در حال شنیدن… ", theme.FG)
            self.root.deiconify()
            self.place_near_mouse()
            self._visible = True

    def hide(self):
        if self._visible:
            self.root.withdraw()
            self._visible = False
            self._recording = False

    @property
    def visible(self) -> bool:
        return self._visible

    def _set_text(self, text: str, color: str):
        self.label.configure(text=text, text_color=color)
        self._reflow()

    def update_text(self, text: str):
        t = text.strip()
        self._set_text(f" {t} " if t else " در حال شنیدن… ", theme.FG)

    def set_processing(self):
        """حالت «در حال تشخیص» — هدر جمع می‌شود تا فقط متن دیده شود."""
        self._recording = False
        self.header.pack_forget()
        self._set_text(" در حال تشخیص… ", theme.FG_DIM)

    def set_font_size(self, size: int):
        """اندازه فونت متن زنده — از تنظیمات (thread اصلی)."""
        self.label.configure(font=(self.fam, int(size)))

    def update_level(self, rms: float):
        self._level = rms

    # ---------- رندر موج ----------
    def _draw_wave(self):
        c = self.wave
        c.delete("all")
        self._phase += 0.3
        target = min(1.0, self._level / 0.04)
        if target < 0.06:
            target = 0.06 + 0.04 * (math.sin(self._phase) + 1) / 2
        jitter = 0.85 + 0.3 * math.sin(self._phase * 1.7 + len(self._bars))
        self._bars = [target * jitter] + self._bars[:-1]
        w = max(c.winfo_width(), 1)
        y_mid = (WAVE_H + 4) / 2
        for i, h_norm in enumerate(self._bars):
            half = max(1.5, h_norm * (WAVE_H / 2 - 1))
            x = w - (i + 1) * (WAVE_BAR_W + WAVE_GAP)
            color = theme.ACCENT if h_norm > 0.22 else theme.SURFACE_3
            c.create_rectangle(x, y_mid - half, x + WAVE_BAR_W, y_mid + half,
                               fill=color, outline="")

    def tick(self):
        """پمپ رویداد + انیمیشن — در حلقه اصلی (thread اصلی) صدا زده می‌شود."""
        if not self._visible:
            try:
                self.root.update()
            except tk.TclError:
                pass
            return
        self._draw_wave()
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass
