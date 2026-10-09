"""پنجره overlay زنده — کارت گوشه‌گرد مینیمال نزدیک نشانگر موس.

فقط دو چیز را نشان می‌دهد: متن در حال تشخیص و یک موج ظریف متصل به سطح
واقعی میکروفون. هدر (نقطه‌ی ضبط + میان‌بر) فقط هنگام ضبط نمایان است و
در حالت «در حال تشخیص» کنار می‌رود تا متن نفَس بگیرد. ارتفاع کارت با
تعداد خطوط متن رشد می‌کند و پایین نمی‌زند — ولی از کسری از ارتفاع صفحه
بیشتر نمی‌شود: سرِ متن قطعی کوتاه و با «…» علامت می‌خورد، چون پنجره‌ی
جاری (چیزی که کاربر همین حالا می‌گوید) همیشه باید کامل دیده شود.

متن دو تکه است: پیشوند قفل‌شده با رنگ روشن و پنجره‌ی جاری — واژه‌هایی
که هنوز قطعی نشده‌اند — با رنگ کم‌رنگ، تا کاربر بداند کدام بخش سند
نهایی و کدام بخش هنوز در حال شکل‌گرفیست.

کارت با موس جابه‌جا می‌شود (کشیدن روی خود کارت): تا وقتی کاربر خودش
جایی نگذاشته، هر ضبط تازه کنار نشانگر موس باز می‌شود؛ از لحظه‌ی
اولین کشیدن، جای دستی کاربر یادش می‌ماند و ضبط‌های بعدی همان‌جا باز
می‌شوند (تا ری‌استارت اپ).
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
MAX_CARD_FRAC = 0.55   # سقف ارتفاع کارت نسبت به ارتفاع صفحه
CHROME_H = 96          # هدر + موج + پدینگ‌ها — ارتفاع ثابت کارت
AVG_WORD_CHARS = 6.0   # میانگین طول واژه‌ی فارسی با فاصله
EDGE_MARGIN = 8        # کمترین فاصله‌ی کارت از لبه‌ی صفحه

# GetSystemMetrics — مستطیل کل دسکتاپ (همه‌ی مانیتورها)
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def virtual_screen() -> tuple[int, int, int, int] | None:
    """مستطیل همه‌ی مانیتورها به‌شکل (x, y, w, h) — None اگر در دسترس نبود.

    Tk روی ویندوز فقط «اسکرین» (مانیتور اصلی) را می‌شناسد؛ برای اینکه کارت
    روی مانیتور دوم هم آزادانه جابه‌جا شود، مرجع باید کل دسکتاپ باشد —
    که مبدأش می‌تواند منفی هم باشد (مانیتور سمت چپ).
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32  # noqa: SLF001
        x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        if w > 0 and h > 0:
            return int(x), int(y), int(w), int(h)
    except Exception:
        pass
    return None


def clamp_pos(x: int, y: int, w: int, h: int,
              screen: tuple[int, int, int, int] | None,
              margin: int = EDGE_MARGIN) -> tuple[int, int]:
    """مختصات را مثل کلیپ درون `screen` نگه می‌دارد — خروجی همیشه int.

    screen=None یعنی منابع مرجع نداریم؛ مختصات دست‌نخورده برمی‌گردد.
    """
    if screen is None or w <= 0 or h <= 0:
        return int(x), int(y)
    sx, sy, sw, sh = screen
    left = sx + margin
    top = sy + margin
    right = sx + sw - w - margin
    bottom = sy + sh - h - margin
    cx = min(max(int(x), left), max(left, right))
    cy = min(max(int(y), top), max(top, bottom))
    return cx, cy


def split_display(text: str, provisional_words: int, limit: int) -> tuple[str, str, bool]:
    """نمایش را به (متن قطعی، پنجره‌ی جاری، بریده‌شده) می‌شکند.

    `provisional_words` = تعداد واژه‌های پایانی که هنوز قطعی نشده‌اند.
    اگر واژه‌ها از `limit` بیشتر شوند، سرِ متن قطعی کوتاه و با «…»
    علامت می‌خورد؛ پنجره‌ی جاری هرگز بریده نمی‌شود.
    """
    words = text.split()
    n = max(0, min(int(provisional_words), len(words)))
    if len(words) <= max(1, limit):
        return " ".join(words[:len(words) - n]), " ".join(words[len(words) - n:]), False
    tail = words[len(words) - n:] if n else []
    keep = limit - len(tail)
    if keep < 2:  # فقط پنجره‌ی جاری جا می‌شود — متنی برای پیشوند نمانده
        return "", " ".join(tail), True
    start = max(0, len(words) - n - (keep - 1))
    head = ["…"] + words[start:len(words) - n]
    return " ".join(head), " ".join(tail), True


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

        # متن زنده — دو تکه: پیشوند قفل‌شده (روشن) + پنجره‌ی جاری (کم‌رنگ)
        self.text_box = ctk.CTkFrame(self.card, fg_color="transparent")
        self.text_box.pack(fill="x", padx=PAD, pady=(8, 10))
        self.label = ctk.CTkLabel(
            self.text_box, text=" در حال شنیدن… ",
            font=(self.fam, 15), text_color=theme.FG,
            justify="right", anchor="e", wraplength=BODY_W,
        )
        self.tail = ctk.CTkLabel(
            self.text_box, text="", font=(self.fam, 15),
            text_color=theme.FG_DIM, justify="right", anchor="e",
            wraplength=BODY_W,
        )
        self._font_size = 15
        self._live_text = ""
        self._live_prov = 0
        self._render_parts("در حال شنیدن…")

        # موج
        self.wave = tk.Canvas(self.card, width=BODY_W, height=WAVE_H + 4,
                              bg=theme.SURFACE, highlightthickness=0)
        self.wave.pack(fill="x", padx=PAD, pady=(0, 12))

        self._level = 0.0
        self._bars = [0.05] * WAVE_BARS
        self._phase = 0.0
        self._visible = False
        self._recording = False

        # جابه‌جایی با موس — _enable_drag روی کارت و همه‌ی فرزندانش بایند می‌شود
        self._drag_off: tuple[int, int] | None = None
        self._manual_pos: tuple[int, int] | None = None
        self._enable_drag(self.card)

    # ---------- جابه‌جایی با موس ----------
    def _enable_drag(self, widget):
        """کشیدن کارت با موس — روی ویجت و همه‌ی فرزندانش (متن/موج/هدر).

        ویجت‌ها رویداد را به والد نمی‌دهند، پس بایند باید تک‌تک شود؛ وگرنه
        کشیدن فقط روی حاشیه‌ی کارت کار می‌کند و روی خود متن نه.

        نکته: ویجت‌های CustomTkinter (CTkFrame/CTkLabel/…) bind خودشان را روی
        canvas درونی‌شان می‌برند و فقط `add="+"` را می‌پذیرند؛ پس همان یکی
        کافی است و اجزای درونی‌شان (`_canvas`/`_label`) دوباره بایند نمی‌شوند.
        """
        self._bind_drag(widget)
        for child in widget.winfo_children():
            if self._is_internal_child(widget, child):
                continue  # اجزای درونی CTk — از bind خودش پوشش داده شده
            self._enable_drag(child)

    @staticmethod
    def _is_internal_child(parent, child) -> bool:
        """ویجت درونی CustomTkinter — هر ویجتی که والد به‌طور خصوصی نگهش داشته."""
        return any(value is child
                   for key, value in vars(parent).items() if key.startswith("_"))

    def _bind_drag(self, widget) -> bool:
        try:
            widget.bind("<ButtonPress-1>", self._drag_start, add="+")
            widget.bind("<B1-Motion>", self._drag_move, add="+")
            widget.bind("<ButtonRelease-1>", self._drag_end, add="+")
        except (tk.TclError, ValueError):
            return False  # ویجت بایندپذیر نبود — بقیه‌ی شاخه‌ها امتحان می‌شوند
        try:
            widget.configure(cursor="fleur")  # نشانه‌ی «قابل کشیدن»
        except Exception:
            pass
        return True

    def _drag_start(self, e):
        """فاصله‌ی موس تا گوشه‌ی پنجره — تا کارت زیر موس نپرد."""
        self._drag_off = (e.x_root - self.root.winfo_x(),
                          e.y_root - self.root.winfo_y())

    def _drag_move(self, e):
        if self._drag_off is None:
            return
        try:
            w = self.root.winfo_width()
            h = self.root.winfo_height()
        except tk.TclError:
            return
        x = e.x_root - self._drag_off[0]
        y = e.y_root - self._drag_off[1]
        x, y = clamp_pos(x, y, w or 1, h or 1, self._screen_rect())
        self.root.geometry(f"+{x}+{y}")
        self._manual_pos = (x, y)

    def _drag_end(self, e=None):
        self._drag_off = None

    def _screen_rect(self) -> tuple[int, int, int, int] | None:
        """مرجع کلیپ‌کردن مختصات — کل دسکتاپ؛ در نبودش مانیتور اصلی."""
        rect = virtual_screen()
        if rect is not None:
            return rect
        try:
            return (0, 0, self.root.winfo_screenwidth(),
                    self.root.winfo_screenheight())
        except tk.TclError:
            return None

    # ---------- مکان‌یابی ----------
    def _reflow(self):
        """اندازه‌ی کارت را با محتوای فعلی هم‌تراز می‌کند و در صفحه نگه می‌دارد."""
        self.root.update_idletasks()
        w = self.card.winfo_reqwidth() + 2
        h = self.card.winfo_reqheight() + 2
        if (w, h) != (self.root.winfo_width(), self.root.winfo_height()):
            self.root.geometry(f"{w}x{h}")
        self._clamp_on_screen()

    def _clamp_on_screen(self):
        """کارت با رشد متن یا کشیدن دستی از لبه‌ی صفحه بیرون نزند."""
        try:
            x, y = self.root.winfo_x(), self.root.winfo_y()
            w, h = self.root.winfo_width(), self.root.winfo_height()
        except tk.TclError:
            return
        pos_x, pos_y = clamp_pos(x, y, w, h, self._screen_rect())
        if (pos_x, pos_y) != (x, y):
            self.root.geometry(f"+{pos_x}+{pos_y}")

    def _max_words(self) -> int:
        """سقف واژه‌های نمایش — کارت از سقف ارتفاع رد نشود."""
        size = float(self._font_size)
        per_line = max(8.0, BODY_W / (0.55 * size))
        usable = max(60.0, self._max_card_h() - CHROME_H)
        lines = max(3.0, usable / (1.5 * size))
        return int(lines * per_line / AVG_WORD_CHARS)

    def _max_card_h(self) -> float:
        try:
            screen_h = float(self.root.winfo_screenheight())
        except tk.TclError:
            screen_h = 1080.0
        return min(screen_h * MAX_CARD_FRAC, screen_h - 2 * (PAD + 8))

    def place_near_mouse(self):
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
        except tk.TclError:
            x, y = 200, 200
        self._reflow()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        # بدون کلیپ مستقیم — کارت همان‌جا باز می‌شود که موس است و بعد
        # روی لبه‌ها کلیپ می‌شود (مانیتور دوم را هم پوشش می‌دهد)
        self.root.geometry(f"+{x - int(w * 0.3)}+{y + 26}")
        self._clamp_on_screen()

    # ---------- رابط عمومی ----------
    def show(self):
        if not self._visible:
            self._bars = [0.05] * WAVE_BARS
            self._recording = True
            self.dot.pack(side="right")
            self.hint.configure(text="بکش تا جابه‌جا شود — همان کلید را دوباره بزن")
            self.header.pack(fill="x", padx=PAD, pady=(10, 0))
            self._live_text = ""
            self._live_prov = 0
            self._render_parts("در حال شنیدن…")
            self.root.deiconify()
            if self._manual_pos is None:
                self.place_near_mouse()
            else:
                # کاربر کارت را جابه‌جا کرده — ضبط بعدی همان‌جا باز می‌شود
                x, y = clamp_pos(*self._manual_pos,
                                 self.root.winfo_width() or 1,
                                 self.root.winfo_height() or 1,
                                 self._screen_rect())
                self._manual_pos = (x, y)
                self.root.geometry(f"+{x}+{y}")
                self._reflow()
            self._visible = True

    def hide(self):
        if self._visible:
            self.root.withdraw()
            self._visible = False
            self._recording = False

    @property
    def visible(self) -> bool:
        return self._visible

    def _render_parts(self, head: str, tail: str = "", head_color: str = theme.FG):
        """متن قطعی (روشن) و پنجره‌ی جاری (کم‌رنگ) را جدا می‌چیند."""
        self.label.pack_forget()
        self.tail.pack_forget()
        head = head.strip()
        tail = tail.strip()
        if head:
            self.label.configure(text=f" {head} ", text_color=head_color)
            self.label.pack(fill="x", pady=(0, 2 if tail else 0))
        if tail:
            self.tail.configure(text=f" {tail} ")
            self.tail.pack(fill="x")
        if not head and not tail:
            self.label.configure(text=" در حال شنیدن… ", text_color=theme.FG)
            self.label.pack(fill="x")
        self._reflow()

    def _render_live(self):
        """نمایش فعلی را با سقف ارتفاع جاری می‌چیند."""
        if not self._live_text:
            self._render_parts("")
            return
        head, tail, _ = split_display(
            self._live_text, self._live_prov, self._max_words()
        )
        self._render_parts(head, tail)

    def update_text(self, text: str, provisional_words: int = 0):
        """متن زنده — `provisional_words` واژه‌های پایانیِ هنوز قطعی‌نشده است."""
        self._live_text = " ".join(text.split())
        self._live_prov = max(0, int(provisional_words))
        self._render_live()

    def set_processing(self):
        """حالت «در حال تشخیص» — هدر جمع می‌شود تا فقط متن دیده شود."""
        self._recording = False
        self.header.pack_forget()
        self._render_parts("در حال تشخیص…", "", theme.FG_DIM)

    def set_font_size(self, size: int):
        """اندازه‌ی فونت متن زنده — از تنظیمات (thread اصلی)."""
        self._font_size = int(size)
        font = (self.fam, self._font_size)
        self.label.configure(font=font)
        self.tail.configure(font=font)
        self._render_live()

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
