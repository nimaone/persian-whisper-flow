"""رندر نرم سراسری برای CustomTkinter — وصله‌ی موتور رسم (یک‌جا برای همه‌ی ویجت‌ها).

موتور رسم CustomTkinter دایره و کلاویک را با گلیف فونت/چندضلعی می‌کشد که
در شعاع‌های کوچک لبه‌ی پله‌ای و کمی بیضی می‌دهد. این ماژول چهار متد هندسی
DrawEngine را با رندر Pillow (ابرنمونه + LANCZOS) جایگزین می‌کند.

قرارداد حفظ‌شده: رنگ‌آمیزی بخش‌ها مثل موتور اصلی با canvas.itemconfig(tag,
fill=...) است؛ draw همیشه True برمی‌گرداند (requires_recoloring). حالت‌های
پیاده‌سازی‌نشده به متد اصلی برمی‌گردند.

═════════════════════════════════════════════════════════════════════════
ریشه‌یابی «Not Responding» (اثبات‌شده با faulthandler — پشته‌ی زنده‌ی ترد UI):

  CTkScrollbar._draw (خط ۱۶۸) ← self._canvas.update_idletasks()
    ↑ CTkBaseClass.configure ← require_redraw → self._draw()
    ↑ CTkScrollbar.configure ← bg_color
    ↑ CTkFrame.configure ← انتشار bg_color به فرزندان
    ↑ CTkTabview._draw ← tab.configure(fg_color=…, bg_color=…)
    ↑ CTkBaseClass._update_dimensions_event ← رویداد <Configure>
    ↑ tkinter update_idletasks  ← تو در تو! و دوباره از بالا…

update_idletasks داخل _draw/configure خود CustomTkinter، کارهای معوق چیدمان
را «همان‌جا و همگام» اجرا می‌کند و رویدادهای <Configure> درونش دقیق و
سینکرون فراخوانی می‌شوند → هر سطح، Configure بعدی را پردازش می‌کند → بازگشت
تو در تو که در طوفان resize هیچ‌وقت تمام نمی‌شود → ترد UI قفل (Not
Responding) و سرانجام سرریز پشته‌ی C (Segfault/کرش).

سه ضمانت در این ماژول:
۱. گارد بازگشتی update_idletasks — فراخوانی تو در تو بی‌عمل است؛
   بیرونی‌ترین فراخوانی همه‌ی کارهای معوق را انجام می‌دهد (عمق ≤ ۱).
۲. هیچ رندری همگام در فعالیت انجام نمی‌شود: اگر از آخرین draw هر ویجتی
   کمتر از _QUIET گذشته باشد (طوفان)، رندر به تایمر after(۶۰ms) سپرده
   می‌شود — تایمرها را update_idletasks اجرا نمی‌کند. هزینه‌ی هر Configure
   در طوفان ≈ صفر.
۳. رندرهای کامل فقط از finalize، با فاصله‌ی ≥ _QUIET از آخرین رندرِ هر
   جای برنامه (ریتم سراسری) — ترد UI بین رندرها نفس می‌کشد.

کش شکل با کلید (اندازه، هندسه، رنگ‌ها) + سقف حافظه LRU.
فعال‌سازی: smooth_ctk.apply() — یک بار، قبل از ساخت اولین ویجت.
"""
from __future__ import annotations

import time
import tkinter as tk
from collections import OrderedDict

from PIL import Image, ImageDraw

_SS_TINY = 8    # ویجت‌های کوچک (سوییچ، رادیو، اسلایدر) — لبه به کیفیت وب
_SS_MID = 4     # متوسط (دکمه‌ها، ورودی‌ها)
_SS_BIG = 2     # بزرگ (کارت‌ها، تب‌ویو)
_QUIET = 0.045  # اگر فاصله از آخرین فعالیت/رندر کمتر از این بود، در طوفان هستیم
_FINALIZE_MS = 60
_RENDER_BUDGET = 600        # سقف رندر کامل در ۱ ثانیه (کل برنامه) — ضمانت کمکی
_COOLDOWN = 1.0             # مدت بازگشت به موتور اصلی در اشباع
_CACHE_MAX_ENTRIES = 400
_CACHE_MAX_BYTES = 64 * 1024 * 1024

_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_cache_bytes = 0
_render_times: list[float] = []      # زمان‌های رندر کامل — بودجه‌ی سراسری
_cooldown_until = 0.0                # تا این لحظه، موتور اصلی می‌کشد
_last_draw_t = 0.0                   # آخرین draw هر ویجتی — آشکارساز طوفان
_last_render_t = 0.0                 # آخرین رندر کامل — ریتم سراسری
_uid_depth = 0                       # عمق update_idletasks — گارد بازگشت
_uid_blocked = 0                     # شمار فراخوانی‌های تو در تو خنثی‌شده
_applied = False


def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


def _ss_for(w: int, h: int) -> int:
    if max(w, h) <= 64:
        return _SS_TINY
    if max(w, h) <= 200:
        return _SS_MID
    return _SS_BIG


def _overheated() -> bool:
    """اگر بودجه‌ی رندر پر شده باشد، True و شروع cooldown — فشار به موتور اصلی."""
    global _cooldown_until, _render_times
    now = time.monotonic()
    if now < _cooldown_until:
        return True
    while _render_times and _render_times[0] < now - 1.0:
        _render_times.pop(0)
    if len(_render_times) > _RENDER_BUDGET:
        _cooldown_until = now + _COOLDOWN
        _render_times.clear()
        return True
    return False


class _Render:
    """رندر یک canvas شکل‌دار: لایه‌ها به ترتیب + رنگ هر تگ + تصویر نهایی.

    یک شیء برای عمر canvas ساخته می‌شود و با update_spec در جا به‌روز می‌شود؛
    تصویر قبلی تا رندر بعدی سر جای خودش می‌ماند (بدون سوسو).
    """

    def __init__(self, canvas, w: int, h: int, parts: list[tuple], bg: tuple):
        self.canvas = canvas
        self.w, self.h = w, h
        self.bg = bg                        # RGB پس‌زمینه — یک‌بار در ساخت
        self.parts = parts                  # [(tag, x0, y0, x1, y1, radius), ...]
        self.colors: dict[str, str] = {}    # tag -> رنگ اعمال‌شده توسط ویجت
        self._photo: tk.PhotoImage | None = None
        self._last_key = None
        self._dirty = True
        self._cheap_t = 0.0                 # زمان آخرین پیش‌نمایش سریع
        self._spec_t = time.monotonic()     # زمان آخرین تغییر هندسه

    def update_spec(self, w: int, h: int, parts: list[tuple]):
        """هندسه‌ی تازه روی همان شیء — اگر واقعاً عوض شده باشد."""
        if (w, h) == (self.w, self.h) and tuple(parts) == tuple(self.parts):
            return  # هیچ چیز عوض نشده — dedup در _draw_now صفر هزینه
        self.w, self.h, self.parts = w, h, parts
        self._last_key = None
        self._dirty = True
        self._spec_t = time.monotonic()

    def recolor(self, tag: str, kw: dict):
        """رنگ جدید یک تگ از ویجت (hover و…) — رندر همگام.

        تغییرِ فقط-رنگ ارزان است (کش + بدون تغییر هندسه) و باید همگام باشد:
        ویجت در همان _draw پس‌زمینه‌ی لیبل متن را فوراً عوض می‌کند؛ اگر
        تصویر بوم دیرتر به‌روز شود، «مستطیلِ نزدیک کلمه» با رنگ قدیمی/جدید
        دیده می‌شود. بودجه‌ی سراسری (_overheated) محافظ است.
        """
        if tag not in {p[0] for p in self.parts}:
            return
        fill = kw.get("fill") or kw.get("outline")
        if fill is not None and self.colors.get(tag) != str(fill):
            self.colors[tag] = str(fill)
            self._draw_now(sync_allowed=True, color_only=True)
            self._sync_text_label()

    def _sync_text_label(self):
        """هم‌رنگ‌کردن لیبل متنِ CTkButton با رنگ درونی تصویر — همیشه.

        لیبل متن یک tk.Label مستطیلی روی بوم است؛ اگر پس‌زمینه‌اش با رنگ
        درونی تصویر فرق کند، «مستطیل نزدیک کلمه» دیده می‌شود. اینجا بعد از
        هر recolor هم‌زمانش می‌کنیم (چه رندر همگام، چه deferred).
        """
        label = getattr(self.canvas.master, "_text_label", None)
        if label is None:
            return
        inner = self.colors.get("inner_parts")
        if inner is not None and str(label.cget("bg")) != str(inner):
            try:
                label.configure(bg=str(inner))
            except tk.TclError:
                pass
    # ---------- رندر ----------
    def _key(self):
        return (self.w, self.h, self.bg,
                tuple(self.parts),
                tuple(sorted(self.colors.items())))

    def _ppm_quality(self, key) -> bytes:
        """بایت‌های PPM با کیفیت کامل — از کش؛ در نبود، رندر ابرنمونه."""
        global _cache_bytes
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

        ss = _ss_for(self.w, self.h)
        img = Image.new("RGBA", (self.w * ss, self.h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        self._paint(d, ss)
        flat = Image.alpha_composite(Image.new("RGBA", img.size, self.bg + (255,)), img)
        small = flat.convert("RGB").resize((self.w, self.h), Image.LANCZOS)
        data = b"P6 %d %d 255 " % (self.w, self.h) + small.tobytes()

        _cache[key] = data
        _cache_bytes += len(data)
        while len(_cache) > _CACHE_MAX_ENTRIES or _cache_bytes > _CACHE_MAX_BYTES:
            _, evicted = _cache.popitem(last=False)
            _cache_bytes -= len(evicted)
        return data

    def _paint(self, d: ImageDraw.ImageDraw, ss: int):
        for tag, x0, y0, x1, y1, radius in self.parts:
            color = self.colors.get(tag)
            if color is None:
                continue  # هنوز رنگی نیامده — با itemconfig می‌آید
            rgb = self._rgb(color)
            if rgb is None:
                continue
            if x1 - 1 < x0 or y1 - 1 < y0:
                continue  # بخش صفر-عرض (مثل progress در مقدار ۰)
            d.rounded_rectangle((x0 * ss, y0 * ss, (x1 - 1) * ss, (y1 - 1) * ss),
                                radius=max(0.0, radius * ss), fill=rgb)

    def _cheap_preview(self):
        """پیش‌نمایش فوری با BILINEAR — فقط وقتی بوم هنوز هیچ تصویری ندارد.

        کیفیت ۲× کافی است چون تصویر جای خالیِ خاکستری را می‌گیرد؛ کیفیت کامل
        با finalize می‌آید. این مسیر از ریتم سراسری مستثناست تا هیچ بومی
        در «نوبت» نماند (دکمه‌های پایین آخر ساخته می‌شوند و بدون این،
        لحظه‌ای خالی/بدشکل دیده می‌شدند).
        """
        if self._photo is not None or self.w < 3 or self.h < 3:
            return
        if not all(p[0] in self.colors for p in self.parts):
            return
        try:
            ss = 2
            img = Image.new("RGBA", (self.w * ss, self.h * ss), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            self._paint(d, ss)
            flat = Image.alpha_composite(
                Image.new("RGBA", img.size, self.bg + (255,)), img)
            small = flat.convert("RGB").resize((self.w, self.h), Image.BILINEAR)
            data = b"P6 %d %d 255 " % (self.w, self.h) + small.tobytes()
            self._photo = tk.PhotoImage(data=data, master=self.canvas)
            self._place()
            self._cheap_t = time.monotonic()
        except tk.TclError:
            pass

    def _draw_now(self, sync_allowed: bool, color_only: bool = False):
        """رندر — فقط از تایمر finalize یا در سکون واقعی؛ هرگز وسط فعالیت.

        sync_allowed فقط وقتی True است که این draw، تنها فعالیت ۴۵ms اخیر
        باشد (سکون). در طوفان فقط dirty + تایمر — ترد UI آزاد می‌ماند.
        color_only: تغییر رنگ بدون تغییر هندسه (hover/state) — همگام رندر
        می‌شود تا لیبل متن و تصویر هم‌زمان شوند؛ چون کش می‌خورد و هندسه
        ثابت است ارزان است و _overheated سقفش را نگه می‌دارد.
        """
        global _last_render_t, _render_times
        if getattr(self.canvas, "_smooth_render", None) is not self:
            return  # منسوخ — نسخه‌ی جدید جایگزین شده
        try:
            if not self.canvas.winfo_exists():
                return
        except tk.TclError:
            return
        if self.w < 3 or self.h < 3:
            self._dirty = True
            return  # اندازه‌ی گذرا وسط resize — finalize کامل می‌کشد
        # پس‌زمینه‌ی واقعی canvas در هر رندر — در اولین draw ویجت هنوز روشنِ
        # پیش‌فرض است (ویجت بعد از برگشت موتور، canvas را تیره می‌کند)؛ اگر
        # یک‌بار در ساخت بخوانیم، گوشه/لبه‌ی روشن در تصویر پخته می‌شود.
        bg = _bg_rgb(self.canvas)
        if bg != self.bg:
            self.bg = bg
            self._last_key = None
        key = self._key()
        if key == self._last_key and self._photo is not None:
            return  # هیچ چیز عوض نشده — صفر میلی‌ثانیه
        # رندرِ رنگِ ناقص ممنوع: بلافاصله بعد از draw، ویجت بقیه‌ی رنگ‌ها را
        # با itemconfig می‌فرستد؛ رندر ناقص فقط کش و تصویر را شلوغ می‌کند.
        # (استثنا: color_only — رنگ‌ها همین حالا کامل‌اند، به‌روزرسانی است)
        if not color_only \
                and not all(p[0] in self.colors for p in self.parts) \
                and (time.monotonic() - self._spec_t) < 0.2:
            self._dirty = True
            self._set_finalize()
            return
        if (not color_only and (not sync_allowed
                or (time.monotonic() - _last_render_t) < _QUIET)) or _overheated():
            self._dirty = True
            self._set_finalize()
            if self._photo is None:
                self._cheap_preview()  # بوم خالی نمانَد — پیش‌نمایش فوری
            return
        try:
            data = self._ppm_quality(key)
            self._photo = tk.PhotoImage(data=data, master=self.canvas)
            self._place()
            self._last_key = key
            self._dirty = False
            _last_render_t = time.monotonic()
            _render_times.append(_last_render_t)
        except tk.TclError:
            pass  # canvas نابود شده

    def _set_finalize(self):
        """یک finalize با after — تایمرها را update_idletasks اجرا نمی‌کند."""
        if getattr(self.canvas, "_smooth_finalize_scheduled", False):
            return
        try:
            self.canvas._smooth_finalize_scheduled = True
            self.canvas.after(_FINALIZE_MS, self._finalize)
        except tk.TclError:
            self.canvas._smooth_finalize_scheduled = False

    def _finalize(self):
        try:
            self.canvas._smooth_finalize_scheduled = False
        except AttributeError:
            pass
        if getattr(self.canvas, "_smooth_render", None) is not self:
            return
        if not self._dirty:
            return
        # اگر بوم هنوز هیچ تصویری ندارد، اول پیش‌نمایش فوری — نه انتظار در نوبت
        if self._photo is None:
            self._cheap_preview()
            if self._photo is not None and \
                    (time.monotonic() - self._cheap_t) < _QUIET:
                return  # کیفیت کامل بعدی در سکون
        self._draw_now(sync_allowed=True)  # اگر هنوز طوفان است، دوباره عقب می‌افتد

    def _place(self):
        item = getattr(self.canvas, "_smooth_item", None)
        if item is None:
            item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            self.canvas.tag_lower(item)  # زیر متن/تیک/فلش ویجت
            self.canvas._smooth_item = item
        else:
            self.canvas.itemconfig(item, image=self._photo)

    def _rgb(self, color: str):
        try:
            r, g, b = self.canvas.winfo_rgb(color)
            return (r >> 8, g >> 8, b >> 8)
        except Exception:
            return None


def _isolated() -> bool:
    """True فقط اگر این رویداد، تنها فعالیت ۴۵ms اخیر باشد (سکون واقعی)."""
    return (time.monotonic() - _last_draw_t) > _QUIET


def _set_spec(canvas, w: float, h: float, parts: list[tuple]) -> bool:
    """رندرِ canvas را با هندسه‌ی تازه به‌روز می‌کند.

    مقدار بازگشتی = requires_recoloring با معنای موتور اصلی: فقط بار اول
    (ساخت رندر تازه) True — چون رنگ‌ها هنوز خالی‌اند و ویجت باید آنها را
    با itemconfig اعمال کند. drawهای بعدی False می‌دهند تا ویجت‌ها در
    مسیر سریع Configure (no_color_updates=True) از بلوک رنگی رد شوند.
    برگشتِ همیشه-True باعث می‌شد CTkTabview در هر Configure تمام
    فرزندانش را configure(bg_color=…) کند → آبشار چیدمان → نوسان ابدی
    ابعاد (حلقه‌ی رویدادیِ مستقل از ورودی) → Not Responding.

    draw فقط وضعیت را عوض می‌کند؛ رندر یا همگامِ ساکت است یا یک finalize
    با تایمر — نه after_idle، که CustomTkinter داخل _draw خودش با
    update_idletasks آن را تو در تو اجرا می‌کرد.
    """
    global _last_draw_t
    now = time.monotonic()
    isolated = (now - _last_draw_t) > _QUIET  # قبل از ثبت این draw
    _last_draw_t = now
    w, h = max(1, round(w)), max(1, round(h))
    render = getattr(canvas, "_smooth_render", None)
    fresh = render is None
    if fresh:
        render = _Render(canvas, w, h, parts, _bg_rgb(canvas))
        canvas._smooth_render = render
    else:
        render.update_spec(w, h, parts)
    render._draw_now(sync_allowed=isolated)
    return fresh


def _drop_render(canvas):
    """حذف حالت رندر نرم — قبل از بازگشت به موتور اصلی (fallback)."""
    render = getattr(canvas, "_smooth_render", None)
    if render is None:
        return
    item = getattr(canvas, "_smooth_item", None)
    if item is not None:
        try:
            canvas.delete(item)
        except tk.TclError:
            pass
        canvas._smooth_item = None
    try:
        canvas._smooth_render = None
        canvas._smooth_finalize_scheduled = False
    except Exception:
        pass


def flush_pending(root) -> None:
    """رندر فوری همه‌ی بوم‌های بدون تصویر — یک‌بار قبل از نمایش پنجره.

    در باز شدن تنظیمات ده‌ها بوم همزمان draw می‌شوند؛ آشکارساز طوفان
    رندر همگام را عقب می‌اندازد و ریتم سراسری یعنی آخرین بوم‌ها (دکمه‌های
    پایین) دیرتر تصویر می‌گیرند. این فراخوانی قبل از smooth_show همه را
    با پیش‌نمایش سریع پر می‌کند تا پنجره از اولین فریم کامل باشد.
    """
    stack = list(root.winfo_children())
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        render = getattr(w, "_smooth_render", None)
        if render is not None and render._photo is None:
            try:
                render._cheap_preview()
            except Exception:
                pass


def _bg_rgb(canvas) -> tuple:
    """پس‌زمینه‌ی واقعی canvas — از متن رنگ یا رنگ والد، به RGB."""
    try:
        r, g, b = canvas.winfo_rgb(canvas.cget("bg"))
        return (r >> 8, g >> 8, b >> 8)
    except Exception:
        return (32, 32, 32)


# ---------- معادل هندسی متدهای DrawEngine (همان ریاضی، رندر PIL) ----------

def _rect_parts(width, height, corner_radius, border_width):
    """border_parts: شکل کامل؛ inner_parts: تو‌رفته به اندازه‌ی border."""
    w, h = float(width), float(height)
    cr = _clamp(round(corner_radius), 0, min(w, h) / 2)
    bw = max(0.0, round(border_width))
    icr = _clamp(cr - bw, 0, min(w - 2 * bw, h - 2 * bw) / 2)
    return [
        ("border_parts", 0.0, 0.0, w, h, cr),
        ("inner_parts", bw, bw, w - bw, h - bw, icr),
    ]


def _spec_rect(width, height, corner_radius, border_width, *_) -> list[tuple]:
    return _rect_parts(width, height, corner_radius, border_width)


def _spec_progress(width, height, corner_radius, border_width,
                   p1, p2, orientation, *_) -> list[tuple] | None:
    if orientation != "w":
        return None  # فقط افقی پیاده شده — بقیه به موتور اصلی
    parts = _rect_parts(width, height, corner_radius, border_width)
    w, h = float(width), float(height)
    bw = max(0.0, round(border_width))
    icr = _clamp(round(corner_radius) - bw, 0, min(w - 2 * bw, h - 2 * bw) / 2)
    p1, p2 = _clamp(float(p1), 0.0, 1.0), _clamp(float(p2), 0.0, 1.0)
    x0 = bw + (w - 2 * bw) * p1
    x1 = bw + (w - 2 * bw) * p2
    parts.append(("progress_parts", x0, bw, x1, h - bw,
                  _clamp(icr, 0, (x1 - x0) / 2 if x1 > x0 else 0)))
    return parts


def _spec_slider(width, height, corner_radius, border_width,
                 button_length, button_corner_radius, value, orientation, *_) -> list[tuple] | None:
    if orientation != "w":
        return None
    parts = _rect_parts(width, height, corner_radius, border_width)
    w, h = float(width), float(height)
    cr = _clamp(round(corner_radius), 0, min(w, h) / 2)
    bw = max(0.0, round(border_width))
    icr = _clamp(cr - bw, 0, min(w - 2 * bw, h - 2 * bw) / 2)
    v = _clamp(float(value), 0.0, 1.0)
    x1 = bw + (w - 2 * bw) * v
    parts.append(("progress_parts", bw, bw, x1, h - bw,
                  _clamp(icr, 0, (x1 - bw) / 2 if x1 > bw else 0)))
    bcr = _clamp(round(button_corner_radius), 0, min(w, h) / 2)
    bl = max(0.0, round(button_length))
    cx = cr + bl / 2 + (w - 2 * cr - bl) * v
    if bl > 0:
        parts.append(("slider_parts", cx - bl / 2, 0.0, cx + bl / 2, h,
                      _clamp(bcr, 0, min(bl, h) / 2)))
    else:
        # دایره‌ی تمام‌ارتفاع (بوم) — قطر = ارتفاع، مثل سوییچ
        parts.append(("slider_parts", cx - bcr, 0.0, cx + bcr, h, bcr))
    return parts


def _spec_scrollbar(width, height, corner_radius, border_spacing,
                    start, end, orientation, *_) -> list[tuple] | None:
    w, h = float(width), float(height)
    bs = max(0.0, round(border_spacing))
    cr = _clamp(round(corner_radius) - bs, 0, min(w - 2 * bs, h - 2 * bs) / 2)
    start, end = _clamp(float(start), 0.0, 1.0), _clamp(float(end), 0.0, 1.0)
    if orientation == "w":
        x0 = bs + (w - 2 * bs) * start
        x1 = bs + (w - 2 * bs) * end
        return [("scrollbar_parts", x0, bs, x1, h - bs,
                 _clamp(cr, 0, (x1 - x0) / 2 if x1 > x0 else 0))]
    if orientation == "s":
        y0 = bs + (h - 2 * bs) * start
        y1 = bs + (h - 2 * bs) * end
        return [("scrollbar_parts", bs, y0, w - bs, y1,
                 _clamp(cr, 0, (y1 - y0) / 2 if y1 > y0 else 0))]
    return None


# ---------- نصب وصله ----------

def _install_uid_guard():
    """فراخوانی تو در تو‌ی update_idletasks را خنثی می‌کند — ریشه‌ی آبشار.

    CustomTkinter داخل _draw و configure از update_idletasks استفاده می‌کند؛
    اجرای همگامِ کارهای معوق چیدمان، رویدادهای <Configure> را همان‌جا
    فراخوانی می‌کند و هر Configure دوباره draw/update_idletasks — بازگشت
    بی‌انتها. بیرونی‌ترین فراخوانی همه‌ی کارها را انجام می‌دهد؛ تو در تو
    لازم نیست (نتیجه‌اش فقط اندازه‌ی کمی کهنه‌تر است که draw بعدی جبران
    می‌کند).
    """
    orig = tk.Misc.update_idletasks
    if getattr(orig, "_smooth_guarded", False):
        return

    def guarded(self):
        global _uid_depth, _uid_blocked
        if _uid_depth:
            _uid_blocked += 1
            return
        _uid_depth += 1
        try:
            orig(self)
        finally:
            _uid_depth -= 1

    guarded._smooth_guarded = True
    tk.Misc.update_idletasks = guarded


def _install_scrollbar_flush_guard():
    """فلاشِ همگامِ چیدمان روی هر دندانه‌ی اسکرول را می‌خواباند — ریشه‌ی
    لگِ تبهای اسکرولشونده (اندازهگیری زنده: ۶۰-۱۰۰ms گیر برای هر گام).

    CTkScrollbar._draw در انتهایش update_idletasks همگام دارد و در اسکرول
    روی هر set() صدا زده میشود؛ با یک فریم ~۴۰ ویجتیِ تب پیشرفته، این
    یعنی فلاش کامل صف چیدمان پنجره در هر دندانه. خنثی‌سازی همیشه‌ای است
    (نه فقط در طوفان — در اسکرول آهسته هم هر گام «ساکت» حساب میشد) و
    فقط همین فراخوانی را هدف میگیرد: نقاشی خودِ بوم (itemconfig) فوری
    میماند، و کارهای معوق را Tk قبل از هر فریم رندر خودش انجام میدهد —
    یعنی فقط «زودتر از لازم» حذف میشود، نه «نشدن».
    """
    from customtkinter.windows.widgets import ctk_scrollbar
    if getattr(ctk_scrollbar.CTkScrollbar._draw, "_smooth_scroll_guarded", False):
        return
    orig = ctk_scrollbar.CTkScrollbar._draw

    def _draw(self, no_color_updates=False):
        global _uid_depth
        _uid_depth += 1  # update_idletasksِ داخل _draw بی‌عمل میشود
        try:
            orig(self, no_color_updates)
        finally:
            _uid_depth -= 1

    _draw._smooth_scroll_guarded = True
    ctk_scrollbar.CTkScrollbar._draw = _draw


def apply() -> None:
    """نصب یک‌باره‌ی وصله روی DrawEngine — قبل از ساخت اولین ویجت."""
    global _applied
    if _applied:
        return
    _install_uid_guard()
    _install_scrollbar_flush_guard()
    from customtkinter.windows.widgets.core_rendering import CTkCanvas, DrawEngine

    patches = {
        "draw_rounded_rect_with_border": _spec_rect,
        "draw_rounded_progress_bar_with_border": _spec_progress,
        "draw_rounded_slider_with_border_and_button": _spec_slider,
        "draw_rounded_scrollbar": _spec_scrollbar,
    }
    for name, spec_fn in patches.items():
        _install(DrawEngine, name, spec_fn)

    # رهگیری itemconfig فقط روی canvasهای CTk — رنگ ویجت به تصویر رندر می‌رسد
    _orig_itemconfig = CTkCanvas.itemconfig

    def _itemconfig(self, tagOrId=None, cnf=None, **kw):
        render = getattr(self, "_smooth_render", None)
        if (render is not None and isinstance(tagOrId, str)
                and ("fill" in kw or "outline" in kw)):
            render.recolor(tagOrId, kw)
        return _orig_itemconfig(self, tagOrId, cnf, **kw)

    CTkCanvas.itemconfig = _itemconfig
    _applied = True


def _install(DrawEngine, name: str, spec_fn):
    original = getattr(DrawEngine, name)

    def wrapper(self, *args, **kwargs):
        try:
            if time.monotonic() < _cooldown_until:
                _drop_render(self._canvas)
                return original(self, *args, **kwargs)  # اشباع — موتور اصلی
            # ویجت‌های با «گوشه‌های رنگی» (SegmentedButton/TabView): موتور اصلی
            # ۴ مربع گوشه روی canvas می‌کشد که با تصویر تخت‌شده جمع نمی‌شود — برگشت
            if name == "draw_rounded_rect_with_border" and \
                    self._canvas.find_withtag("background_parts"):
                _drop_render(self._canvas)
                return original(self, *args, **kwargs)
            parts = spec_fn(*args, **kwargs)
        except Exception:
            parts = None
        if parts is None:
            _drop_render(self._canvas)
            return original(self, *args, **kwargs)  # حالت پیاده‌سازی‌نشده
        return _set_spec(self._canvas, args[0], args[1], parts)

    setattr(DrawEngine, name, wrapper)
