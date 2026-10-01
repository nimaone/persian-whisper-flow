"""پنجره تنظیمات دیکته‌یار با Flet — قرینه‌ی app/settings_ui.py.

۵ تب: عمومی، میکروفون، درج متن، پیشرفته، راهنما.
همان کارت‌ها، همان متن‌ها، همان تم. تست صدا با ضبط واقعی RMS
(قرینه‌ی MicTester نسخه‌ی CTk) و انیمیشن روی ایونت‌لوپ فلت.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time

import flet as ft
import numpy as np

from app.config import APP_TITLE, APP_TITLE_FULL, APP_VERSION, DEFAULTS, Config
from flet_ui import theme as t

REC_MAX_SEC = 10.0  # سقف ایمنی ضبط ثبت واژه — توقف خودکار

# رنگ نشانگر کیفیت ورودی تب میکروفون — بر اساس سطح input_quality()
_QUALITY_COLORS = {"good": t.ACCENT, "warn": t.WARN,
                   "bad": t.DANGER, "none": t.FG_DIM}


def _friendly_audio_error(msg: str) -> str:
    """پیام خطای صوتی قابل‌فهم — مسیرهای WDM-KS/مجازی باز نمی‌شوند."""
    low = (msg or "").lower()
    if "unanticipated host error" in low or "error starting stream" in low:
        return ("این مسیر دستگاه روی این سیستم باز نمی‌شود — اگر میکروفون مجازی "
                "است برنامه‌اش را اجرا کن، یا مسیر دیگری (مثلاً WASAPI) همان "
                "میکروفون را انتخاب کن")
    return msg


def _load_enroll_store():
    """فروشگاه واژه‌های ثبت‌شده — بی‌صدا خالی اگر فایل خراب باشد."""
    from app import enroll as enroll_mod
    try:
        return enroll_mod.EnrollStore.load()
    except Exception:
        return enroll_mod.EnrollStore()

HK_HINT = "برای ثبت میان‌بر جدید، روی کادر کلیک کن و ترکیب دلخواه را بفشار (لغو: Esc)"
HK_BAD = "ترکیب باید شامل کلید ترکیبی (کنترل، آلت یا شیفت) باشد، یا یک کلید F"
HK_INVALID = "کلید میانبر نامعتبر است"
HW_EMPTY = "حالت واژه‌های حساس روشن است ولی هیچ واژه‌ای وارد نشده"
_MOD_MAP = {"left ctrl": "ctrl", "right ctrl": "ctrl",
            "left shift": "shift", "right shift": "shift",
            "left alt": "alt", "right alt": "alt"}
_WIN_KEYS = ("left windows", "right windows")


def _valid_hotkey(hk: str) -> bool:
    """اعتبارسنجی ترکیب میان‌بر — قرینه‌ی _valid_hotkey نسخه CTk."""
    parts = [p for p in hk.split("+") if p]
    if not parts:
        return False
    if parts[-1] in ("ctrl", "shift", "alt"):
        return False  # کلید نهایی نمی‌تواند خودِ modifier باشد
    if len(parts) == 1:
        # تک‌کلیدی فقط برای F-keyها مجاز است
        return parts[0].startswith("f") and parts[0][1:].isdigit()
    return True


class SettingsWindow:
    """قرینه‌ی open_settings — هر متد عمومی با نسخه‌ی CTk هم‌نام است."""

    def __init__(self, page: ft.Page):
        self.page = page
        t.install_fonts(page)
        t.apply_icon(page)
        # مثل نسخه CTk: فقط نام کوتاه — «تنظیمات — دیکته‌یار»
        page.title = f"تنظیمات — {APP_TITLE}"
        page.bgcolor = t.BG
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 560   # هم‌اندازه‌ی win.geometry("560x640") نسخه CTk
        page.window.height = 640
        page.window.resizable = True
        page.padding = 0
        page.rtl = True

        cfg = self._load()
        self.cfg = cfg
        self.result: dict | None = None   # None = بسته بدون ذخیره؛ dict = ذخیره
        self.on_save = None

        # ---------- تب‌ها (Flet 0.86: TabBar + TabBarView داخل Tabs) ----------
        tab_names = ["عمومی", "میکروفون", "درج متن", "پیشرفته", "راهنما"]
        tab_views = [
            self._tab_general(), self._tab_mic(), self._tab_insert(),
            self._tab_advanced(), self._tab_help(),
        ]
        self.tab_bar = ft.TabBar(
            tabs=[ft.Tab(label=n) for n in tab_names],
            label_text_style=t.fam("bold", 13),
            indicator_color=t.ACCENT,
            divider_color=t.BORDER,
            # عرض طبیعی هر تب (به‌جای پخش تمام‌عرض) + وسط‌چین —
            # قرینه‌ی هدر جمع‌شده‌ی وسط CTkTabview
            scrollable=True,
            tab_alignment=ft.TabAlignment.CENTER,
        )
        self.tabs = ft.Tabs(
            length=len(tab_names),
            selected_index=0,
            animation_duration=150,
            content=ft.Column(
                expand=True,
                controls=[
                    self.tab_bar,
                    ft.TabBarView(
                        expand=True,
                        controls=tab_views,
                    ),
                ],
            ),
            expand=True,
        )

        # ---------- نوار دکمه‌های ثابت پایین (RTL: اولین = راست‌ترین؛
        # مثل CTk: ذخیره راست، بعد انصراف، بازنشانی چپ) ----------
        btn_bar = ft.Row(
            [
                ft.Button(
                    content="ذخیره", on_click=self._save, width=130, height=40,
                    bgcolor=t.ACCENT, color=t.ON_ACCENT,
                    style=t.btn_style(weight="bold"),
                ),
                ft.Button(
                    content="انصراف", on_click=self._close, width=110, height=40,
                    bgcolor=t.SURFACE_2, color=t.FG,
                    style=t.btn_style(),
                ),
                ft.Container(expand=True),
                ft.Button(
                    content="بازنشانی", on_click=self._reset, width=100, height=40,
                    bgcolor=t.SURFACE_2, color=t.FG,
                    style=t.btn_style(),
                ),
            ],
        )

        page.add(
            ft.Column(
                [ft.Container(self.tabs, expand=True,
                              padding=ft.Padding(left=18, top=14, right=18, bottom=0)),
                 ft.Container(btn_bar, padding=ft.Padding(left=18, top=8, right=18, bottom=8))],
                expand=True,
            )
        )

    # ---------- داده‌ی config (dict خام؛ None یعنی DEFAULTS) ----------
    def _load(self) -> dict:
        from app.config import Config
        return dict(Config.load().data)

    # ================================================= عمومی
    def _tab_general(self):
        cfg = self.cfg
        # فقط‌خواندنی — ترکیب با کیبورد کپچر می‌شود، تایپ دستی نه
        # (قرینه‌ی bind("<Key>", capture_hotkey) در CTk)
        self.var_hotkey = ft.TextField(
            value=cfg.get("hotkey"), read_only=True,
            on_focus=self._hk_focus, on_blur=self._hk_blur, expand=True,
            bgcolor=t.SURFACE_2, border_color=t.BORDER, border_radius=8,
            text_style=t.fam("Regular", 14), height=40, rtl=True,
        )
        self.hk_hint = ft.Text(
            HK_HINT,
            style=t.fam("Regular", 12), color=t.FG_DIM, text_align=ft.TextAlign.RIGHT,
        )

        c_hotkey = t.card("کلید میانبر شروع/توقف ضبط", self.var_hotkey, self.hk_hint)

        self.var_overlay = ft.Switch(value=bool(cfg.get("overlay_enabled")),
                                     active_color=t.ACCENT, scale=0.9)
        self.font_lbl = ft.Text(f"اندازه متن: {int(cfg.get('overlay_font_size') or 15)}",
                                style=t.fam("Regular", 12), color=t.FG_DIM,
                                text_align=ft.TextAlign.RIGHT)
        self.var_font = ft.Slider(min=12, max=20, divisions=8,
                                  value=int(cfg.get("overlay_font_size") or 15),
                                  active_color=t.ACCENT, expand=True,
                                  on_change=self._font_slide)
        self.sample = ft.Text("نمونه متن فارسی", style=t.fam("Regular", 15), color=t.FG,
                              text_align=ft.TextAlign.RIGHT)

        c_live = t.card(
            "پنجره زنده و سیستم",
            ft.Row([t.row_label("نمایش پنجره زنده هنگام ضبط"), self.var_overlay]),
            ft.Container(padding=ft.Padding(left=0, top=4)),
            ft.Row([self.font_lbl, self.var_font]),
            self.sample,
        )

        self.var_autostart = ft.Switch(value=bool(cfg.get("autostart")),
                                       active_color=t.ACCENT, scale=0.9)
        c_live.content.controls.append(
            ft.Container(
                ft.Row([t.row_label("اجرای خودکار با ورود به ویندوز"), self.var_autostart]),
                padding=ft.Padding(left=0, top=10),
            )
        )

        return ft.Column([c_hotkey, c_live], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def _font_slide(self, e):
        v = int(e.control.value)
        self.font_lbl.value = f"اندازه متن: {v}"
        self.sample.style = t.fam("Regular", v)
        self.page.update()

    # ---------- کپچر کلید میانبر (قرینه‌ی capture_hotkey در CTk) ----------
    # فلگ‌های ctrl/alt در KeyboardEvent کلاینت ویندوز فلت قابل اعتماد
    # نیستند (اسپایک: حتی با SendInput واقعی False می‌آیند) — وضعیت
    # modifier از جریان hook کتابخانه‌ی keyboard ساخته می‌شود؛ همان
    # کتابخانه‌ای که اپ برای ثبت هات‌کی به‌کار می‌برد، پس نام کلیدها
    # (وابسته به لی‌اوت) با ثبت نهایی سازگار است.
    def _set_hk_hint(self, text: str, color: str):
        self.hk_hint.value = text
        self.hk_hint.color = color
        self._schedule_update()

    def _schedule_update(self):
        """ارسال پچ از ایونت‌لوپ فلت — کال‌بک hook در ترد خودش اجرا
        می‌شود و put_nowait از ترد فرعی به صف ارسال کلاینت نمی‌رسد."""
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            run_task(self._flush_update)
        else:
            self._safe_update(self.hk_hint)    # MockPage — مسیر sync تست
            self._safe_update(self.var_hotkey)

    async def _flush_update(self):
        try:
            self.page.update()
        except Exception:
            pass  # پنجره بسته شده

    def _hk_hook_on(self):
        try:
            import keyboard as kb
            kb.hook(self._hk_raw)
            self._hk_hooked = True
        except Exception:
            self._hk_hooked = False

    def _hk_hook_off(self):
        if getattr(self, "_hk_hooked", False):
            self._hk_hooked = False
            try:
                import keyboard as kb
                kb.unhook(self._hk_raw)
            except Exception:
                pass

    def _hk_focus(self, e=None):
        self._hk_capturing = True
        self._hk_prev = self.var_hotkey.value
        self._hk_mods = set()
        self._hk_hook_on()
        self._set_hk_hint(HK_HINT, t.FG_DIM)

    def _hk_blur(self, e=None):
        self._hk_capturing = False
        self._hk_mods = set()
        self._hk_hook_off()

    def _hk_raw(self, e):
        """کال‌بک hook کتابخانه keyboard — روی ترد hook اجرا می‌شود.

        Esc = لغو و بازگشت به مقدار قبلی؛ کلیدهای modifier فقط وضعیت
        می‌سازند؛ تک‌کلیدی بدون modifier تایپ عادی ویندوز را می‌شکند
        (فقط F-key مجاز — مثل CTk).
        """
        name = e.name
        if e.event_type == "up":
            mod = _MOD_MAP.get(name)
            if mod:
                self._hk_mods.discard(mod)
            return
        if not getattr(self, "_hk_capturing", False):
            return
        mod = _MOD_MAP.get(name)
        if mod:
            self._hk_mods.add(mod)
            return
        if name in _WIN_KEYS:
            return
        if name == "esc":
            self.var_hotkey.value = getattr(self, "_hk_prev", "") or ""
            self._set_hk_hint(HK_HINT, t.FG_DIM)
            return
        mods = [m for m in ("ctrl", "shift", "alt")
                if m in getattr(self, "_hk_mods", set())]
        if not mods and not (name.startswith("f") and name[1:].isdigit()):
            self.var_hotkey.value = getattr(self, "_hk_prev", "") or ""
            self._set_hk_hint(HK_BAD, t.DANGER)
            return
        self.var_hotkey.value = "+".join(mods + [name])
        self._set_hk_hint(HK_HINT, t.FG_DIM)

    # ================================================= میکروفون
    def _tab_mic(self):
        cfg = self.cfg
        # فهرست یکدست: یک مدخل برای هر میکروفون فیزیکی — ویندوز هر دستگاه
        # را به ازای هر Host API یک بار فهرست می‌کند (۳ میکروفون → ۱۵+
        # مدخل). برچسب بدون ایندکس خام است (بین بوت‌ها ناپایدار)؛ کلید
        # پایدار «نام — API» است (قرینه‌ی تب میکروفون CTk).
        try:
            from app.recorder import (current_input_devices,
                                      dedupe_input_devices, device_label)
            all_inputs = current_input_devices()
            devices = [(d["index"], device_label(d))
                       for d in dedupe_input_devices(all_inputs)]
            cur_pin = cfg.get("input_device")
            if cur_pin is not None and cur_pin not in (idx for idx, _ in devices):
                # پین فعلی در فهرست یکدست نیامد (API کم‌ترجیح) — برای
                # دیده‌شدن انتخاب فعلی اضافه شود
                for e in all_inputs:
                    if e["index"] == cur_pin:
                        devices.append((e["index"], device_label(e)))
                        break
        except Exception:
            all_inputs = []
            devices = []
        self._devices = devices        # [(index, label)] — کلید پایدار/تست
        self._all_inputs = all_inputs  # مسیرهای جایگزین تست صدا
        self.auto_label = "خودکار (پرسیگنال‌ترین)"
        cur_dev = cfg.get("input_device")
        init_val = self.auto_label
        if cur_dev is not None:
            match = [lbl for idx, lbl in devices if idx == cur_dev]
            if match:
                init_val = match[0]
        else:
            # در حالت خودکار، اسم میکروفونی که دیکته برگزیده کنار «خودکار»
            # می‌آید — بک‌اند دستگاه فعلی‌اش را با env به این پروسه می‌فرستد
            try:
                found = int(os.environ.get("DIKTEYAR_DEVICE") or "")
            except ValueError:
                found = None
            if found is not None:
                try:
                    from app.recorder import device_label
                    lbl = next((device_label(e) for e in all_inputs
                                if e["index"] == found), None)
                    if lbl:
                        self.auto_label = f"خودکار — {lbl}"
                except Exception:
                    pass
        self.var_device = t.dropdown(
            [self.auto_label] + [lbl for _, lbl in devices], init_val, height=36)
        # داخل Row → تمامعرض مثل fill="x" در CTk. مستقیم در Columnِ کارت،
        # expand فقط ارتفاع را میکشد و کمبو به اندازهی متن گزینهها میماند
        c_dev = t.card("دستگاه ورودی", ft.Row([self.var_device]))
        c_dev.content.controls.append(
            t.dim("خودکار = پرسیگنال‌ترین میکروفون فعال در شروع هر ضبط")
        )

        # ---------- تست صدا — ضبط واقعی RMS با MicTester (قرینه‌ی CTk) ----------
        # میله‌ها در Row با expand — هر میله سهم برابر از عرض نوار می‌گیرد
        # و با تغییر اندازه‌ی پنجره هم کل نوار را پر می‌کند (گام ثابت ۹px
        # در پنجره‌ی عریض، ابتدای نوار خالی می‌گذاشت). Row داخل Containerِ
        # با ارتفاع ثابت است: تغییر height میله ارتفاع Row را عوض نمیکند و
        # فقط خود میله رندر میشود — باگ قدیمی باز-layout کل TabBarView
        # برنمی‌گردد. vertical_alignment=END کف مشترک میسازد: میله از پایین
        # به بالا رشد میکند (خط کفی SPECS_H+4 در CTk).
        self._bars: list[ft.Container] = []
        for i in range(48):
            self._bars.append(ft.Container(height=3, bgcolor=t.SURFACE_3,
                                           border_radius=2, expand=True))
        self.spec_bars = ft.Row(self._bars, spacing=3,
                                vertical_alignment=ft.CrossAxisAlignment.END)
        self.verdict = ft.Text("", style=t.fam("bold", 13), text_align=ft.TextAlign.RIGHT)
        # نشانگر کیفیت ورودی — نویز پایه/اوج/SNR زنده حین تست
        self.quality_lbl = ft.Text("", style=t.fam("Regular", 12),
                                   color=t.FG_DIM, text_align=ft.TextAlign.RIGHT)
        self.test_btn = ft.Button(
            content="شروع تست", on_click=self._toggle_test, width=120, height=34,
            bgcolor=t.SURFACE_2, color=t.FG,
            style=t.btn_style(weight="bold"),
        )
        # ترتیب و چینش مثل CTk: موج → دکمه وسط‌چین → حکم. نوار تمامعرض با
        # پسزمینهی DEEP — قرینهی کانوس fill="x" با bg=DEEP در CTk؛
        # padding پایین ۴ = خط کفی SPECS_H+4، میلهها روی آن رشد میکنند
        self.spec_wave = ft.Container(
            self.spec_bars, bgcolor=t.DEEP, height=28, expand=True,
            padding=ft.Padding(left=8, right=8, top=0, bottom=4),
        )
        c_test = t.card("تست صدا",
                        self.spec_wave,
                        ft.Row([self.test_btn], alignment=ft.MainAxisAlignment.CENTER),
                        self.verdict,
                        self.quality_lbl)

        auto_stop_labels = ["خاموش", "۳ ثانیه", "۵ ثانیه", "۱۰ ثانیه"]
        self.var_auto_stop = t.dropdown(
            auto_stop_labels,
            # مقدار ذخیره‌شده‌ی cfg — وگرنه هر بازشدن «خاموش» نشان می‌داد و
            # ذخیره‌ی بعدی، تنظیم کاربر را بی‌صدا صفر می‌کرد (قرینه‌ی CTk)
            {0: "خاموش", 3: "۳ ثانیه", 5: "۵ ثانیه", 10: "۱۰ ثانیه"}.get(
                int(cfg.get("auto_stop_sec") or 0), "خاموش"),
            width=110,
        )
        c_beh = t.card(
            "رفتار ضبط",
            ft.Row([t.row_label("توقف خودکار پس از سکوت"), self.var_auto_stop]),
            t.dim("اگر بعد از صحبت، N ثانیه سکوت کنی ضبط خودکار تمام و متن درج می‌شود"),
        )

        self.var_sound = ft.Switch(value=bool(cfg.get("sound_feedback")),
                                   active_color=t.ACCENT, scale=0.9)
        c_beh.content.controls.append(
            ft.Container(
                ft.Row([t.row_label("بوق کوتاه هنگام شروع و پایان ضبط"), self.var_sound]),
                padding=ft.Padding(left=0, top=6),
            )
        )

        return ft.Column([c_dev, c_test, c_beh], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def _safe_update(self, ctrl):
        """آپدیت یک کنترل — بی‌صدا اگر به صفحه وصل نیست (تست/پنجره بسته)."""
        try:
            ctrl.update()
        except Exception:
            try:
                self.page.update()
            except Exception:
                pass

    def _page_alive(self) -> bool:
        """پنجره هنوز باز است؟ (توقف تایمر تست بعد از بستن پنجره)"""
        try:
            _ = self.page.window
            return True
        except Exception:
            return False

    # ---------- تست صدا: ضبط واقعی RMS (قرینه‌ی MicTester در CTk) ----------
    def _tester_tried(self, dev: int | None) -> list:
        """زنجیره‌ی تست صدا: مسیر اصلی → مسیرهای دیگر Host API همان
        میکروفون فیزیکی — شکست یک مسیر تست را کلاً نمی‌اندازد."""
        try:
            from app.recorder import current_input_devices, device_siblings
            inputs = getattr(self, "_all_inputs", []) or current_input_devices()
            self._test_fallbacks = [e["index"] for e in
                                    device_siblings(inputs, dev)]
        except Exception:
            self._test_fallbacks = []
        return [dev] + list(self._test_fallbacks)

    def _start_tester(self):
        """MicTester را روی دستگاه انتخابی روشن کن؛ زنجیره‌ی جایگزین
        (مسیرهای دیگر Host API همان میکروفون) اگر مسیر اصلی باز نشود.

        صف و رویدادِ توقف، محلیِ کلوژرِ ترد گرفته می‌شوند — نه attribute —
        تا تردِ تستر قبلی با Event خودش قطعاً تمام شود و به صف تازه نریزد.
        """
        import queue as _q
        import threading as _th

        self._stop_tester()   # هر تستر قبلی خاموش شود
        self._test_vals = []  # آمار کیفیت از تازه — نه بقایای تست قبلی
        q: _q.Queue = _q.Queue()
        stop = _th.Event()
        self._tester_q = q
        self._tester_stop = stop

        dev = self._selected_device()
        if dev is None:
            # خودکار = همان دستگاهی که ضبط واقعی استفاده می‌کند (بک‌اند
            # با env می‌فرستد) — probe دوباره نه؛ نتایج detect ناپایدارند
            # و ممکن است به دستگاهی بیفتد که استریم باز نمی‌کند
            try:
                dev = int(os.environ.get("DIKTEYAR_DEVICE") or "")
            except ValueError:
                dev = None
            if dev is None:
                try:
                    from app.recorder import detect_best_device
                    dev = detect_best_device()
                except Exception:
                    dev = None
                if dev is None:
                    import sounddevice as sd
                    dev = int(sd.default.device[0])
        # مسیرهای جایگزین همان میکروفون فیزیکی — اگر مسیر اصلی
        # باز نشود، تست روی مسیر دیگر (مثلاً WASAPI) می‌رود
        tried = self._tester_tried(dev)

        def run():
            import numpy as np
            try:
                import sounddevice as sd
                last_err: Exception | None = None
                for d in tried:
                    try:
                        info = sd.query_devices(d, "input")
                        sr = int(info["default_samplerate"])

                        def cb(indata, frames, t, status):
                            if stop.is_set():
                                raise sd.CallbackStop
                            rms = float(np.sqrt(
                                (indata[:, 0].astype(np.float64) ** 2).mean()))
                            try:
                                q.put_nowait(("rms", rms))
                            except _q.Full:
                                pass

                        with sd.InputStream(device=d, channels=1,
                                            samplerate=sr, dtype="float32",
                                            blocksize=int(sr * 0.05),
                                            callback=cb):
                            while not stop.is_set():
                                _th.Event().wait(0.05)
                        return  # توقف تمیز — تمام
                    except Exception as e:
                        if stop.is_set():
                            return
                        last_err = e
                if last_err is not None:
                    q.put_nowait(("err", str(last_err)[:60]))
            except Exception as e:
                try:
                    q.put_nowait(("err", str(e)[:60]))
                except _q.Full:
                    pass

        _th.Thread(target=run, daemon=True).start()

    def _stop_tester(self):
        """قطع ضبط تست — بی‌صدا اگر هرگز روشن نشده بود."""
        if hasattr(self, "_tester_stop"):
            self._tester_stop.set()

    def _toggle_test(self, e):
        """شروع/توقف تست صدا — صدای واقعی میکروفون، مواج از RMS.

        حلقه‌ی انیمیشن باید روی ایونت‌لوپ فلت برود — داکیومنت رسمی:
        page.run_task = «Run handler coroutine as a new Task in the event
        loop». آپدیت از تردِ threading.Timer به صف‌ی ارسال کلاینت
        (asyncio.Queue) نمی‌رسد و هیچ‌وقت رندر نمی‌شود؛ میله‌ها در Rowِ
        داخل Containerِ هم‌ارتفاع‌اند و یک update() والد، دیف همه‌ی میله‌ها را می‌فرستد.
        """
        self._testing = not getattr(self, "_testing", False)
        self.test_btn.content = "توقف تست" if self._testing else "شروع تست"
        self._safe_update(self.test_btn)
        if self._testing:
            self._test_vals = []      # نمونه‌های RMS — ورودی سنجه‌ی کیفیت
            self._quality_tick = 0
            self._start_tester()
            vals = [0.0] * 48
            # پله‌ی تطبیقی (AGC): env نرمِ بیشینه‌ی صدای شنیده‌شده است؛
            # تقسیم rms/0.04 ثابت روی میکروفون‌های کم‌صدا موج را زیر
            # پیکسل می‌برد و انیمیشن دیده نمی‌شود. env با v واقعی بالا
            # می‌رود و آرام پایین می‌آید تا موج همیشه مرئی باشد.
            env = [0.004]   # شروع از آستانه‌ی «صدای واضح» CTk
            run_task = getattr(self.page, "run_task", None)
            if run_task is not None:
                # مسیر اصلی: Task روی ایونت‌لوپ — تیک‌ها رندر می‌شوند
                self._test_task = run_task(self._anim_loop, vals, env)
            else:
                # MockPage (تست‌ها): بدون ایونت‌لوپ — تیک اول sync + تایمر
                if self._tick_once(vals, env):
                    self._start_timer(vals, env)
        else:
            self._stop_tester()
            for bar in self._bars:
                bar.height = 3
                bar.bgcolor = t.SURFACE_3
            self.verdict.value = ""
            self.quality_lbl.value = ""
            self._safe_update(self.spec_bars)
            self._safe_update(self.verdict)

    def _tick_once(self, vals: list, env: list) -> bool:
        """یک تیک انیمیشن — میله‌ها و حکم را تازه می‌کند؛ False یعنی توقف."""
        if not getattr(self, "_testing", False):
            return False
        # داده‌ی جدید از میکروفون — حداکثر RMS از همه‌ی بلاک‌های صف
        got_err = None
        got_data = False
        rms = 0.0
        while True:
            try:
                kind, v = self._tester_q.get_nowait()
            except Exception:
                break
            if kind == "err":
                got_err = v
            else:
                got_data = True
                rms = max(rms, v)
        if got_err:
            self._testing = False
            self._stop_tester()
            self.test_btn.content = "شروع تست"
            self._safe_update(self.test_btn)
            self.verdict.value = f"خطا: {_friendly_audio_error(got_err)}"
            self.verdict.color = t.DANGER
            self._safe_update(self.verdict)
            return False
        # نشانگر کیفیت ورودی — فقط نمونه‌های واقعی میکروفون. صفرِ «صف
        # خالی» نباید جمع شود وگرنه کف نویز صفر و SNR متورم می‌شود و
        # میکروفون پرنویز «خوب» خوانده میشد (قرینه‌ی انباشت vals در CTk)
        if got_data:
            if not hasattr(self, "_test_vals"):
                self._test_vals = []
            self._test_vals.append(rms)
            del self._test_vals[:-400]
            from app.recorder import input_quality
            text, level = input_quality(self._test_vals)
            self.quality_lbl.value = text
            self.quality_lbl.color = _QUALITY_COLORS.get(level, t.FG_DIM)
            self._safe_update(self.quality_lbl)
        # پله‌ی تطبیقی: بالا رفتن سریع، افت آرام (~۲s تا نصف)
        if rms > env[0]:
            env[0] = rms
        else:
            env[0] = max(0.0008, env[0] * 0.985)
        # موج نمایشی: صدای واقعی نسبت به پله، + نفس‌کشیدن همیشگی —
        # حتی دستگاه کاملاً ساکت باید خط پایه‌ی زنده نشان دهد تا
        # کاربر بفهمد تست در حال اجراست (مثل موج خنثی CTk)
        import random as _rnd
        breath = 0.22 + 0.13 * _rnd.random()  # نفس پایه ۲۲–۳۵٪ (SPECS_H=20)
        speech = min(1.0, rms / (env[0] * 1.5)) if rms > 0 else 0.0
        vals[:-1] = vals[1:]
        vals[-1] = max(breath, speech)
        for bar, v in zip(self._bars, reversed(vals)):
            bar.height = max(3, v * 20)   # SPECS_H=20 مثل CTk
            bar.bgcolor = (t.ACCENT if v > 0.5 else
                           "#4f8f68" if v > 0.15 else
                           "#3c3c3c" if v > 0.02 else "#2e2e2e")
        # حکم سه‌سطحی مثل CTk — از RMS خام، نه مقیاس‌شده
        if rms > 0.004:
            self.verdict.value = "میکروفون کار می‌کند — صدای واضح"
            self.verdict.color = t.ACCENT
        elif rms > 0.0005:
            self.verdict.value = "صدای خیلی کم — تقویت ورودی را بالا ببر"
            self.verdict.color = t.WARN
        else:
            self.verdict.value = "سیگنالی نمی‌آید — دستگاه دیگری را امتحان کن"
            self.verdict.color = t.DANGER
        # یک update والد = یک پیام برای دیف کل ۴۸ میله + حکم
        self._safe_update(self.spec_bars)
        self._safe_update(self.verdict)
        if not self._page_alive():
            self._testing = False
            self._stop_tester()
            return False  # پنجره بسته شده — تیک بعدی هم معنا ندارد
        return True

    async def _anim_loop(self, vals: list, env: list):
        """حلقه‌ی انیمیشن تست صدا — به‌عنوان Task روی ایونت‌لوپ فلت.

        تیک‌ها روی ترد ایونت‌لوپ اجرا می‌شوند تا پچ‌ها واقعاً به صف‌ی
        ارسال کلاینت برسند؛ asyncio.sleep جای تایمر را می‌گیرد.
        """
        try:
            while getattr(self, "_testing", False):
                if not self._tick_once(vals, env):
                    return
                await asyncio.sleep(0.08)
        except Exception:
            pass  # صفحه/پنجره بسته شده — لوپ تمام

    def _start_timer(self, vals: list, env: list):
        """مسیر پشتیبان برای محیط بدون ایونت‌لوپ (تست‌های MockPage)."""
        import threading
        self._test_timer = threading.Timer(0.08, self._tick_timer, args=(vals, env))
        self._test_timer.daemon = True
        self._test_timer.start()

    def _tick_timer(self, vals: list, env: list):
        if self._tick_once(vals, env):
            self._start_timer(vals, env)

    # ================================================= درج متن
    def _tab_insert(self):
        cfg = self.cfg
        # RadioGroup کنترل‌شده است: بدون on_change، کلیک کاربر به مقدار
        # قبلی برمی‌گردد — پس مقدار جدید را همین‌جا ثبت می‌کنیم.
        self.var_paste = ft.RadioGroup(
            value=cfg.get("paste_method"),
            on_change=self._paste_changed,
            content=ft.Column([
                ft.Radio(value="clipboard", label="کلیپ‌بورد (پیشنهادی)",
                         active_color=t.ACCENT, fill_color=t.ACCENT,
                         label_style=t.fam("Regular", 13),
                         label_position=ft.LabelPosition.LEFT),
                ft.Radio(value="type", label="تایپ مستقیم (کندتر)",
                         active_color=t.ACCENT, fill_color=t.ACCENT,
                         label_style=t.fam("Regular", 13),
                         label_position=ft.LabelPosition.LEFT),
            ]),
        )
        c_method = t.card(
            "روش درج",
            t.dim("متن تشخیص‌داده‌شده چگونه در برنامه مقصد برسد؟"),
            self.var_paste,
        )

        self.var_restore = ft.Switch(value=bool(cfg.get("restore_clipboard")),
                                     active_color=t.ACCENT, scale=0.9)
        self.var_commands = ft.Switch(value=bool(cfg.get("voice_commands")),
                                      active_color=t.ACCENT, scale=0.9)
        self.var_itn = ft.Switch(value=bool(cfg.get("persian_itn")),
                                 active_color=t.ACCENT, scale=0.9)
        c_clip = t.card(
            "کلیپ‌بورد و فرمان‌ها",
            ft.Row([t.row_label("بازیابی محتوای قبلی کلیپ‌بورد بعد از درج"), self.var_restore]),
            t.dim("اگر غیرفعال شود، متن دیکته در کلیپ‌بورد می‌ماند"),
            ft.Row([t.row_label("فرمان‌های صوتی"), self.var_commands]),
            t.dim("نقطه، ویرگول، علامت سوال، گیومه باز/بسته، نقطه ویرگول، خط جدید، حذف آخرین کلمه"),
            ft.Row([t.row_label("تبدیل اعداد حروفی به رقم"), self.var_itn]),
            t.dim("اعداد حروفی خودکار به رقم تبدیل می‌شوند؛ اعداد تکی مثل «یک» حروفی می‌مانند"),
        )
        return ft.Column([c_method, c_clip], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def _paste_changed(self, e):
        """RadioGroup کنترل‌شده است — مقدار جدید در e.data است
        (e.control.value در لحظه‌ی رویداد هنوز مقدار قدیمی است)."""
        self.var_paste.value = e.data

    # ================================================= پیشرفته
    def _tab_advanced(self):
        cfg = self.cfg
        self.var_threads = t.dropdown([str(i) for i in range(1, 9)],
                                      str(int(cfg.get("num_threads") or 4)), width=80)
        c_proc = t.card(
            "پردازش",
            ft.Row([t.row_label("تعداد هسته پردازش مدل (با ری‌استارت اعمال می‌شود)"),
                    self.var_threads]),
        )

        self.var_hotword = ft.Switch(value=bool(cfg.get("hotword_boost")),
                                     active_color=t.ACCENT, scale=0.9)
        self.txt_hotwords = ft.TextField(
            multiline=True, min_lines=4, max_lines=6, expand=True,
            bgcolor=t.SURFACE_2, border_color=t.BORDER, border_radius=8,
            text_style=t.fam("Regular", 13), rtl=True,
            content_padding=ft.Padding(left=10, top=8, right=10, bottom=8),
        )
        self.txt_hotwords.value = "\n".join(str(w) for w in (cfg.get("hotwords") or []))
        # پیام خطای هات‌وورد — زیر کادر، جایی که کاربر هست (در CTk روی
        # برچسب هات‌کی در تب دیگر می‌رفت و دیده نمی‌شد)
        self.hw_hint = ft.Text("", style=t.fam("Regular", 12), color=t.DANGER,
                               text_align=ft.TextAlign.RIGHT)
        c_hw = t.card(
            "واژه‌های حساس (هات‌وورد) — آزمایشی",
            ft.Row([t.row_label("تقویت واژه‌های مشخص هنگام تشخیص"), self.var_hotword]),
            self.txt_hotwords,
            self.hw_hint,
            t.dim("هر خط یک واژه، حداقل ۲ حرف — اسم‌ها و برندهایی که مدل مدام اشتباه می‌گیرد"),
            t.dim("با روشن‌کردن، پردازش کمی کندتر می‌شود و ممکن است نشانه‌های پایانی جمله (مثل نقطه) هم درج شوند"),
        )

        self.var_stable = ft.Switch(value=bool(cfg.get("stable_live")),
                                    active_color=t.ACCENT, scale=0.9,
                                    on_change=self._sync_enroll_ui)
        c_stable = t.card(
            "متن زنده پایدار — آزمایشی",
            ft.Row([t.row_label("قفل‌کردن پیشوند با امتیاز اطمینان"), self.var_stable]),
            t.dim("واژه فقط وقتی قطعی می‌شود که در پنجره‌های پیاپی پایدار باشد، رقیب هم‌زمان نداشته باشد و از لبه خارج نشده باشد؛ نوسان نمایش کمتر می‌شود"),
        )

        # ---------- ثبت صوتی واژه (قرینه‌ی تب پیشرفته CTk) ----------
        self.var_enroll = ft.Switch(value=bool(cfg.get("enroll_alias")),
                                    active_color=t.ACCENT, scale=0.9,
                                    on_change=self._sync_enroll_ui)
        self.enroll_stable_note = ft.Text(
            "«متن زنده پایدار» روشن است — تا خاموشش نکنی این لایه روی خروجی زنده و نهایی اعمال نمی‌شود",
            style=t.fam("Regular", 12), color=t.WARN,
            text_align=ft.TextAlign.RIGHT, visible=bool(cfg.get("stable_live")))
        self.enroll_list = ft.Column(spacing=4)
        self.enroll_add_btn = ft.Button(
            content="+ ثبت واژه جدید", on_click=lambda e: self._open_enroll_editor(),
            width=140, height=34, bgcolor=t.SURFACE_2, color=t.FG,
            style=t.btn_style(weight="bold"),
        )
        self._enroll_store = _load_enroll_store()
        c_enroll = t.card(
            "ثبت صوتی واژه",
            ft.Row([t.row_label("اصلاح واژه‌های ثبت‌شده در خروجی"), self.var_enroll]),
            self.enroll_stable_note,
            t.dim("اثر هم روی متن زنده و هم روی متن نهایی دارد؛ در حالت «متن زنده پایدار» اعمال نمی‌شود"),
            self.enroll_list,
            ft.Row([self.enroll_add_btn], alignment=ft.MainAxisAlignment.END),
        )
        self._enroll_list_card = c_enroll
        self._rebuild_enroll_list()

        # ادیتور واژه — پنل جدای درون‌صفحه‌ای (فلت دیالوگ Toplevel ندارد)؛
        # موقع باز شدن، کارت فهرست مخفی می‌شود
        self._enroll_editor_card = t.card("ثبت واژه جدید", ft.Column(spacing=6))
        self._enroll_editor_card.visible = False

        c_info = t.card(
            "درباره موتور تشخیص",
            ft.Container(ft.Text("Shenava-Koochik v1.0", style=t.fam("bold", 13), color=t.FG,
                                 text_align=ft.TextAlign.RIGHT),
                         alignment=ft.Alignment(1, 0)),
            t.dim("کاملاً آفلاین — ۱۱۴ میلیون پارامتر (معماری FastConformer)"),
            t.dim("کیفیت روی جملات دیکته‌شده بهتر از مکالمه آزاد است"),
        )
        self._sync_enroll_ui()
        return ft.Column([c_proc, c_hw, c_stable, c_enroll,
                          self._enroll_editor_card, c_info],
                         spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    # ================= ثبت صوتی واژه (قرینه‌ی open_enroll_dialog در CTk) =================

    def _sync_enroll_ui(self, e=None):
        """در حالت پایدار، لایه‌ی ثبت واژه اعمال نمی‌شود — سوییچ و ضبط
        غیرفعال می‌شوند و نکته‌اش دیده می‌شود (قرینه‌ی _sync_enroll_ui در CTk)."""
        stable = bool(self.var_stable.value)
        self.var_enroll.disabled = stable
        off = stable or not bool(self.var_enroll.value)
        self.enroll_add_btn.disabled = off
        self.enroll_stable_note.visible = stable
        self._safe_update(self.enroll_stable_note)
        self._safe_update(self.var_enroll)

    def _rebuild_enroll_list(self):
        rows: list[ft.Control] = []
        if not self._enroll_store.entries:
            rows.append(t.dim("هنوز واژه‌ای ثبت نشده"))
        for e in self._enroll_store.entries:
            n_var = len(e.get("variants", []))
            word = str(e.get("word", ""))
            edit_btn = ft.Button(
                content="ویرایش", width=64, height=26,
                bgcolor=t.SURFACE_2, color=t.FG, style=t.btn_style(size=11),
                on_click=lambda ev, wd=word: self._open_enroll_editor(wd),
            )
            del_btn = ft.Button(
                content="حذف", width=56, height=26,
                bgcolor=t.SURFACE_2, color=t.DANGER, style=t.btn_style(size=11),
                on_click=lambda ev, wd=word: self._enroll_remove(wd),
            )
            rows.append(ft.Row([
                del_btn, edit_btn,
                ft.Text(f"«{word}» — {n_var} واریانت تأییدشده",
                        style=t.fam("Regular", 13), color=t.FG,
                        text_align=ft.TextAlign.RIGHT, expand=True),
            ]))
        self.enroll_list.controls = rows
        self._safe_update(self.enroll_list)

    def _enroll_remove(self, word: str):
        self._enroll_store.remove_entry(word)
        self._enroll_store.save()
        self._rebuild_enroll_list()

    def _open_enroll_editor(self, word: str | None = None):
        """word=None → ثبت واژه جدید؛ str → ویرایش همان واژه.

        فهرست مخفی و ادیتور نشان داده می‌شود — با ذخیره/انصراف برمی‌گردد.
        """
        entry = None
        if word is not None:
            entry = next((e for e in self._enroll_store.entries
                          if str(e.get("word", "")) == word), None)
        self._enroll_entry = entry
        self._enroll_editing = True
        self._enroll_busy = -1
        self._enroll_playing = -1
        self._enroll_heard = [str(v) for v in (entry or {}).get("variants", [])]
        self._enroll_checks: dict[str, bool] = {}
        self._enroll_slots = [{"samples": None, "rec": None, "t0": None,
                               "play_t0": None, "play_dur": 0.0} for _ in range(3)]
        self._enroll_result_q: queue.Queue = queue.Queue()
        self._enroll_engine_loaded = False

        self.en_word = ft.TextField(
            value=str((entry or {}).get("word", "")),
            height=38, border_radius=8, bgcolor=t.SURFACE_2,
            border_color=t.BORDER, text_style=t.fam("Regular", 14), rtl=True,
            content_padding=ft.Padding(left=10, top=8, right=10, bottom=8),
        )
        self.en_manual = ft.TextField(
            height=32, border_radius=6, bgcolor=t.SURFACE_2,
            border_color=t.BORDER, text_style=t.fam("Regular", 13), rtl=True,
            content_padding=ft.Padding(left=8, top=6, right=8, bottom=6),
        )
        self.en_checks = ft.Column(spacing=2)
        self.en_status = [ft.Text("—", style=t.fam("Regular", 12), color=t.FG_DIM,
                                  text_align=ft.TextAlign.RIGHT, expand=True)
                          for _ in range(3)]
        self.en_play = [ft.Button(content="پخش", width=60, height=30,
                                  bgcolor=t.SURFACE_2, color=t.FG,
                                  style=t.btn_style(size=11), disabled=True,
                                  on_click=lambda ev, i=i: self._enroll_play(i))
                        for i in range(3)]
        self.en_rec = [ft.Button(content=f"ضبط {'۱۲۳'[i]}", width=80, height=30,
                                 bgcolor=t.SURFACE_2, color=t.FG,
                                 style=t.btn_style(size=11, weight="bold"),
                                 on_click=lambda ev, i=i: self._enroll_toggle_rec(i))
                       for i in range(3)]
        slot_rows = [ft.Row([st, pb, rb])
                     for st, pb, rb in zip(self.en_status, self.en_play, self.en_rec)]

        btn_save = ft.Button(content="ذخیره واژه", width=130, height=36,
                             bgcolor=t.ACCENT, color=t.ON_ACCENT,
                             style=t.btn_style(weight="bold"),
                             on_click=self._enroll_save)
        btn_cancel = ft.Button(content="انصراف", width=100, height=36,
                               bgcolor=t.SURFACE_2, color=t.FG,
                               style=t.btn_style(),
                               on_click=lambda ev: self._close_enroll_editor())
        editor_body = self._enroll_editor_card.content
        editor_body.controls = [
            ft.Text("واژه‌ی درست — همان‌طور که باید نوشته شود:",
                    style=t.fam("Regular", 13), color=t.FG,
                    text_align=ft.TextAlign.RIGHT),
            self.en_word,
            ft.Text("سه بار واضح بگو — ضبط را شروع کن، بگو، و قطع کن؛ بعد با پخش گوش بده:",
                    style=t.fam("Regular", 13), color=t.FG,
                    text_align=ft.TextAlign.RIGHT),
            ft.Column(slot_rows, spacing=2),
            self.en_checks,
            ft.Row([
                self.en_manual,
                ft.Button(content="+ افزودن دستی", width=110, height=30,
                          bgcolor=t.SURFACE_2, color=t.FG, style=t.btn_style(size=11),
                          on_click=lambda ev: self._enroll_add_manual()),
            ], spacing=6),
            ft.Row([btn_cancel, ft.Container(expand=True), btn_save]),
        ]
        self._rebuild_enroll_checks()
        # کارت فهرست ← مخفی؛ ادیتور ← نمایان
        self._enroll_list_card.visible = False
        self._enroll_editor_card.visible = True
        self._safe_update(self._enroll_editor_card)
        self._start_enroll_poll()

    def _close_enroll_editor(self):
        self._enroll_editing = False
        self._enroll_stop_recording_quiet()
        self._enroll_editor_card.visible = False
        if getattr(self, "_enroll_list_card", None) is not None:
            self._enroll_list_card.visible = True
        self._rebuild_enroll_list()
        self._safe_update(self.enroll_list)

    def _enroll_stop_recording_quiet(self):
        for s in getattr(self, "_enroll_slots", []):
            rec, s["rec"] = s["rec"], None
            if rec is not None:
                try:
                    rec.stop()
                except Exception:
                    pass

    def _rebuild_enroll_checks(self):
        heard = self._enroll_heard
        for v in list(self._enroll_checks):
            if v not in heard:
                del self._enroll_checks[v]
        for v in heard:
            self._enroll_checks.setdefault(v, True)
        rows: list[ft.Control] = []
        if not heard:
            rows.append(t.dim("هنوز واریانتی نیست — ضبط کن یا دستی اضافه کن"))
        else:
            rows.append(t.dim("شکل‌های شنیده‌شده — هر کدام را تأیید می‌کنی در خروجی جای واژه‌ی درست می‌نشیند:"))
            for v in heard:
                cb = ft.Checkbox(label=f"«{v}»", value=self._enroll_checks[v],
                                 active_color=t.ACCENT, label_style=t.fam("Regular", 12),
                                 on_change=lambda ev, vv=v: self._enroll_checks.update({vv: bool(ev.control.value)}))
                rows.append(cb)
        self.en_checks.controls = rows
        self._safe_update(self.en_checks)

    def _enroll_add_manual(self):
        # افزودن دستی واریانت — شکل شنیده‌شده را که در متن زنده دیدی،
        # بدون ضبط مجدد همین‌جا تایپ کن؛ دقیق‌ترین منبع واریانت همان متن زنده است
        v = (self.en_manual.value or "").strip()
        if len(v) < 2:
            return
        if v not in self._enroll_heard:
            self._enroll_heard.append(v)
        self.en_manual.value = ""
        self._rebuild_enroll_checks()
        self._safe_update(self.en_manual)

    def _enroll_engine(self):
        """موتور دیکد واریانت‌ها — بار اول در همین پروسه لود می‌شود.

        تنظیمات پروسه‌ی جداست و به موتور اپ دسترسی ندارد؛ لود تنبل
        چند ثانیه طول می‌کشد و فقط برای همین نشست کش می‌شود.
        فقط از ترد کارگر صدا زده می‌شود — هیچ دسترسی UI اینجا ممنوع.
        """
        if not getattr(self, "_enroll_engine_loaded", False):
            from app.asr import load_engine
            from app.config import model_dir
            engine = load_engine(
                model_dir=model_dir(),
                num_threads=int((self.cfg or {}).get("num_threads") or 4))
            self._enroll_engine_obj = engine
            self._enroll_engine_loaded = True
        return self._enroll_engine_obj

    def _enroll_status(self, slot: int, text: str, color: str):
        self.en_status[slot].value = text
        self.en_status[slot].color = color
        self._safe_update(self.en_status[slot])

    def _enroll_device(self) -> int | None:
        """دستگاه ضبط ثبت واژه — کمبو → env بک‌اند → پیش‌فرض سیستم."""
        dev = self._selected_device()
        if dev is not None:
            return dev
        try:
            return int(os.environ.get("DIKTEYAR_DEVICE") or "")
        except ValueError:
            return None

    def _enroll_toggle_rec(self, slot: int):
        if self._enroll_busy == slot:
            self._enroll_stop_rec(slot)
            return
        if self._enroll_busy >= 0:
            return
        word = (self.en_word.value or "").strip()
        if not word:
            self._enroll_status(slot, "اول واژه‌ی درست را بنویس", t.DANGER)
            return
        try:
            from app.recorder import Recorder
            rec = Recorder(device=self._enroll_device(), block_ms=50)
            rec.start()  # همان مسیر ضبط دیکته — سریع و بی‌probe
        except Exception as e:
            self._enroll_status(slot, f"خطا: {str(e)[:50]}", t.DANGER)
            return
        self._enroll_busy = slot
        self._enroll_slots[slot]["rec"] = rec
        self._enroll_slots[slot]["t0"] = time.monotonic()
        self.en_rec[slot].content = "توقف"
        self.en_rec[slot].bgcolor = t.DANGER
        self.en_play[slot].disabled = True
        for j, b in enumerate(self.en_rec):
            if j != slot:
                b.disabled = True
        self._enroll_status(slot, "در حال ضبط…", t.WARN)

    def _enroll_stop_rec(self, slot: int):
        s = self._enroll_slots[slot]
        rec, s["rec"] = s["rec"], None
        s["t0"] = None
        self._enroll_busy = -1
        if rec is None:
            return
        rec.stop()
        data = s["samples"] = rec.get_buffer_16k()
        # بدون این، دکمه پخش برای همیشه disabled می‌ماند — ریشه‌ی
        # «پخش کار نمی‌کند»؛ حتی ضبط کوتاه برای تشخیص قابل پخش است
        self.en_play[slot].disabled = False
        self.en_rec[slot].content = f"ضبط {'۱۲۳'[slot]}"
        self.en_rec[slot].bgcolor = t.SURFACE_2
        for b in self.en_rec:
            b.disabled = False
        self._safe_update(self.en_rec[slot])
        self._safe_update(self.en_play[slot])
        if data.size < 0.3 * 16000:
            self._enroll_status(slot, "ضبط خیلی کوتاه بود — دوباره ضبط کن", t.WARN)
            return
        # word همین‌جا در ترد اصلی خوانده می‌شود — TextField.value از
        # ترد کارگر امن نیست
        word = (self.en_word.value or "").strip()
        if not getattr(self, "_enroll_engine_loaded", False):
            self._enroll_status(slot, "در حال بارگذاری موتور… (بار اول چند ثانیه)",
                                t.WARN)
        else:
            self._enroll_status(slot, "در حال پردازش…", t.WARN)
        threading.Thread(target=self._enroll_decode_worker,
                         args=(slot, data, word), daemon=True).start()

    def _enroll_decode_worker(self, slot: int, data, word: str):
        # هیچ دسترسی UI اینجا ممنوع — فقط صف؛ رندر با حلقه‌ی poll
        try:
            engine = self._enroll_engine()
            text = str(engine.transcribe(data, 16000) or "")
            from app import enroll as enroll_mod
            variants = enroll_mod.harvest_variants(
                engine, data, word, text=text)
            self._enroll_result_q.put(("done", slot, text, variants))
        except Exception as e:
            self._enroll_result_q.put(("err", slot, str(e)[:60], []))

    def _enroll_play(self, slot: int):
        import sounddevice as sd

        s = self._enroll_slots[slot]
        data = s["samples"]
        if data is None or not data.size:
            return
        if self._enroll_playing == slot:  # در حال پخش — قطع
            sd.stop()
            self._enroll_playing = -1
            self.en_play[slot].content = "پخش"
            self._safe_update(self.en_play[slot])
            return
        sd.stop()
        # ضبط میکروفون معمولاً خیلی کم‌صدا است — برای پخش به peak
        # نرمال می‌شود (سقف تقویت ×۳۰)
        out = data
        peak = float(np.abs(data).max())
        if 0.0 < peak < 0.15:
            out = np.clip(data * min(30.0, 0.5 / peak), -1.0, 1.0)
        try:
            sd.play(out, 16000)
        except Exception as e:
            self._enroll_status(slot, f"خطای پخش: {str(e)[:40]}", t.DANGER)
            return
        self._enroll_playing = slot
        s["play_t0"] = time.monotonic()
        s["play_dur"] = data.size / 16000.0
        self.en_play[slot].content = "قطع"
        self._safe_update(self.en_play[slot])

    def _start_enroll_poll(self):
        """حلقه‌ی poll ادیتور — روی ایونت‌لوپ فلت (قرینه‌ی dlg.after در CTk)."""
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            self._enroll_task = run_task(self._enroll_poll_loop)
        else:
            # MockPage (تست‌ها): بدون ایونت‌لوپ — تایمر
            self._enroll_poll_timer()

    async def _enroll_poll_loop(self):
        try:
            while self._enroll_editing:
                self._enroll_poll_tick()
                await asyncio.sleep(0.1)
        except Exception:
            pass  # صفحه/پنجره بسته شده

    def _enroll_poll_timer(self):
        if not self._enroll_editing:
            return
        self._enroll_poll_tick()
        import threading
        tm = threading.Timer(0.1, self._enroll_poll_timer)
        tm.daemon = True
        tm.start()

    def _enroll_poll_tick(self):
        try:
            while True:
                kind, slot, payload, variants = self._enroll_result_q.get_nowait()
                if kind == "err":
                    self._enroll_status(slot, f"خطا: {payload}", t.DANGER)
                else:
                    new = [v for v in variants if v not in self._enroll_heard]
                    self._enroll_heard.extend(new)
                    shown = payload.strip() or "چیزی شنیده نشد"
                    self._enroll_status(slot, f"شنیده شد: {shown}",
                                        t.ACCENT if new else t.WARN)
                    self._rebuild_enroll_checks()
        except queue.Empty:
            pass
        # سقف ایمنی ضبط — توقف خودکار پس از REC_MAX_SEC
        busy = self._enroll_busy
        if busy >= 0:
            s = self._enroll_slots[busy]
            if s["t0"] is not None and time.monotonic() - s["t0"] > REC_MAX_SEC:
                self._enroll_stop_rec(busy)
        # پایان طبیعی پخش — بدون این، فشار بعدی «قطع» می‌شد و صدا نمی‌داد
        pl = self._enroll_playing
        if pl >= 0:
            s = self._enroll_slots[pl]
            if s["play_t0"] is not None and \
                    time.monotonic() - s["play_t0"] > s["play_dur"] + 0.3:
                self._enroll_playing = -1
                self.en_play[pl].content = "پخش"
                self._safe_update(self.en_play[pl])

    def _enroll_save(self, e=None):
        word = (self.en_word.value or "").strip()
        if len(word) < 2:
            return
        checked = [v for v, ok in self._enroll_checks.items() if ok]
        from app import enroll as enroll_mod
        if self._enroll_entry is not None and \
                enroll_mod.norm_word(str(self._enroll_entry.get("word", ""))) != \
                enroll_mod.norm_word(word):
            # متن واژه عوض شده — مدخل با نام قبلی حذف شود
            self._enroll_store.remove_entry(str(self._enroll_entry.get("word", "")))
        self._enroll_store.add_entry(word, checked)
        self._enroll_store.save()
        self._close_enroll_editor()

    # ================================================= راهنما
    def _tab_help(self):
        _fa_ver = str(APP_VERSION).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
        ch = t.card(
            f"{APP_TITLE_FULL} — نسخه {_fa_ver}",
            t.dim("دیکته صوتی فارسی، کاملاً آفلاین — تجربه‌ای شبیه ویسپر فلو"),
            t.dim("مدل شنوا کوچیک، ۱۱۴ میلیون پارامتر — هیچ داده‌ای از سیستم شما خارج نمی‌شود"),
        )
        c1 = t.card(
            "استفاده سریع",
            t.dim("گام اول — در هر برنامه‌ای (نوت‌پد، تلگرام، مرورگر…) کلید میان‌بر را بزن"),
            t.dim("گام دوم — صحبت کن؛ پنجره زنده کنار موس متن را همزمان نشان می‌دهد"),
            t.dim("گام سوم — همان کلید را دوباره بزن تا متن در محل کرسر درج شود"),
        )
        c2 = t.card(
            "فرمان‌های صوتی",
            t.dim("بگو «نقطه» یا «ویرگول» یا «علامت سوال» تا نشانه درج شود"),
            t.dim("«گیومه باز» و «گیومه بسته» برای « »، «نقطه ویرگول» برای ؛"),
            t.dim("برای رفتن به خط بعد، «خط جدید» را بگو"),
            t.dim("«حذف آخرین کلمه» آخرین کلمه درج‌شده را پاک می‌کند"),
        )
        c3 = t.card(
            "نکته‌ها",
            t.dim("کلید میان‌بر فقط برای همین اپ مصرف می‌شود و به برنامه مقصد فرستاده نمی‌شود"),
            t.dim("اگر با میان‌بر برنامه دیگری تداخل داشت، از تب عمومی یک ترکیب تازه بگیر"),
            t.dim("در برنامه‌هایی که با دسترسی مدیر باز شده‌اند درج کار نمی‌کند؛ اپ را هم مدیر اجرا کن یا روش درج را عوض کن"),
            t.dim("اگر میکروفون را عوض کردی، از تب میکروفون دستگاه را انتخاب کن یا حالت خودکار را نگه دار"),
            t.dim("اعداد حروفی خودکار به رقم تبدیل می‌شوند؛ خاموش یا روشن‌کردنش از تب درج متن است"),
        )
        return ft.Column([ch, c1, c2, c3], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    # ================================================= ذخیره / بازنشانی / بستن
    def _collect(self) -> dict:
        """خواندن همه‌ی مقادیر از ویجت‌ها — قرینه‌ی save() در CTk."""
        auto_stop_labels = {"خاموش": 0, "۳ ثانیه": 3, "۵ ثانیه": 5, "۱۰ ثانیه": 10}
        hw_on = bool(self.var_hotword.value)
        d = {
            "hotkey": (self.var_hotkey.value or "").strip(),
            "paste_method": self.var_paste.value or "clipboard",
            "voice_commands": bool(self.var_commands.value),
            "persian_itn": bool(self.var_itn.value),
            "restore_clipboard": bool(self.var_restore.value),
            "sound_feedback": bool(self.var_sound.value),
            "overlay_enabled": bool(self.var_overlay.value),
            "autostart": bool(self.var_autostart.value),
            "num_threads": int(self.var_threads.value or 4),
            "overlay_font_size": int(self.var_font.value),
            "auto_stop_sec": auto_stop_labels.get(self.var_auto_stop.value, 0),
            "hotword_boost": hw_on,
            "stable_live": bool(self.var_stable.value),
            # از خود سوییچ — وگرنه تغییر کاربر بی‌اثر بود و مقدار دیسک
            # pass-through می‌شد (قرینه‌ی cfg.set("enroll_alias") در CTk)
            "enroll_alias": bool(self.var_enroll.value),
            "hotwords": [ln.strip() for ln in (self.txt_hotwords.value or "").splitlines()
                         if len(ln.strip()) >= 2],
        }
        # کلیدهایی که این UI ویرایش نمی‌کند (rejoin_prefixes، input_device_key)
        # — از تنظیمات فعلی عبور بدهد و جای خالی را با DEFAULTS پر کند تا
        # ذخیره، آن‌ها را ریست یا پاک نکند.
        for k, default in DEFAULTS.items():
            if k not in d:
                v = (self.cfg or {}).get(k, default)
                d[k] = default if v is None else v
        return d

    def _selected_device(self):
        """دستگاه انتخابی در کمبو — None یعنی تشخیص خودکار (قرینه‌ی CTk)."""
        v = self.var_device.value
        if v == self.auto_label or not v:
            return None
        for idx, lbl in getattr(self, "_devices", []):
            if lbl == v:
                return idx
        return None

    def _selected_device_key(self):
        """کلید پایدار انتخاب فعلی (نام — API) — برای بازیابی بعد از
        جابه‌جایی ایندکس‌ها بین بوت‌ها."""
        idx = self._selected_device()
        if idx is None:
            return None
        for i, lbl in self._devices:
            if i == idx:
                return lbl
        return None

    def _apply(self, data: dict):
        """بارگذاری مقادیر روی ویجت‌ها — قرینه‌ی _sync()."""
        self.var_hotkey.value = data.get("hotkey")
        self.var_paste.value = data.get("paste_method", "clipboard")
        self.var_commands.value = bool(data.get("voice_commands"))
        self.var_itn.value = bool(data.get("persian_itn"))
        self.var_stable.value = bool(data.get("stable_live"))
        self.var_enroll.value = bool(data.get("enroll_alias"))
        self.var_restore.value = bool(data.get("restore_clipboard"))
        self.var_sound.value = bool(data.get("sound_feedback"))
        self.var_overlay.value = bool(data.get("overlay_enabled"))
        self.var_autostart.value = bool(data.get("autostart"))
        self.var_threads.value = str(int(data.get("num_threads") or 4))
        self.var_font.value = int(data.get("overlay_font_size") or 15)
        self.font_lbl.value = f"اندازه متن: {int(self.var_font.value)}"
        self.sample.style = t.fam("Regular", int(self.var_font.value))
        self.var_auto_stop.value = {0: "خاموش", 3: "۳ ثانیه", 5: "۵ ثانیه", 10: "۱۰ ثانیه"} \
            .get(int(data.get("auto_stop_sec") or 0), "خاموش")
        self.var_hotword.value = bool(data.get("hotword_boost"))
        self.txt_hotwords.value = "\n".join(str(w) for w in (data.get("hotwords") or []))
        # سوییچ/یادداشت ثبت صوتی با حالت پایدارِ تازه همگام شود — وگرنه
        # پس از بازنشانی، حالت کهنه‌ی سوییچ و نکته‌ی هشدار می‌ماند
        self._sync_enroll_ui()
        # دستگاه ورودی — None یعنی خودکار؛ دستگاه ذخیره‌شده باید در کمبو
        # نمایش داده شود (قرینه‌ی current_name در CTk)
        dev = data.get("input_device")
        if dev is None:
            self.var_device.value = self.auto_label
        else:
            match = [lbl for i, lbl in getattr(self, "_devices", [])
                     if i == dev]
            self.var_device.value = match[0] if match else self.auto_label

    def _reset(self, e=None):
        self._end_test()
        self._apply(dict(DEFAULTS))
        self._set_hk_hint(HK_HINT, t.FG_DIM)
        if getattr(self, "hw_hint", None) is not None:
            self.hw_hint.value = ""
            self._safe_update(self.hw_hint)
        self.page.update()

    def _validate(self, data: dict):
        """اعتبارسنجی پیش از ذخیره — قرینه‌ی save() نسخه CTk.

        خروجی: (برچسب خطا، پیام) یا None یعنی مجاز. ذخیره‌ی تنظیمات
        خراب (هات‌کی بی‌اثر، هات‌وورد بی‌واژه) رد می‌شود.
        """
        if not _valid_hotkey(str(data.get("hotkey") or "").strip()):
            return ("hotkey", HK_INVALID)
        if data.get("hotword_boost") and not data.get("hotwords"):
            return ("hotwords", HW_EMPTY)
        return None

    async def _save(self, e=None):
        self._end_test()
        data = self._collect()
        data["input_device"] = self._selected_device()
        data["input_device_key"] = self._selected_device_key()
        err = self._validate(data)
        if err is not None:
            where, msg = err
            lbl = self.hk_hint if where == "hotkey" else self.hw_hint
            lbl.value = msg
            lbl.color = t.DANGER
            self._safe_update(lbl)
            return  # مثل CTk: پنجره باز می‌ماند و ذخیره نمی‌شود
        # ذخیره‌ی واقعی روی دیسک — مثل save() نسخه CTk
        loaded = Config.load()
        for k, v in data.items():
            loaded.set(k, v)
        loaded.save()
        self.result = dict(data)
        if self.on_save:
            self.on_save(self.result)
        await self._close()

    async def _close(self, e=None):
        # قطع ضبط تست و کپچر هات‌کی قبل از بستن — InputStream/hook
        # نباید بعد از پنجره زنده بمانند
        self._end_test()
        self._hk_blur()
        # Window.destroy در Flet 0.86 کوروتین است — اگر هندلر sync باشد
        # کوروتین هیچ‌وقت await نمی‌شود و پنجره باز می‌ماند.
        await self.page.window.destroy()

    def _end_test(self):
        """توقف کامل تست صدا از هر مسیر — دکمه، ذخیره، انصراف، بستن پنجره."""
        if getattr(self, "_testing", False):
            self._testing = False
            self.test_btn.content = "شروع تست"
            self._safe_update(self.test_btn)
        self._stop_tester()
        task = getattr(self, "_test_task", None)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
            self._test_task = None
        timer = getattr(self, "_test_timer", None)
        if timer is not None:
            timer.cancel()
            self._test_timer = None
        for bar in getattr(self, "_bars", []):
            bar.height = 3
            bar.bgcolor = t.SURFACE_3
            self._safe_update(bar)
