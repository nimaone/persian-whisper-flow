"""پنجره تنظیمات دیکته‌یار با Flet — قرینه‌ی app/settings_ui.py.

۵ تب: عمومی، میکروفون، درج متن، پیشرفته، راهنما.
همان کارت‌ها، همان متن‌ها، همان تم. تست صدا با ضبط واقعی RMS
(قرینه‌ی MicTester نسخه‌ی CTk) و انیمیشن روی ایونت‌لوپ فلت.
"""
from __future__ import annotations

import asyncio
import os

import flet as ft

from app.config import APP_TITLE_FULL, APP_VERSION, DEFAULTS, Config
from flet_ui import theme as t

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
        page.title = APP_TITLE_FULL + " — تنظیمات"
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

        c_dev = t.card("دستگاه ورودی", self.var_device)
        c_dev.content.controls.append(
            t.dim("خودکار = پرسیگنال‌ترین میکروفون فعال در شروع هر ضبط")
        )

        # ---------- تست صدا — ضبط واقعی RMS با MicTester (قرینه‌ی CTk) ----------
        # Stack با میله‌های left ثابت و کفِ ثابت (bottom=4 مثل خط کفی
        # SPECS_H+4 در CTk): میله از پایین به بالا رشد می‌کند و تغییر
        # height فقط خودِ میله را رندر می‌کند.
        self._bars: list[ft.Container] = []
        for i in range(48):
            self._bars.append(ft.Container(width=6, height=3,
                                           bgcolor=t.SURFACE_3,
                                           border_radius=2,
                                           left=(47 - i) * 9,
                                           bottom=4))
        self.spec_stack = ft.Stack(self._bars, width=435, height=28)
        self.verdict = ft.Text("", style=t.fam("bold", 13), text_align=ft.TextAlign.RIGHT)
        # نشانگر کیفیت ورودی — نویز پایه/اوج/SNR زنده حین تست
        self.quality_lbl = ft.Text("", style=t.fam("Regular", 12),
                                   color=t.FG_DIM, text_align=ft.TextAlign.RIGHT)
        self.test_btn = ft.Button(
            content="شروع تست", on_click=self._toggle_test, width=120, height=34,
            bgcolor=t.SURFACE_2, color=t.FG,
            style=t.btn_style(weight="bold"),
        )
        # ترتیب و چینش مثل CTk: موج → دکمه وسط‌چین → حکم. Rowها
        # تمام‌عرض می‌گیرند و کارت را مثل fill="x" پُر می‌کنند — وگرنه
        # Stackِ تنها عرض کارت را به اندازه‌ی خودش محدود می‌کند.
        c_test = t.card("تست صدا",
                        ft.Row([self.spec_stack]),
                        ft.Row([self.test_btn], alignment=ft.MainAxisAlignment.CENTER),
                        self.verdict,
                        self.quality_lbl)

        auto_stop_labels = ["خاموش", "۳ ثانیه", "۵ ثانیه", "۱۰ ثانیه"]
        self.var_auto_stop = t.dropdown(
            auto_stop_labels, auto_stop_labels[0], width=110,
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
    def _start_tester(self):
        """MicTester را روی دستگاه انتخابی روشن کن؛ زنجیره‌ی جایگزین
        (مسیرهای دیگر Host API همان میکروفون) اگر مسیر اصلی باز نشود.

        صف و رویدادِ توقف، محلیِ کلوژرِ ترد گرفته می‌شوند — نه attribute —
        تا تردِ تستر قبلی با Event خودش قطعاً تمام شود و به صف تازه نریزد.
        """
        import queue as _q
        import threading as _th

        self._stop_tester()   # هر تستر قبلی خاموش شود
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
        try:
            from app.recorder import current_input_devices, device_siblings
            inputs = getattr(self, "_all_inputs", []) or current_input_devices()
            self._test_fallbacks = [e["index"] for e in
                                    device_siblings(inputs, dev)]
        except Exception:
            self._test_fallbacks = []

        def run():
            import numpy as np
            try:
                import sounddevice as sd
                # زنجیره‌ی تست: مسیر اصلی → مسیرهای دیگر همان میکروفون —
                # شکست یک مسیر تست را کلاً نمی‌اندازد (قرینه‌ی ضبط)
                tried = [dev] + [e["index"] for e in
                                 getattr(self, "_test_fallbacks", [])]
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
        (asyncio.Queue) نمی‌رسد و هیچ‌وقت رندر نمی‌شود؛ میله‌ها در Stack
        با left ثابت‌اند و یک update() والد، دیف همه‌ی میله‌ها را می‌فرستد.
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
            self._safe_update(self.spec_stack)
            self._safe_update(self.verdict)

    def _tick_once(self, vals: list, env: list) -> bool:
        """یک تیک انیمیشن — میله‌ها و حکم را تازه می‌کند؛ False یعنی توقف."""
        if not getattr(self, "_testing", False):
            return False
        # داده‌ی جدید از میکروفون — حداکثر RMS از همه‌ی بلاک‌های صف
        got_err = None
        rms = 0.0
        while True:
            try:
                kind, v = self._tester_q.get_nowait()
            except Exception:
                break
            if kind == "err":
                got_err = v
            else:
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
        # نشانگر کیفیت ورودی — نویز پایه/اوج/SNR زنده حین تست
        if not hasattr(self, "_test_vals"):
            self._test_vals = []
            self._quality_tick = 0
        self._test_vals.append(rms)
        del self._test_vals[:-120]
        self._quality_tick += 1
        if self._quality_tick % 12 == 0:
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
        self._safe_update(self.spec_stack)
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
                                    active_color=t.ACCENT, scale=0.9)
        c_stable = t.card(
            "متن زنده پایدار — آزمایشی",
            ft.Row([t.row_label("قفل‌کردن پیشوند با امتیاز اطمینان"), self.var_stable]),
            t.dim("واژه فقط وقتی قطعی می‌شود که در پنجره‌های پیاپی پایدار باشد، رقیب هم‌زمان نداشته باشد و از لبه خارج نشده باشد؛ نوسان نمایش کمتر می‌شود"),
        )

        c_info = t.card(
            "درباره موتور تشخیص",
            ft.Container(ft.Text("Shenava-Koochik v1.0", style=t.fam("bold", 13), color=t.FG,
                                 text_align=ft.TextAlign.RIGHT),
                         alignment=ft.Alignment(1, 0)),
            t.dim("کاملاً آفلاین — ۱۱۴ میلیون پارامتر (معماری FastConformer)"),
            t.dim("کیفیت روی جملات دیکته‌شده بهتر از مکالمه آزاد است"),
        )
        return ft.Column([c_proc, c_hw, c_stable, c_info], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

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
            "hotwords": [ln.strip() for ln in (self.txt_hotwords.value or "").splitlines()
                         if len(ln.strip()) >= 2],
        }
        # کلیدهایی که این UI ویرایش نمی‌کند (rejoin_prefixes، enroll_alias،
        # stable_live، input_device_key) — از تنظیمات فعلی عبور بدهد و جای
        # خالی را با DEFAULTS پر کند تا ذخیره، آن‌ها را ریست یا پاک نکند.
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
