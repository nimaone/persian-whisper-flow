"""پنجره تنظیمات دیکته‌یار با Flet — قرینه‌ی app/settings_ui.py.

۵ تب: عمومی، میکروفون، درج متن، پیشرفته، راهنما.
همان کارت‌ها، همان متن‌ها، همان تم. حالت‌های زنده (تست میکروفون)
در این نسخه فقط نمایشی است؛ منطق اصلی با MicTester نسخه‌ی CTk حلق می‌شود.
"""
from __future__ import annotations

import flet as ft

from app.config import APP_TITLE_FULL, APP_VERSION, DEFAULTS
from flet_ui import theme as t


class SettingsWindow:
    """قرینه‌ی open_settings — هر متد عمومی با نسخه‌ی CTk هم‌نام است."""

    def __init__(self, page: ft.Page):
        self.page = page
        t.install_fonts(page)
        page.title = APP_TITLE_FULL + " — تنظیمات"
        page.bgcolor = t.BG
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 620
        page.window.height = 560
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
        # مثل CTk: ذخیره راست، بازنشانی چپ) ----------
        btn_bar = ft.Row(
            [
                ft.ElevatedButton(
                    content="ذخیره", on_click=self._save, width=130, height=40,
                    bgcolor=t.ACCENT, color=t.ON_ACCENT,
                    style=t.btn_style(weight="bold"),
                ),
                ft.Container(expand=True),
                ft.ElevatedButton(
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

    # ---------- داده‌ی config (برای دمو از DEFAULTS استفاده می‌کند) ----------
    def _load(self) -> dict:
        try:
            from app.config import Config
            return dict(Config.load())
        except Exception:
            return dict(DEFAULTS)

    # ================================================= عمومی
    def _tab_general(self):
        cfg = self.cfg
        self.var_hotkey = ft.TextField(
            value=cfg.get("hotkey"), read_only=False, expand=True,
            bgcolor=t.SURFACE_2, border_color=t.BORDER, border_radius=8,
            text_style=t.fam("Regular", 14), height=40, rtl=True,
        )
        self.hk_hint = ft.Text(
            "برای ثبت میان‌بر جدید، روی کادر کلیک کن و ترکیب دلخواه را بفشار (لغو: Esc)",
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

    # ================================================= میکروفون
    def _tab_mic(self):
        cfg = self.cfg
        try:
            import sounddevice as sd
            devices = [f"[{i}] {d['name']}"
                       for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception:
            devices = ["خودکار (پرسیگنال‌ترین)"]
        self.auto_label = "خودکار (پرسیگنال‌ترین)"
        values = [self.auto_label] + devices
        self.var_device = t.dropdown(values, values[0], height=36)

        c_dev = t.card("دستگاه ورودی", self.var_device)
        c_dev.content.controls.append(
            t.dim("خودکار = پرسیگنال‌ترین میکروفون فعال در شروع هر ضبط")
        )

        # ---------- تست صدا (نوارهای نمایشی — در نسخه‌ی CTk زنده است) ----------
        self.spec_row = ft.Row([], spacing=3, height=28,
                               alignment=ft.MainAxisAlignment.END)
        self.verdict = ft.Text("", style=t.fam("bold", 13), text_align=ft.TextAlign.RIGHT)
        self.test_btn = ft.ElevatedButton(
            content="شروع تست", on_click=self._toggle_test, width=120, height=34,
            bgcolor=t.SURFACE_2, color=t.FG,
            style=t.btn_style(weight="bold"),
        )
        c_test = t.card("تست صدا", self.spec_row, self.verdict, self.test_btn)

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

    def _toggle_test(self, e):
        """دموی نمایشی موج — در نسخه‌ی CTk به MicTester وصل است.

        Flet تایمر GUI ندارد؛ از threading.Timer استفاده می‌کنیم و آپدیت
        کنترل‌ها (thread-safe در Flet) از ترد پس‌زمینه انجام می‌شود.
        """
        import math, random, threading
        from time import monotonic as _mono
        self._testing = not getattr(self, "_testing", False)
        self.test_btn.content = "توقف تست" if self._testing else "شروع تست"
        if self._testing:
            bars = [0.0] * 48
            t0 = _mono()

            def tick():
                if not self._testing:
                    return
                bars[:-1] = bars[1:]
                bars[-1] = min(1.0, abs(random.random() * (1 + math.sin(_mono() - t0))))
                self.spec_row.controls = [
                    ft.Container(width=6, height=max(3, v * 28),
                                 bgcolor=t.ACCENT if v > 0.5 else
                                        ("#4f8f68" if v > 0.15 else t.SURFACE_3),
                                 border_radius=2, alignment=ft.Alignment(0, 1))
                    for i, v in enumerate(reversed(bars))
                ]
                self.verdict.value = "میکروفون کار می‌کند — صدای واضح"
                self.verdict.color = t.ACCENT
                try:
                    self.page.update()
                except Exception:
                    return  # پنجره بسته شده
                threading.Timer(0.08, tick).start()

            tick()
        else:
            self.spec_row.controls = []
            self.verdict.value = ""
            self.page.update()

    # ================================================= درج متن
    def _tab_insert(self):
        cfg = self.cfg
        self.var_paste = ft.RadioGroup(
            value=cfg.get("paste_method"),
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
        c_hw = t.card(
            "واژه‌های حساس (هات‌وورد) — آزمایشی",
            ft.Row([t.row_label("تقویت واژه‌های مشخص هنگام تشخیص"), self.var_hotword]),
            self.txt_hotwords,
            t.dim("هر خط یک واژه، حداقل ۲ حرف — اسم‌ها و برندهایی که مدل مدام اشتباه می‌گیرد"),
            t.dim("با روشن‌کردن، پردازش کمی کندتر می‌شود و ممکن است نشانه‌های پایانی جمله (مثل نقطه) هم درج شوند"),
        )

        c_info = t.card(
            "درباره موتور تشخیص",
            ft.Container(ft.Text("Shenava-Koochik v1.0", style=t.fam("bold", 13), color=t.FG,
                                 text_align=ft.TextAlign.RIGHT),
                         alignment=ft.Alignment(1, 0)),
            t.dim("کاملاً آفلاین — ۱۱۴ میلیون پارامتر (معماری FastConformer)"),
            t.dim("کیفیت روی جملات دیکته‌شده بهتر از مکالمه آزاد است"),
        )
        return ft.Column([c_proc, c_hw, c_info], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

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
        return {
            "hotkey": self.var_hotkey.value.strip(),
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
            "hotwords": [ln.strip() for ln in (self.txt_hotwords.value or "").splitlines()
                         if len(ln.strip()) >= 2],
        }

    def _apply(self, data: dict):
        """بارگذاری مقادیر روی ویجت‌ها — قرینه‌ی _sync()."""
        self.var_hotkey.value = data.get("hotkey")
        self.var_paste.value = data.get("paste_method", "clipboard")
        self.var_commands.value = bool(data.get("voice_commands"))
        self.var_itn.value = bool(data.get("persian_itn"))
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

    def _reset(self, e=None):
        self._apply(dict(DEFAULTS))
        self.page.update()

    def _save(self, e=None):
        self.result = self._collect()
        if self.on_save:
            self.on_save(self.result)
        self._close()

    def _close(self, e=None):
        self.page.window.destroy()

    def show(self):
        """نمایش به‌عنوان دیالوگ — تا بسته شود صبر می‌کند."""
        self.page.window.prevent_close = True
        self.page.window.on_dismiss = self._close
        self.page.show_dialog = None  # placeholder — Flet window در نسخه‌ی کامل
