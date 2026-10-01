"""تست‌های خودکار flet_ui — لایه‌ی منطق (بدون GUI).

اجرا:
    .venv/Scripts/python.exe -m pytest tests/test_flet_ui.py -v
یا بدون pytest:
    .venv/Scripts/python.exe -m tests.test_flet_ui

ساختار: هر پنجره با یک Page شبیه‌سازی‌شده (MockPage) ساخته می‌شود؛
تمام فراخوانی‌های page.update/destroy ضبط می‌شوند. مقادیر config هم
به DEFAULTS قفل می‌شود تا تست قطعی (deterministic) باشد.

تست GUI جعبه‌سیاه (کلیک واقعی) جداگانه و دستی انجام می‌شود — این
فایل پوشش منطقِ حالت‌ها/داده/دکمه‌هاست.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flet as ft

from app.config import APP_TITLE, APP_VERSION, DEFAULTS


# ---------- ابزار شبیه‌سازی ----------

class MockWindow:
    def __init__(self):
        self.width = self.height = None
        self.resizable = None
        self.icon = None
        self.destroyed = False

    async def destroy(self):
        # قرینه‌ی Window.destroy در Flet 0.86 که کوروتین است
        self.destroyed = True


class MockPage:
    """حداقلِ چیزی که پنجره‌های flet_ui از Page می‌خواهند."""

    def __init__(self):
        self.window = MockWindow()
        self.fonts = {}
        self.updates = 0
        self.added = []
        self.title = None
        self.bgcolor = None
        self.theme_mode = None
        self.padding = None
        self.rtl = None

    def update(self):
        self.updates += 1

    def add(self, *controls):
        self.added.extend(controls)


def _walk(control):
    """پیمایش درخت کنترل‌ها — فرزندانِ list/tuple/content/controls."""
    yield control
    for attr in ("content", "controls", "tabs", "options"):
        child = getattr(control, attr, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for c in child:
                yield from _walk(c)
        else:
            yield from _walk(child)


def _find_all(root, cls):
    return [c for c in _walk(root) if isinstance(c, cls)]


def _btn_texts(root):
    return [b.content for b in _find_all(root, ft.Button)
            if isinstance(b.content, str)]


# ---------- پنجره کنترل ----------

class TestControlWindow(unittest.TestCase):
    def setUp(self):
        from flet_ui.control_window import ControlWindow
        self.page = MockPage()
        self.win = ControlWindow(self.page)
        self.btn = self.win.rec_btn

    def test_window_setup(self):
        self.assertEqual(self.page.title, APP_TITLE)
        self.assertEqual(self.page.bgcolor, "#1e1e1e")
        self.assertTrue(self.page.rtl)
        self.assertEqual((self.page.window.width, self.page.window.height), (360, 250))
        self.assertIsNotNone(self.page.window.icon, "آیکون لوگو باید ست شود")
        self.assertTrue(self.page.window.icon.endswith(".ico"))

    def test_initial_state_is_loading(self):
        self.assertTrue(self.btn.disabled)
        self.assertEqual(self.btn.content, "شروع ضبط")
        self.assertEqual(self.btn.bgcolor, "#333333")          # SURFACE_2
        self.assertEqual(self.win.state_text.value, "در حال بارگذاری")
        self.assertIn("بارگذاری", self.win.status_text.value)

    def test_idle_state_enables_green(self):
        self.win.set_state("idle", "ctrl+shift+space")
        self.assertFalse(self.btn.disabled)
        self.assertEqual(self.btn.bgcolor, "#22c55e")          # ACCENT
        self.assertEqual(self.btn.color, "#081409")            # ON_ACCENT
        self.assertEqual(self.win.state_text.value, "آماده")
        self.assertEqual(self.win.state_text.color, "#22c55e")
        self.assertEqual(self.win.status_text.value, "کلید میانبر: ctrl+shift+space")
        self.assertIn("صحبت کن", self.win.hint_text.value)

    def test_recording_state_red(self):
        self.win.set_state("recording", "ctrl+shift+space")
        self.assertFalse(self.btn.disabled)
        self.assertEqual(self.btn.bgcolor, "#e5484d")          # DANGER
        self.assertEqual(self.btn.color, "#ffffff")            # ON_DANGER
        self.assertEqual(self.btn.content, "توقف و درج متن")
        self.assertEqual(self.win.state_text.color, "#e5484d")
        self.assertEqual(self.win.status_text.value, "در حال شنیدن…")

    def test_transcribing_state_disabled(self):
        self.win.set_state("transcribing", "ctrl+shift+space")
        self.assertTrue(self.btn.disabled)
        self.assertEqual(self.btn.content, "در حال تشخیص")
        self.assertEqual(self.win.status_text.value, "متن را می‌نویسد…")

    def test_state_cycle_returns_to_idle_cleanly(self):
        for st in ("idle", "recording", "transcribing", "idle"):
            self.win.set_state(st, "f9")
        self.assertFalse(self.btn.disabled)
        self.assertEqual(self.btn.bgcolor, "#22c55e")
        self.assertEqual(self.win.state_text.value, "آماده")

    def test_set_error(self):
        self.win.set_error("دستگاه پیدا نشد")
        self.assertTrue(self.win.status_text.value.startswith("خطا:"))
        self.assertEqual(self.win.status_text.color, "#e5484d")

    def test_callbacks_fire(self):
        fired = {"toggle": 0, "settings": 0}
        self.win.on_toggle = lambda: fired.__setitem__("toggle", fired["toggle"] + 1)
        self.win.on_settings = lambda: fired.__setitem__("settings", fired["settings"] + 1)
        self.win._fire_toggle()
        self.win._fire_settings()
        self.assertEqual(fired, {"toggle": 1, "settings": 1})

    def test_callbacks_absent_no_crash(self):
        self.win.on_toggle = None
        self.win.on_settings = None
        self.win._fire_toggle()      # نباید exception بدهد
        self.win._fire_settings()

    def test_button_bar_layout(self):
        texts = [t for c in self.page.added for t in _btn_texts(c)]
        self.assertEqual(texts, ["شروع ضبط", "تنظیمات"])

    def test_version_label(self):
        texts = [c.value for c in _walk_all(self.page.added)
                 if isinstance(c, ft.Text) and c.value and c.value.startswith("v")]
        self.assertIn(f"v{APP_VERSION}", texts)

    def test_updates_from_foreign_thread_go_through_event_loop(self):
        # باگ رگرسیون: page.update از ترد فرعی (هات‌کی/worker ضبط) به
        # صف ارسال کلاینت نمی‌رسد — ارسال باید با page.run_task روی
        # ایونت‌لوپ برود (داکیومنت رسمی فلت)
        import asyncio
        from flet_ui.control_window import ControlWindow

        class RunTaskPage(MockPage):
            def __init__(self):
                super().__init__()
                self.scheduled = []

            def run_task(self, handler, *args):
                self.scheduled.append((handler, args))

        page = RunTaskPage()
        win = ControlWindow(page)
        win.set_state("idle", "ctrl+k")
        self.assertEqual(len(page.scheduled), 1, "ارسال باید با run_task برنامه‌ریزی شود")
        self.assertEqual(page.updates, 0, "از ترد فرعی نباید مستقیم update شود")
        handler, args = page.scheduled[0]
        asyncio.run(handler(*args))          # شبیه‌سازی اجرا روی ایونت‌لوپ
        self.assertEqual(page.updates, 1)
        self.assertEqual(win.rec_btn.content, "شروع ضبط")
        self.assertFalse(win.rec_btn.disabled)
        self.assertEqual(win.status_text.value, "کلید میانبر: ctrl+k")
        # set_status هم همان مسیر
        win.set_status("سلام")
        self.assertEqual(len(page.scheduled), 2)
        handler, args = page.scheduled[1]
        asyncio.run(handler(*args))
        self.assertEqual(win.status_text.value, "سلام")


def _walk_all(controls):
    for c in controls:
        yield from _walk(c)


# ---------- پنجره تنظیمات ----------

class SettingsTestBase(unittest.TestCase):
    """پایه: SettingsWindow با config قطعی (DEFAULTS)."""

    def setUp(self):
        from flet_ui.settings_window import SettingsWindow
        page = MockPage()
        self.page = page

        class Fixed(SettingsWindow):
            def _load(self):
                return dict(DEFAULTS)

        self.win = Fixed(page)
        # اولین add — ستون اصلی شامل tabs و نوار دکمه‌ها
        self.root = self.page.added[0]


class TestSettingsWindow(SettingsTestBase):
    def test_window_setup(self):
        # مثل نسخه CTk: نام کوتاه — بدون «(ویسپر فلوی فارسی)»
        self.assertEqual(self.page.title, f"تنظیمات — {APP_TITLE}")
        self.assertEqual((self.page.window.width, self.page.window.height), (560, 640))
        self.assertTrue(self.page.rtl)
        self.assertIsNotNone(self.page.window.icon)

    def test_five_tabs(self):
        self.assertEqual(self.win.tabs.length, 5)
        labels = [tb.label for tb in self.win.tab_bar.tabs]
        self.assertEqual(labels, ["عمومی", "میکروفون", "درج متن", "پیشرفته", "راهنما"])

    def test_button_bar_has_three_buttons(self):
        # فقط ردیفِ نوار دکمه‌ها — نه دکمه‌های داخل تب‌ها (مثل «شروع تست»)
        bar_rows = [r for r in _find_all(self.root, ft.Row)
                    if _btn_texts(r) and len(_btn_texts(r)) >= 3]
        texts = _btn_texts(bar_rows[-1]) if bar_rows else []
        self.assertEqual(texts, ["ذخیره", "انصراف", "بازنشانی"])

    def test_collect_defaults(self):
        got = self.win._collect()
        for key in ("hotkey", "paste_method", "overlay_font_size", "auto_stop_sec",
                    "num_threads"):
            self.assertEqual(got[key], DEFAULTS[key], key)
        # انواع
        self.assertIsInstance(got["overlay_font_size"], int)
        self.assertIsInstance(got["num_threads"], int)
        self.assertIsInstance(got["hotwords"], list)

    def test_apply_roundtrip(self):
        custom = dict(DEFAULTS)
        custom.update(hotkey="f9", paste_method="type", overlay_enabled=False,
                      voice_commands=True, persian_itn=False, restore_clipboard=False,
                      sound_feedback=True, autostart=False, num_threads=6,
                      overlay_font_size=19, auto_stop_sec=5, hotword_boost=True,
                      hotwords=["نیما", "دیکته‌یار"])
        self.win._apply(custom)
        got = self.win._collect()
        # فقط کلیدهای قابل‌ویرایش در UI — input_device کمبو دارد ولی در
        # _collect عمومی نیست (دمو)
        editable = [k for k in custom if k != "input_device"]
        for key in editable:
            self.assertEqual(got[key], custom[key], key)

    def test_reset_restores_defaults(self):
        # مقادیر را به‌هم بریز
        self.win.var_overlay.value = False
        self.win.var_sound.value = True
        self.win.var_font.value = 20
        self.win.var_threads.value = "8"
        self.win.txt_hotwords.value = "واژه ساختگی تست"
        self.win._reset(None)
        got = self.win._collect()
        self.assertEqual(got["overlay_enabled"], DEFAULTS["overlay_enabled"])
        self.assertEqual(got["sound_feedback"], DEFAULTS["sound_feedback"])
        self.assertEqual(got["overlay_font_size"], DEFAULTS["overlay_font_size"])
        self.assertEqual(got["num_threads"], DEFAULTS["num_threads"])
        self.assertEqual(got["hotwords"], DEFAULTS["hotwords"])
        self.assertGreater(self.page.updates, 0, "بعد از reset باید update بشود")

    def test_font_slider_updates_label_and_sample(self):
        e = SimpleNamespace(control=SimpleNamespace(value=18))
        self.win._font_slide(e)
        self.assertEqual(self.win.font_lbl.value, "اندازه متن: 18")
        self.assertEqual(self.win.sample.style.size, 18)

    def test_save_records_result_and_closes(self):
        seen = []
        self.win.on_save = lambda data: seen.append(data)
        self.win.var_sound.value = True
        asyncio.run(self.win._save(None))
        self.assertEqual(len(seen), 1, "on_save باید یک‌بار صدا زده شود")
        self.assertTrue(seen[0]["sound_feedback"])
        self.assertEqual(self.win.result, seen[0])
        self.assertTrue(self.page.window.destroyed, "ذخیره باید پنجره را ببندد")

    def test_cancel_closes_without_saving(self):
        seen = []
        self.win.on_save = lambda data: seen.append(data)
        self.win.var_sound.value = True          # تغییر بی‌اهمیت — نباید ذخیره شود
        asyncio.run(self.win._close(None))
        self.assertEqual(seen, [], "انصراف نباید on_save صدا بزند")
        self.assertIsNone(self.win.result, "انصراف نباید result بنویسد")
        self.assertTrue(self.page.window.destroyed)

    def test_general_tab_controls_exist(self):
        self.assertIsInstance(self.win.var_hotkey, ft.TextField)
        self.assertEqual(self.win.var_hotkey.value, DEFAULTS["hotkey"])
        self.assertIsInstance(self.win.var_overlay, ft.Switch)
        self.assertIsInstance(self.win.var_font, ft.Slider)
        self.assertIsInstance(self.win.var_autostart, ft.Switch)

    def test_insert_tab_radio_group(self):
        self.assertIsInstance(self.win.var_paste, ft.RadioGroup)
        self.assertEqual(self.win.var_paste.value, DEFAULTS["paste_method"])
        for name in ("var_restore", "var_commands", "var_itn"):
            self.assertIsInstance(getattr(self.win, name), ft.Switch, name)

    def test_advanced_tab_controls_exist(self):
        self.assertIn(self.win.var_threads.value, [str(i) for i in range(1, 9)])
        self.assertIsInstance(self.win.var_hotword, ft.Switch)
        self.assertIsInstance(self.win.txt_hotwords, ft.TextField)

    def test_mic_tab_verdict_initially_empty(self):
        self.assertEqual(self.win.verdict.value, "")
        self.assertEqual(self.win.test_btn.content, "شروع تست")

    def test_test_button_toggles_label(self):
        # تستر واقعی را فیک می‌کنیم — بدون میکروفون، فقط چرخه‌ی دکمه
        import queue as _q
        self.win._tester_q = _q.Queue()
        import threading as _th
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None     # فیک: بدون باز کردن InputStream
        self.win._toggle_test(None)     # شروع
        self.assertEqual(self.win.test_btn.content, "توقف تست")
        self.win._end_test()           # توقف تمیز — مثل کلیک دکمه توقف
        self.assertEqual(self.win.test_btn.content, "شروع تست")
        self.assertFalse(getattr(self.win, "_testing", False))

    def test_reset_restores_device_dropdown(self):
        # باگ رگرسیون: بازنشانی دستگاه ورودی را به خودکار برنمی‌گرداند
        self.win.var_device.value = "Microphone Array (Realtek) — Windows WASAPI"
        self.win._reset(None)
        self.assertEqual(self.win.var_device.value, self.win.auto_label)

    def test_radio_group_change_handler(self):
        # باگ رگرسیون: RadioGroup کنترل‌شده است — بدون on_change کلیک کاربر
        # به مقدار قبلی برمی‌گردد
        self.assertEqual(self.win.var_paste.on_change, self.win._paste_changed)
        e = SimpleNamespace(control=SimpleNamespace(value="clipboard"), data="type")
        self.win._paste_changed(e)
        self.assertEqual(self.win.var_paste.value, "type")

    def test_device_dropdown_starts_at_auto(self):
        # باگ رگرسیون: دراپ‌داون میکروفون باید از روی config مقدار بگیرد؛
        # None یعنی «خودکار» — انتخاب دستگاه نباید باعث از دست رفتن ذخیره شود
        self.assertEqual(self.win.var_device.value, self.win.auto_label)

    def test_selected_device_parses_from_value(self):
        # بعد از انتخاب آیتم (فلت خودش value را ست می‌کند)، _selected_device
        # باید اندیس دستگاه را از برچسب دربیاورد — برچسب‌ها بدون ایندکس
        # خام‌اند (ناپایدار بین بوت‌ها) و کلید پایدار «نام — API» است
        devices = getattr(self.win, "_devices", [])
        if not devices:
            self.skipTest("دستگاه ورودی در این محیط یافت نشد")
        idx, label = devices[0]
        self.win.var_device.value = label
        self.assertEqual(self.win._selected_device(), idx)
        self.assertEqual(self.win._selected_device_key(), label)
        # مسیر ذخیره: _collect + input_device مثل _save
        data = self.win._collect()
        data["input_device"] = self.win._selected_device()
        self.assertEqual(data["input_device"], idx)

    def test_apply_shows_saved_device_in_dropdown(self):
        # باگ رگرسیون: _apply فقط حالت None را هندل می‌کرد؛ دستگاه ذخیره‌شده
        # باید در کمبو نمایش داده شود وگرنه کاربر فکر می‌کند ذخیره نشده
        devices = getattr(self.win, "_devices", [])
        if not devices:
            self.skipTest("دستگاه ورودی در این محیط یافت نشد")
        idx, label = devices[0]
        data = dict(DEFAULTS)
        data["input_device"] = idx
        self.win._apply(data)
        self.assertEqual(self.win.var_device.value, label)
        # و بازگشت به خودکار
        self.win._apply(dict(DEFAULTS))
        self.assertEqual(self.win.var_device.value, self.win.auto_label)

    def test_load_returns_dict_from_disk(self):
        # باگ رگرسیون: _load قبلاً dict(Config.load()) می‌زد که TypeError
        # می‌داد (Config شیء است نه dict) و except بی‌صدا DEFAULTS برمی‌گرداند
        # → دستگاه ذخیره‌شده همیشه «خودکار» دیده می‌شد.
        # _load واقعی باید dict برگرداند و کلیدهای دیسک را داشته باشد.
        from flet_ui.settings_window import SettingsWindow
        d = SettingsWindow._load(object.__new__(SettingsWindow))
        self.assertIsInstance(d, dict)
        for key in DEFAULTS:
            self.assertIn(key, d, key)

    # ---------- ثبت صوتی واژه ----------

    def _isolated_enroll_store(self):
        """فروشگاه ثبت واژه روی فایل موقت — بدون دست‌زدن به داده‌ی واقعی."""
        import tempfile
        from pathlib import Path as _P
        from unittest import mock
        from app import enroll as _enroll_mod
        fake = _P(tempfile.gettempdir()) / "dikteyar-tests-enroll-ui.json"
        fake.unlink(missing_ok=True)
        cm = mock.patch.object(_enroll_mod, "store_path", lambda: fake)
        return cm, _enroll_mod, fake

    def test_enroll_editor_open_add_manual_save(self):
        # ادیتور مدال (AlertDialog): باز شدن مدال تازه میسازد؛ واریانت دستی
        # به فهرست تأییدها میآید؛ ذخیره، مدخل را روی دیسک مینویسد و میبندد
        cm, enroll_mod, fake = self._isolated_enroll_store()
        with cm:
            self.win._enroll_store = enroll_mod.EnrollStore.load()
            self.win._open_enroll_editor()
            self.assertTrue(self.win._enroll_editing)
            self.assertIsNotNone(self.win._enroll_dialog)
            self.assertTrue(self.win._enroll_dialog.open)   # MockPage: فقط حالت پایتون
            self.win.en_word.value = "دیکته‌یار"
            self.win.en_manual.value = "ویسپر فارسی"
            self.win._enroll_add_manual()
            self.assertIn("ویسپر فارسی", self.win._enroll_heard)
            self.assertTrue(self.win._enroll_checks["ویسپر فارسی"])
            self.win._enroll_save()
            self.assertFalse(self.win._enroll_editing)
            self.assertIsNone(self.win._enroll_dialog)
            saved = enroll_mod.EnrollStore.load()
            self.assertEqual([e["word"] for e in saved.entries], ["دیکته‌یار"])
            self.assertIn("ویسپر فارسی", saved.entries[0]["variants"])
            fake.unlink(missing_ok=True)

    def test_enroll_editor_reopen_cycle(self):
        # رگرسیون «ذخیره/انصراف یکبار در میان کار میکرد»: هر باز/بسته باید
        # حالت پایتون را کامل جابهجا کند — مدال تازه، بدون ماندهی وضعیت
        cm, enroll_mod, fake = self._isolated_enroll_store()
        with cm:
            self.win._enroll_store = enroll_mod.EnrollStore.load()
            for round_no in range(3):
                self.win._open_enroll_editor()
                self.assertTrue(self.win._enroll_editing, round_no)
                self.assertIsNotNone(self.win._enroll_dialog, round_no)
                self.assertTrue(self.win._enroll_dialog.open, round_no)
                # انصراف — سپس فراخوانی دوباره (on_dismiss بعد از pop میآید)
                self.win._close_enroll_editor()
                self.win._close_enroll_editor()   # idempotent
                self.assertFalse(self.win._enroll_editing, round_no)
                self.assertIsNone(self.win._enroll_dialog, round_no)
                # ذخیره هم باید مدال را ببندد
                self.win._open_enroll_editor()
                self.win.en_word.value = f"واژه {round_no}"
                self.win._enroll_save()
                self.assertIsNone(self.win._enroll_dialog, round_no)
            saved = enroll_mod.EnrollStore.load()
            words = [e["word"] for e in saved.entries]
            self.assertEqual(words, ["واژه 0", "واژه 1", "واژه 2"])
            fake.unlink(missing_ok=True)

    def test_enroll_save_empty_word_shows_hint(self):
        # واژهی کوتاه → ذخیره بیاثر میماند ولی سرنخ قرمز میدهد —
        # CTk بیصدا برمیگشت و «دکمه کار نکرد» به نظر میآمد
        cm, enroll_mod, fake = self._isolated_enroll_store()
        with cm:
            self.win._enroll_store = enroll_mod.EnrollStore.load()
            self.win._open_enroll_editor()
            self.win.en_word.value = "ا"
            self.win._enroll_save()
            self.assertTrue(self.win.en_save_hint.visible)
            self.assertTrue(self.win._enroll_editing)      # مدال باز مانده
            self.assertFalse(enroll_mod.EnrollStore.load().entries)
            self.win.en_word.value = "دو حرف"
            self.win._enroll_save()
            self.assertFalse(self.win._enroll_editing)
            fake.unlink(missing_ok=True)

    def test_enroll_ui_disabled_in_stable_mode(self):
        # در حالت متن زنده پایدار، لایه‌ی ثبت واژه اعمال نمی‌شود —
        # سوییچ غیرفعال و نکته‌ی هشدار دیده می‌شود (قرینه‌ی CTk)
        self.win.var_stable.value = True
        self.win._sync_enroll_ui()
        self.assertTrue(self.win.var_enroll.disabled)
        self.assertTrue(self.win.enroll_stable_note.visible)
        self.win.var_stable.value = False
        self.win._sync_enroll_ui()
        self.assertFalse(self.win.var_enroll.disabled)
        self.assertFalse(self.win.enroll_stable_note.visible)

    def test_device_dropdown_fills_card_width(self):
        # رگرسیون: کمبوی دستگاه مستقیم داخل Column کارت بود — expand در
        # Column فقط ارتفاع را میکشد و عرض کمبو به اندازهی متن گزینهها
        # میماند؛ باید مثل fill="x" در CTk تمامعرض کارت شود (داخل Row)
        row = next(r for r in _find_all(self.root, ft.Row)
                   if self.win.var_device in r.controls)
        self.assertTrue(self.win.var_device.expand,
                        "کمبو باید expand=True داشته باشد تا در Row پُر شود")

    def test_wave_strip_is_full_width_deep_bar(self):
        # رگرسیون: Stack میلهها شناور روی کارت بود — مثل کانوس CTk باید
        # نوار تمامعرض DEEP باشد که میلهها کل عرضش را پر میکنند
        wave = self.win.spec_wave
        self.assertEqual(wave.bgcolor, "#181818")            # DEEP
        self.assertTrue(wave.expand, "نوار موج باید تمامعرض کارت شود")
        self.assertIs(wave.content, self.win.spec_bars)
        self.assertEqual(wave.height, 28)
        self.assertEqual(wave.padding.bottom, 4, "خط کفی SPECS_H+4 مثل CTk")

    def test_spectrum_bars_fill_strip_width(self):
        # رگرسیون: میلهها با گام ثابت ۹px به راست چسبیده بودند و در
        # پنجرهی عریض ابتدای نوار خالی میماند. حالا هر میله expand دارد
        # و کل عرض نوار را پر میکند؛ vertical_alignment=END کف مشترک
        # میسازد (رشد از پایین مثل CTk).
        self.assertIsInstance(self.win.spec_bars, ft.Row)
        self.assertEqual(len(self.win._bars), 48)
        self.assertEqual(self.win.spec_bars.spacing, 3)
        self.assertEqual(self.win.spec_bars.vertical_alignment,
                         ft.CrossAxisAlignment.END)
        for i, bar in enumerate(self.win._bars):
            self.assertTrue(bar.expand, f"bar {i} باید expand باشد")
            self.assertIsNone(bar.left, "میله دیگر Positioned نیست")
        # ارتفاع ثابتِ نوار: تغییر height میله نباید Row را جابهجا کند
        self.assertEqual(self.win.spec_wave.height, 28)

    def test_animation_moves_bar_heights(self):
        # باگ رگرسیون: تست صدا باید از RMS واقعی میکروفون مواج بسازد.
        # فیک تستر: صف از قبل RMS دارد → تیک اول باید میله‌ها را بلند کند
        # و حکم سبز «کار می‌کند» بدهد.
        import queue as _q
        import threading as _th
        self.win._tester_q = _q.Queue()
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None     # فیک: بدون InputStream
        for _ in range(4):
            self.win._tester_q.put(("rms", 0.03))  # صدای بلند
        self.win._toggle_test(None)     # شروع — تیک اول صف را می‌خواند
        moved = [b for b in self.win._bars if b.height > 3]
        self.assertTrue(moved, "با صدای تزریقی باید میله‌ها بلند شوند")
        colors = {b.bgcolor for b in self.win._bars}
        self.assertTrue(colors & {"#22c55e", "#4f8f68"},
                        "میله‌های فعال باید رنگ موج بگیرند")
        self.assertEqual(self.win.verdict.value, "میکروفون کار می‌کند — صدای واضح")
        self.assertEqual(self.win.verdict.color, "#22c55e")  # ACCENT
        self.win._end_test()           # توقف تمیز — تایمر و میله‌ها
        self.assertTrue(all(b.height == 3 for b in self.win._bars))
        self.assertEqual(self.win.test_btn.content, "شروع تست")

    def test_quality_ignores_empty_queue_zeros(self):
        # رگرسیون: صفرِ «صف خالی» در آمار کیفیت جمع نمیشد فقط باید
        # نمونه‌های واقعی بلاک میکروفون جمع شوند — وگرنه کف نویز صفر و
        # SNR متورم میشد و میکروفون پرنویز «خوب» خوانده میشد
        import queue as _q
        import threading as _th
        self.win._tester_q = _q.Queue()
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None
        self.win._testing = True
        env = [0.0]
        vals = [0.0] * 48   # تاریخچه‌ی موج (float) — میله‌ها خودشان از این می‌سند
        # تیک با صف خالی — هیچ نمونه‌ای جمع نشود
        self.win._tick_once(vals, env)
        self.assertEqual(getattr(self.win, "_test_vals", []),
                         [], "صف خالی نباید نمونه‌ی صفر تولید کند")
        self.assertEqual(self.win.quality_lbl.value, "")
        # نمونه‌ی واقعی — جمع و کیفیت حساب می‌شود
        self.win._tester_q.put(("rms", 0.05))
        self.win._tick_once(vals, env)
        self.assertEqual(len(self.win._test_vals), 1)
        self.assertAlmostEqual(self.win._test_vals[0], 0.05)
        self.assertIn("کیفیت ورودی", self.win.quality_lbl.value)

    def test_quality_window_caps_at_400(self):
        import queue as _q
        import threading as _th
        self.win._tester_q = _q.Queue()
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None
        self.win._testing = True
        env = [0.0]
        vals = [0.0] * 48
        for _ in range(405):
            self.win._tester_q.put(("rms", 0.02))
            self.win._tick_once(vals, env)
        self.assertEqual(len(self.win._test_vals), 400, "پنجره‌ی کیفیت باید سقف داشته باشد")

    def test_tester_tried_chain_is_indices(self):
        # رگرسیون: زنجیره‌ی تست باید ایندکس باشد — استخراج دوباره‌ی
        # e['index'] روی int خطای «'int' object is not subscriptable»
        # می‌داد و کل تست صدا همان اول می‌مرد
        devices = getattr(self.win, "_devices", [])
        if not devices:
            self.skipTest("دستگاه ورودی در این محیط یافت نشد")
        idx, label = devices[0]
        tried = self.win._tester_tried(idx)
        self.assertEqual(tried[0], idx)
        for d in tried:
            self.assertIsInstance(d, int)
        # مسیرهای جایگزین = خروجی device_siblings همان دستگاه
        from app.recorder import current_input_devices, device_siblings
        expected = [e["index"] for e in
                    device_siblings(self.win._all_inputs, idx)]
        self.assertEqual(tried[1:], expected)

    def test_animation_silence_shows_no_signal(self):
        # بدون صدا: RMS=0 → حکم قرمز «سیگنالی نمی‌آید»؛ موج پایه‌ی نفس
        # عمداً زنده است (تا کاربر بفهمد تست اجراست) ولی دامنه‌ی محدود —
        # میله‌ها نباید از ~۸px (نفس) بلندتر شوند
        import queue as _q
        import threading as _th
        self.win._tester_q = _q.Queue()
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None
        self.win._toggle_test(None)     # شروع — صف خالی → RMS=0
        heights = [b.height for b in self.win._bars]
        self.assertLessEqual(max(heights), 10.0,
                             "نفس پایه نباید از ~۸px بلندتر شود")
        self.assertGreaterEqual(max(heights), 4.0,
                                "نفس پایه باید مرئی باشد")
        self.assertEqual(self.win.verdict.value, "سیگنالی نمی‌آید — دستگاه دیگری را امتحان کن")
        self.assertEqual(self.win.verdict.color, "#e5484d")  # DANGER
        self.win._end_test()

    def test_toggle_test_stops_on_stream_error(self):
        # باگ رگرسیون: اگر باز کردن میکروفون خطا بدهد، دکمه باید به
        # «شروع تست» برگردد و حکم خطا قرمز نمایش داده شود
        import queue as _q
        import threading as _th
        self.win._tester_q = _q.Queue()
        self.win._tester_stop = _th.Event()
        self.win._start_tester = lambda: None
        self.win._toggle_test(None)     # شروع
        self.win._tester_q.put(("err", "device not found"))
        # تیک بعدی ارور را می‌خواند — صبر برای یک تیک تایمر (80ms)
        import time
        deadline = time.monotonic() + 1.0
        while self.win.test_btn.content == "توقف تست" and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.win.test_btn.content, "شروع تست",
                         "خطای استریم باید تست را خاموش کند")
        self.assertTrue(self.win.verdict.value.startswith("خطا:"))
        self.assertEqual(self.win.verdict.color, "#e5484d")
        self.assertFalse(getattr(self.win, "_testing", False))


# ---------- هلپرهای تم ----------

class TestThemeHelpers(unittest.TestCase):
    def test_palette_matches_ctk_theme(self):
        from flet_ui import theme as t
        from app import theme as ctk_theme
        for name in ("BG", "SURFACE", "SURFACE_2", "SURFACE_3", "BORDER", "DEEP",
                     "FG", "FG_DIM", "ACCENT", "ACCENT_HOVER", "ON_ACCENT",
                     "DANGER", "DANGER_HOVER", "ON_DANGER", "WARN"):
            self.assertEqual(getattr(t, name), getattr(ctk_theme, name), name)

    def test_card_structure(self):
        from flet_ui import theme as t
        c = t.card("عنوان", ft.Text("متن"))
        self.assertEqual(c.bgcolor, "#2a2a2a")
        self.assertEqual(c.border_radius, 10)
        texts = [x.value for x in _walk(c) if isinstance(x, ft.Text)]
        self.assertIn("عنوان", texts)

    def test_card_merges_consecutive_dims(self):
        # درخواست کاربر: dimهای پشتسرهم باید در یک متن پیوسته بچسبند —
        # نه اینکه هر پاراگراف در خط جدیدی شروع شود؛ کنترلِ بینشان
        # (سوییچ/ورودی) مرز است و ادغام نمیشود
        from flet_ui import theme as t
        c = t.card("کارت", t.dim("پاراگراف یک"), t.dim("پاراگراف دو"),
                   ft.Text("کنترل میانی"), t.dim("پاراگراف سه"))
        texts = [x.value for x in _walk(c) if isinstance(x, ft.Text)]
        self.assertIn("پاراگراف یک  •  پاراگراف دو", texts,
                      "دو dim پشتسرهم باید ادغام شوند")
        self.assertIn("پاراگراف سه", texts,
                      "dim بعد از کنترل میانی باید جدا بماند")
        self.assertNotIn("پاراگراف دو", texts)

    def test_btn_style_no_hover_color_kwarg(self):
        from flet_ui import theme as t
        s = t.btn_style()
        self.assertEqual(s.shape.radius, 8)
        self.assertEqual(s.elevation, 0)

    def test_dropdown_uses_on_select(self):
        from flet_ui import theme as t
        d = t.dropdown(["الف", "ب"], "ب")
        self.assertEqual(d.value, "ب")
        # باگ رگرسیون: Option(متن) فقط key را ست می‌کرد و label خالی می‌ماند
        self.assertEqual([(o.key, o.text) for o in d.options],
                         [("الف", "الف"), ("ب", "ب")])


class TestHotkeyCapture(SettingsTestBase):
    """کپچر کلید میانبر از جریان hook کتابخانه keyboard — قرینه‌ی CTk."""

    def setUp(self):
        super().setUp()
        # خودِ hook سراسری (LL hook ویندوز) در تست نصب نمی‌شود —
        # منطق ساخت ترکیب با رویداد خام تست می‌شود، سیم‌کشی در GUI
        self.win._hk_hook_on = lambda: None
        self.win._hk_hook_off = lambda: None

    def _raw(self, name, etype="down"):
        self.win._hk_raw(SimpleNamespace(name=name, event_type=etype))

    def test_valid_hotkey_cases(self):
        from flet_ui.settings_window import _valid_hotkey
        for hk in ("ctrl+shift+space", "ctrl+alt+f5", "f5", "f12", "shift+a"):
            self.assertTrue(_valid_hotkey(hk), hk)
        for hk in ("", "ctrl", "a", "ctrl+shift", "x"):
            self.assertFalse(_valid_hotkey(hk), hk)

    def test_focus_starts_capture_and_blur_stops_it(self):
        w = self.win
        w._hk_focus()
        self.assertTrue(w._hk_capturing)
        self.assertEqual(w._hk_mods, set())
        w._hk_blur()
        self.assertFalse(w._hk_capturing)

    def test_combo_capture(self):
        w = self.win
        w._hk_focus()
        self._raw("left ctrl")
        self._raw("left alt")
        self._raw("k")
        self.assertEqual(w.var_hotkey.value, "ctrl+alt+k")
        self.assertEqual(w.hk_hint.color, "#9b9b9b")   # FG_DIM — بدون خطا
        self._raw("left alt", "up")
        self._raw("left ctrl", "up")
        self.assertEqual(w._hk_mods, set())

    def test_space_combo(self):
        w = self.win
        w._hk_focus()
        self._raw("left ctrl")
        self._raw("left shift")
        self._raw("space")
        self.assertEqual(w.var_hotkey.value, "ctrl+shift+space")

    def test_escape_cancels_to_previous(self):
        w = self.win
        w._hk_focus()
        self._raw("f9")
        self.assertEqual(w.var_hotkey.value, "f9")
        self._raw("esc")
        self.assertEqual(w.var_hotkey.value, DEFAULTS["hotkey"])
        self.assertEqual(w.hk_hint.color, "#9b9b9b")

    def test_plain_key_rejected_with_danger_hint(self):
        w = self.win
        w._hk_focus()
        self._raw("a")   # تک‌کلیدی بدون modifier و بدون F
        self.assertEqual(w.var_hotkey.value, DEFAULTS["hotkey"])
        self.assertEqual(w.hk_hint.color, "#e5484d")   # DANGER

    def test_modifier_only_press_changes_nothing(self):
        w = self.win
        w._hk_focus()
        self._raw("left shift")
        self._raw("left shift", "up")
        self.assertEqual(w.var_hotkey.value, DEFAULTS["hotkey"])

    def test_capture_ignored_after_blur(self):
        w = self.win
        w._hk_blur()
        self._raw("f9")
        self.assertEqual(w.var_hotkey.value, DEFAULTS["hotkey"])


class TestSaveValidation(SettingsTestBase):
    """اعتبارسنجی پیش از ذخیره — قرینه‌ی save() نسخه CTk."""

    def test_validate_ok_on_defaults(self):
        self.assertIsNone(self.win._validate(self.win._collect()))

    def test_validate_rejects_bad_hotkey(self):
        from flet_ui.settings_window import HK_INVALID
        data = self.win._collect()
        data["hotkey"] = "hello"
        self.assertEqual(self.win._validate(data), ("hotkey", HK_INVALID))

    def test_validate_rejects_hotword_boost_without_words(self):
        data = self.win._collect()
        data["hotword_boost"] = True
        data["hotwords"] = []
        self.assertEqual(self.win._validate(data)[0], "hotwords")

    def test_save_refusal_keeps_window_open_and_writes_nothing(self):
        # ذخیره‌ی نامعتبر: مثل CTk پنجره باز می‌ماند، دیسک نوشته نمی‌شود
        import asyncio
        w = self.win
        w.var_hotkey.value = "hello"
        asyncio.run(w._save())
        self.assertFalse(self.page.window.destroyed, "پنجره نباید بسته شود")
        self.assertEqual(w.hk_hint.color, "#e5484d")
        self.assertIsNone(w.result)


class TestAutoStopAndEnroll(SettingsTestBase):
    """auto_stop_sec از cfg خوانده شود و enroll_alias از سوییچ ذخیره شود."""

    def _win_with(self, **cfg_over):
        from flet_ui.settings_window import SettingsWindow
        page = MockPage()

        class Fixed(SettingsWindow):
            def _load(self):
                return {**DEFAULTS, **cfg_over}

        win = Fixed(page)
        self.page = page
        self.win = win
        self.root = page.added[0]
        return win

    def test_saved_auto_stop_shows_in_combo(self):
        win = self._win_with(auto_stop_sec=10)
        self.assertEqual(win.var_auto_stop.value, "۱۰ ثانیه",
                         "مقدار ذخیرهشده باید در کمبو بیاید — نه همیشه «خاموش»")

    def test_default_auto_stop_is_off(self):
        win = self._win_with()
        self.assertEqual(win.var_auto_stop.value, "خاموش")

    def test_collect_reads_enroll_switch(self):
        win = self._win_with(enroll_alias=False)
        win.var_enroll.value = True
        self.assertTrue(win._collect()["enroll_alias"])
        win.var_enroll.value = False
        self.assertFalse(win._collect()["enroll_alias"],
                         "تغییر سوییچ باید در ذخیره بیاید — نه مقدار دیسک")

    def test_reset_syncs_enroll_ui(self):
        win = self._win_with(enroll_alias=False, stable_live=True)
        # حالت پایدار: سوییچ غیرفعال + نکته‌ی هشدار پیدا
        self.assertTrue(win.var_enroll.disabled)
        self.assertTrue(win.enroll_stable_note.visible)
        # بازنشانی → DEFAULTS: حالت پایدار خاموش، سوییچ فعال، نکته پنهان
        win._reset()
        self.assertFalse(win.var_stable.value)
        self.assertFalse(win.var_enroll.disabled, "پس از ریست سوییچ باید از حالت پایدار دربیاید")
        self.assertFalse(win.enroll_stable_note.visible)
        self.assertEqual(win.var_enroll.value, bool(DEFAULTS["enroll_alias"]))

    def test_apply_sets_enroll_from_data(self):
        win = self._win_with()
        win._apply({**DEFAULTS, "enroll_alias": True})
        self.assertTrue(win.var_enroll.value)


class TestWindowEvents(unittest.TestCase):
    """نوع رویداد پنجره — flet بسته به نسخه رشته یا WindowEventType می‌دهد."""

    def test_string_type(self):
        from flet_ui.run import window_event_kind
        self.assertEqual(window_event_kind(SimpleNamespace(type="close")), "close")
        self.assertEqual(window_event_kind(SimpleNamespace(type="MINIMIZE")), "minimize")

    def test_enum_type(self):
        from flet_ui.run import window_event_kind
        ev = SimpleNamespace(type=SimpleNamespace(value="minimize"))
        self.assertEqual(window_event_kind(ev), "minimize")

    def test_missing_type(self):
        from flet_ui.run import window_event_kind
        self.assertEqual(window_event_kind(SimpleNamespace()), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
