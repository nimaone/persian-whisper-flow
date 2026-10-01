"""تست چرخه‌ی ضبط/ترنسکرایب بک‌اند فلت — بدون مدل و میکروفون واقعی.

اجرا:
    .venv/Scripts/python.exe -m tests.test_flet_backend

همه‌ی وابستگی‌های بیرونی فیک می‌شوند (engine/recorder/insert/hotkey)
و فقط هماهنگی حالت‌ها، ترنسکرایب زنده، توقف خودکار با سکوت، درج متن
و اعمال تنظیمات آزموده می‌شود.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.config import DEFAULTS, Config
from flet_ui.backend import (STATE_IDLE, STATE_LOADING, STATE_RECORDING,
                             STATE_TRANSCRIBING, DictationApp)


# ---------- فیک‌ها ----------

class FakeWin:
    """قرینه‌ی ControlWindow فلت — فقط ضبط."""

    def __init__(self):
        self.states = []
        self.statuses = []
        self.errors = []

    def set_state(self, state, hotkey):
        self.states.append(state)

    def set_status(self, msg):
        self.statuses.append(msg)

    def set_error(self, msg):
        self.errors.append(msg)


class FakeRecorder:
    """قرینه‌ی Recorder — بافر ثابت با «صدا»ی مصنوعی + فاز سکوت."""

    def __init__(self, device=None, level=0.05):
        self.device = device
        self.level = level
        self._buf = (np.sin(np.linspace(0, 100, 16000 * 2))
                     .astype(np.float32) * level)
        self.started = False
        self.stopped = False
        self.speech_detected = False  # حلقه‌ی زنده حداقل یک‌بار صدای واقعی دیده؟

    def go_silent(self):
        """شروع فاز سکوت — بافر صفر می‌شود (RMS زیر آستانه)."""
        self._buf = np.zeros(16000 * 2, dtype=np.float32)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_tail_16k(self, window_sec):
        tail = self._buf[-int(window_sec * 16000):]
        rms = (float(np.sqrt((tail.astype(np.float64) ** 2).mean()))
               if tail.size else 0.0)
        if rms > 0.003:
            self.speech_detected = True
        return tail

    def get_buffer_16k(self):
        return self._buf

    def duration_sec(self):
        """طول بافر بر حسب ثانیه — قرینه‌ی Recorder برای t_offset پنجره."""
        return len(self._buf) / 16000.0

    def recent_rms(self):
        return self.level


class FakeEngine:
    def transcribe(self, samples, sample_rate=16000):
        return "سلام دنیا"


def make_app(**kw):
    """بک‌اند با همه‌ی وابستگی‌های فیک."""
    cfg = Config(dict(DEFAULTS))
    win = FakeWin()
    rec = FakeRecorder()
    inserted = []
    app = DictationApp(
        win, cfg=cfg,
        engine_loader=lambda: FakeEngine(),
        recorder_factory=lambda device: rec,
        insert_fn=lambda text, method, restore: inserted.append(text),
        send_key_fn=lambda name: inserted.append(f"key:{name}"),
        device_probe=lambda: 0,
        register_hotkey=kw.pop("register_hotkey", False),
        **kw,
    )
    return app, win, rec, inserted


def wait_for(pred, timeout=5.0):
    """انتظار برای شرط در threadهای بک‌اند."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


class BackendCycle(unittest.TestCase):
    def test_startup_loads_model_and_goes_idle(self):
        app, win, _, _ = make_app()
        app.start()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE),
                        "پس از لود مدل باید idle شود")
        self.assertEqual(win.states, [STATE_LOADING, STATE_IDLE])
        self.assertIsNotNone(app.live)
        self.assertTrue(app._device_ready.is_set())

    def test_full_record_transcribe_insert_cycle(self):
        app, win, rec, inserted = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app.toggle_recording()
        self.assertTrue(wait_for(lambda: app.state == STATE_RECORDING),
                        "باید شروع به ضبط کند")
        self.assertTrue(rec.started)
        # متن زنده باید به overlay برود (win.statuses فقط برای notify است)
        self.assertTrue(wait_for(lambda: rec.speech_detected),
                        "حلقه‌ی زنده باید بافر را بخواند")
        # توقف — سپس transcribing (ترنسکرایب نهایی در thread جدا) و بعد idle + درج
        app.toggle_recording()
        self.assertTrue(wait_for(lambda: app.state == STATE_TRANSCRIBING),
                        "پس از توقف باید transcribing شود")
        self.assertTrue(rec.stopped)
        self.assertTrue(wait_for(lambda: inserted == ["سلام دنیا"], timeout=8),
                        "متن نهایی باید درج شود")
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE),
                        "پس از ترنسکرایب باید idle شود")
        self.assertEqual(win.states[-1], STATE_IDLE)

    def test_toggle_during_loading_ignored(self):
        app, win, rec, inserted = make_app()
        # بدون start — state همان loading می‌ماند
        app.toggle_recording()
        time.sleep(0.2)
        self.assertEqual(app.state, STATE_LOADING)
        self.assertFalse(rec.started)
        self.assertEqual(inserted, [])

    def test_stop_without_start_returns_to_idle(self):
        app, win, rec, inserted = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        # وضعیت را دستی recording می‌کنیم ولی رکوردر را None می‌گذاریم
        with app._state_lock:
            app.state = STATE_RECORDING
            app.recorder = None
        app.toggle_recording()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))

    def test_auto_stop_on_silence(self):
        # ۱ ثانیه توقف خودکار؛ رکوردار: ۱ ثانیه صدا، بعد سکوت مطلق
        app, win, rec, inserted = make_app()
        app.cfg.set("auto_stop_sec", 1)
        # فاز صدا — RMS بالای آستانه، speech_seen=True می‌شود
        rec.level = 0.05
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app.toggle_recording()
        self.assertTrue(wait_for(lambda: app.state == STATE_RECORDING))
        self.assertTrue(wait_for(lambda: rec.speech_detected, timeout=6),
                        "فاز صدا باید در حلقه‌ی زنده دیده شود")
        # فاز سکوت — بافر ساکت می‌شود؛ بعد از ۱ ثانیه باید خودش متوقف شود
        rec.go_silent()
        self.assertTrue(wait_for(lambda: app.state == STATE_TRANSCRIBING, timeout=8),
                        "توقف خودکار باید رخ دهد")
        self.assertTrue(wait_for(lambda: inserted == ["سلام دنیا"], timeout=8),
                        "پس از ترنسکرایب، متن باید درج شود")
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))

    def test_empty_final_shows_notify_no_insert(self):
        app, win, rec, inserted = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app.toggle_recording()
        wait_for(lambda: app.state == STATE_RECORDING)
        # ترنسکرایب نهایی را خالی کنیم
        app.live = None
        app.toggle_recording()
        self.assertTrue(wait_for(lambda: app.state == STATE_LOADING))
        time.sleep(0.3)
        self.assertEqual(inserted, [], "بدون متن نباید درج شود")
        self.assertTrue(any("تشخیص داده نشد" in s for s in win.statuses),
                        "باید پیام اطلاع بدهد")

    def test_alias_map_applied_unless_stable(self):
        # واژه‌های ثبت‌صوتی روی درج نهایی اعمال می‌شوند؛ در حالت
        # متن زنده پایدار این لایه عمداً خاموش است (گویش beam-2)
        app, _, _, inserted = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app._alias_map = {"ویسپرفارسی": "دیکته‌یار"}
        app.cfg.set("stable_live", False)
        app._insert("ويسپر فارسی خوب است")
        self.assertEqual(inserted, ["دیکته‌یار خوب است"])
        app.cfg.set("stable_live", True)
        app._insert("ويسپر فارسی خوب است")
        # ITN حرف عربی ي را به فارسی ی نرمال می‌کند — بدون جایگزینی
        self.assertEqual(inserted[-1], "ویسپر فارسی خوب است")

    def test_voice_commands_routing(self):
        from unittest.mock import patch
        from app.voice_commands import Segment
        app, win, rec, inserted = make_app()
        app.cfg.set("voice_commands", True)
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app.toggle_recording()
        wait_for(lambda: app.state == STATE_RECORDING)
        fake_segments = [Segment(kind="text", value="سلام"),
                         Segment(kind="key", value="enter")]
        # patch باید تا پایان ترنسکرایب زنده بماند (در thread جدا اجرا می‌شود)
        with patch("app.voice_commands.parse", return_value=fake_segments), \
             patch("app.voice_commands.execute",
                   side_effect=lambda segs, ins, key: (ins("سلام"), key("enter"))):
            app.toggle_recording()
            self.assertTrue(
                wait_for(lambda: inserted == ["سلام", "key:enter"], timeout=8),
                "فرمان صوتی باید به insert و send_key مسیر شود")


class BackendConfig(unittest.TestCase):
    def test_apply_config_changes_hotkey_signature_and_device(self):
        app, win, _, _ = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        # رجیستری واقعی لمس نشود — این تست درباره‌ی دستگاه است
        with unittest.mock.patch("flet_ui.backend._apply_autostart", lambda e: None):
            # apply_config از دیسک بازخوانی می‌کند — مثل پنجره تنظیمات در فایل می‌نویسیم
            saved = app.settings_path_backup = Config.load()
            data = dict(saved.data)
            data["input_device"] = 3
            Config(data).save()
            app.apply_config()
            self.assertEqual(app.device, 3)
            self.assertTrue(app._device_ready.is_set())
            # بازگشت به خودکار → تشخیص مجدد (پروب واقعی WASAPI گاهی کند است)
            Config(dict(saved.data)).save()
            app.apply_config()
            self.assertTrue(wait_for(lambda: app.device == 0, timeout=15))
            # فایل تنظیمات را به حالت اول برگردان
            saved.save()

    def test_engine_key_reflects_hotword_settings(self):
        # فروشگاه ثبت واژه به مسیر خالی می‌رود — نتایج نباید به
        # enrollments.json واقعی ماشین وابسته باشند
        import tempfile
        from pathlib import Path as _P
        from unittest import mock
        from app import enroll as _enroll_mod
        fake = _P(tempfile.gettempdir()) / "dikteyar-tests-no-enrollments.json"
        with mock.patch.object(_enroll_mod, "store_path", lambda: fake):
            app, _, _, _ = make_app()
            k0 = app._engine_key
            app.cfg.set("hotword_boost", True)
            app.cfg.set("hotwords", ["نیما", "x", "ويسپر"])
            k1 = app._engine_key
            self.assertNotEqual(k0, k1)
            # واژه‌ی تک‌حرفی فیلتر می‌شود
            self.assertEqual(k1, (True, ("نیما", "ويسپر"), False))
            # واژه‌های ثبت‌صوتی هم به هات‌وورد اضافه می‌شوند (هر دو حالت روشن)
            _enroll_mod.store_path().write_text(
                '{"entries": [{"word": "دیکته‌یار", "variants": ["ویسپر فلوی فارسی"]}]}',
                encoding="utf-8")
            k2 = app._engine_key
            self.assertEqual(k2, (True, ("نیما", "ويسپر", "دیکته‌یار"), False))
            # تغییر متن زنده پایدار هم موتور را عوض می‌کند
            app.cfg.set("stable_live", True)
            self.assertNotEqual(k2, app._engine_key)
            fake.unlink(missing_ok=True)

    def test_rebuild_engine_while_recording_defers(self):
        app, win, _, _ = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        with app._state_lock:
            app.state = STATE_RECORDING
        app._rebuild_engine()
        self.assertTrue(app._engine_dirty, "وسط ضبط باید به _finish موکول شود")
        self.assertEqual(app.state, STATE_RECORDING)

    def test_quit_stops_recorder(self):
        app, win, rec, _ = make_app()
        app.start()
        wait_for(lambda: app.state == STATE_IDLE)
        app.toggle_recording()
        wait_for(lambda: app.state == STATE_RECORDING)
        app.quit()
        self.assertFalse(app._running)
        self.assertTrue(rec.stopped)


class Autostart(unittest.TestCase):
    """اجرای خودکار باید نسخه‌ی فلت را بالا بیاورد، نه CTk را."""

    def test_command_points_to_flet_launcher(self):
        from flet_ui.backend import _flet_autostart_command
        cmd = _flet_autostart_command()
        self.assertIn("dikteyar_flet.pyw", cmd)
        self.assertNotIn("main.py", cmd)
        self.assertIn("pythonw", cmd.lower(), "بدون کنسول — pythonw ترجیح دارد")

    def test_launcher_file_exists_next_to_command(self):
        from flet_ui.backend import _PROJECT_ROOT
        self.assertTrue((_PROJECT_ROOT / "dikteyar_flet.pyw").exists(),
                        "لانچرِ ثبت‌شده در رجیستری باید واقعاً موجود باشد")

    def test_source_mode_writes_flet_path_under_app_name(self):
        import flet_ui.backend as B
        from app.config import APP_NAME
        rec = {}

        class FakeKey:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake = SimpleNamespace(
            HKEY_CURRENT_USER=object(), KEY_SET_VALUE=1, REG_SZ=1,
            OpenKey=lambda *a, **k: FakeKey(),
            SetValueEx=lambda key, name, res, typ, val: rec.setdefault("set", (name, val)),
            DeleteValue=lambda key, name: rec.setdefault("deleted", name),
        )
        with mock.patch.object(B, "winreg", fake):
            B._apply_autostart(True)
        self.assertEqual(rec["set"][0], APP_NAME)
        self.assertIn("dikteyar_flet.pyw", rec["set"][1])

    def test_disable_deletes_registry_value(self):
        import flet_ui.backend as B
        from app.config import APP_NAME
        rec = {}

        class FakeKey:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake = SimpleNamespace(
            HKEY_CURRENT_USER=object(), KEY_SET_VALUE=1, REG_SZ=1,
            OpenKey=lambda *a, **k: FakeKey(),
            SetValueEx=lambda *a: rec.setdefault("set", True),
            DeleteValue=lambda key, name: rec.setdefault("deleted", name),
        )
        with mock.patch.object(B, "winreg", fake):
            B._apply_autostart(False)
        self.assertEqual(rec.get("deleted"), APP_NAME)
        self.assertNotIn("set", rec)

    def test_frozen_delegates_to_shared_set_autostart(self):
        import flet_ui.backend as B
        calls = []
        with mock.patch.object(B, "set_autostart", lambda e: calls.append(e)), \
             mock.patch.object(sys, "frozen", True, create=True):
            B._apply_autostart(True)
        self.assertEqual(calls, [True])


class OpenSettingsFailure(unittest.TestCase):
    """شکست Popen نباید هاتکی را معلق بگذارد (قرینه‌ی CTk)."""

    def test_popen_failure_restores_hotkey_and_notifies(self):
        app, _, _, _ = make_app()
        resumed, notified = [], []
        with mock.patch.object(app, "resume_hotkey", lambda: resumed.append(1)), \
             mock.patch.object(app, "_notify", lambda m: notified.append(m)), \
             mock.patch("flet_ui.backend.subprocess.Popen",
                        side_effect=OSError("پروسه باز نشد")):
            app.open_settings()
        self.assertEqual(resumed, [1], "هاتکی باید برگردد")
        self.assertEqual(len(notified), 1)
        self.assertIsNone(app._settings_proc)


class EngineRelease(unittest.TestCase):
    """موتور قدیمی بعد از تعویض/خروج رها شود — نشت حافظه نباشد."""

    def test_rebuild_releases_old_engine(self):
        app, _, _, _ = make_app()
        app.start()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))
        old = app.engine
        released = []
        old.release_detail_decoder = lambda: released.append(1)
        app.cfg.set("stable_live", True)   # امضای موتور عوض می‌شود
        app._rebuild_engine()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))
        self.assertIsNot(app.engine, old)
        self.assertEqual(released, [1])

    def test_rebuild_keeps_old_engine_when_new_fails(self):
        app, _, _, _ = make_app()
        app.start()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))
        old = app.engine
        released = []
        old.release_detail_decoder = lambda: released.append(1)

        def boom():
            raise RuntimeError("لود نشد")

        app._engine_loader = boom
        app._rebuild_engine()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))
        self.assertIs(app.engine, old, "موتور قبلی باید قابل استفاده بماند")
        self.assertEqual(released, [], "موتور سالمِ درحال‌استفاده رها نشود")

    def test_quit_releases_current_engine(self):
        app, _, _, _ = make_app()
        app.start()
        self.assertTrue(wait_for(lambda: app.state == STATE_IDLE))
        released = []
        app.engine.release_detail_decoder = lambda: released.append(1)
        app.quit()
        self.assertEqual(released, [1])


class OverlayLevel(unittest.TestCase):
    """موج زنده: ۱۰Hz از ترد level و صفر پس از توقف — مثل حلقه UI نسخه CTk."""

    def test_level_loop_pushes_then_zeroes_after_stop(self):
        app, _, rec, _ = make_app()
        ops = []
        app._ov = lambda op, arg=None: ops.append((op, arg))
        app.state = STATE_RECORDING
        app.recorder = rec
        t = threading.Thread(target=app._level_loop, args=(rec,), daemon=True)
        t.start()
        time.sleep(0.35)
        self.assertTrue(any(op == "level" and arg and arg > 0 for op, arg in ops),
                        "حین ضبط باید سطح واقعی پوش شود")
        app.recorder = None   # پایان ضبط
        t.join(timeout=3)
        self.assertFalse(t.is_alive(), "ترد باید بعد از توقف تمام شود")
        self.assertEqual(ops[-1], ("level", 0.0), "موج نباید روی آخرین مقدار یخ بزند")

    def test_partial_loop_no_longer_pushes_level(self):
        # حلقه‌ی partial هر ۰٫۸s پوش میکرد → موج پرشدار؛ حالا جای دیگری است
        import inspect
        import flet_ui.backend as B
        src = inspect.getsource(B.DictationApp._partial_loop)
        self.assertNotIn('"level"', src)


class OverlayLoopResilience(unittest.TestCase):
    """یک خطای درون op نباید ترد اوورلی را بیندازد (قرینه‌ی CTk)."""

    def test_bad_op_does_not_kill_loop(self):
        app, _, _, _ = make_app(with_overlay=False)
        calls = []

        class FakeOverlay:
            def set_font_size(self, n):
                calls.append("font")

            def show(self):
                calls.append("show")

            def hide(self):
                calls.append("hide")

            def update_text(self, *a):
                calls.append("text")

            def set_processing(self):
                calls.append("processing")

            def update_level(self, arg):
                raise RuntimeError("tk خراب")  # خطای واقعی در op

            def tick(self):
                pass

        # _overlay_loop خودش Overlay را از app.overlay import می‌کند
        with mock.patch("app.overlay.Overlay", FakeOverlay):
            t = threading.Thread(target=app._overlay_loop, daemon=True)
            t.start()
            time.sleep(0.15)
            app._ov_q.put(("level", 0.5))     # op خراب
            app._ov_q.put(("hide", None))     # op بعدی باید هم برسد
            time.sleep(0.3)
            app._running = False
            t.join(timeout=2)
        self.assertFalse(t.is_alive(), "ترد نباید از خطای op مرده باشد")
        self.assertIn("hide", calls, "opهای بعد از خطا باید پردازش شوند")


class BackendStartup(unittest.TestCase):
    def test_model_failure_after_3_attempts_sets_error(self):
        win = FakeWin()
        cfg = Config(dict(DEFAULTS))
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("مدل گم شده")

        app = DictationApp(
            win, cfg=cfg, engine_loader=boom,
            recorder_factory=lambda d: FakeRecorder(),
            insert_fn=lambda *a: None,
            device_probe=lambda: 0,
            register_hotkey=False,
        )
        # بجای sleep واقعی ۵ ثانیه‌ای، فقط ۳ بار صدا زدن را چک می‌کنیم
        with unittest.mock.patch.object(app.__class__, "_load_model",
                                        lambda self: None):
            pass
        # اجرای نسخه‌ی واقعی با sleep کوتاه — مستقیم تابع داخلی را صدا می‌زنیم
        import flet_ui.backend as B
        orig_sleep = time.sleep
        time.sleep = lambda s: None  # تسریع تلاش‌ها
        try:
            app._load_model()
        finally:
            time.sleep = orig_sleep
        self.assertEqual(len(calls), 3, "باید ۳ بار تلاش کند")
        self.assertEqual(len(win.errors), 1, "بعد از ۳ شکست باید خطا بدهد")
        self.assertIn("مدل گم شده", win.errors[0])


class FakeTray:
    """قرینه‌ی Tray فلت — فقط ثبت فراخوانی‌ها."""

    def __init__(self):
        self.states = []
        self.notified = []
        self.stopped = False

    def set_state(self, state):
        self.states.append(state)

    def notify(self, text):
        self.notified.append(text)

    def stop(self):
        self.stopped = True


class TrayHooks(unittest.TestCase):
    """اتصال سینی به بک‌اند — tooltip هر حالت و بالون اعلان (مثل CTk)."""

    def test_state_change_updates_tray_tooltip(self):
        app, win, _, _ = make_app()
        app.tray = FakeTray()
        app._ui_set_state(STATE_RECORDING)
        self.assertEqual(app.tray.states, [STATE_RECORDING])
        self.assertEqual(win.states, [STATE_RECORDING])

    def test_notify_prefers_tray_balloon(self):
        app, win, _, _ = make_app()
        app.tray = FakeTray()
        app._notify("میکروفون باز نشد")
        self.assertEqual(app.tray.notified, ["میکروفون باز نشد"])
        self.assertEqual(win.statuses, [], "با سینی، نوار وضعیت بازنویسی نمی‌شود")

    def test_notify_falls_back_to_status_without_tray(self):
        app, win, _, _ = make_app()
        app._notify("میکروفون باز نشد")
        self.assertEqual(win.statuses, ["میکروفون باز نشد"])

    def test_broken_tray_falls_back_to_status(self):
        class BadTray(FakeTray):
            def notify(self, text):
                raise RuntimeError("سینی مرده")

        app, win, _, _ = make_app()
        app.tray = BadTray()
        app._notify("x")
        self.assertEqual(win.statuses, ["x"])

    def test_quit_stops_tray(self):
        app, _, _, _ = make_app()
        app.tray = FakeTray()
        app.quit()
        self.assertTrue(app.tray.stopped)

    def test_quit_terminates_settings_process(self):
        class FakeProc:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None  # هنوز باز است

            def terminate(self):
                self.terminated = True

        app, _, _, _ = make_app()
        proc = FakeProc()
        app._settings_proc = proc
        app.quit()
        self.assertTrue(proc.terminated, "پنجره‌ی تنظیمات نباید یتیم بماند")

    def test_quit_leaves_finished_settings_process(self):
        class DoneProc:
            def poll(self):
                return 0  # قبلاً بسته شده

            def terminate(self):
                raise AssertionError("پروسه‌ی بسته نباید terminate شود")

        app, _, _, _ = make_app()
        app._settings_proc = DoneProc()
        app.quit()  # نباید استثنا بدهد


if __name__ == "__main__":
    unittest.main(verbosity=2)
