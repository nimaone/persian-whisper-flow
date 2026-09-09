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
        # apply_config از دیسک بازخوانی می‌کند — مثل پنجره تنظیمات در فایل می‌نویسیم
        saved = app.settings_path_backup = Config.load()
        data = dict(saved.data)
        data["input_device"] = 3
        Config(data).save()
        app.apply_config()
        self.assertEqual(app.device, 3)
        self.assertTrue(app._device_ready.is_set())
        # بازگشت به خودکار → تشخیص مجدد
        Config(dict(saved.data)).save()
        app.apply_config()
        self.assertTrue(wait_for(lambda: app.device == 0))
        # فایل تنظیمات را به حالت اول برگردان
        saved.save()

    def test_engine_key_reflects_hotword_settings(self):
        app, _, _, _ = make_app()
        k0 = app._engine_key
        app.cfg.set("hotword_boost", True)
        app.cfg.set("hotwords", ["نیما", "x", "ويسپر"])
        k1 = app._engine_key
        self.assertNotEqual(k0, k1)
        # واژه‌ی تک‌حرفی فیلتر می‌شود
        self.assertEqual(k1, (True, ("نیما", "ويسپر")))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
