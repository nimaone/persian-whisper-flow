"""وِیس‌فلو فارسی — نقطه ورود.

اپ ترِی: کلید میانبر جهانی → ضبط با نمایش زنده → ترنسکرایپ → درج در برنامه فعال.

قانون threadها: همه‌ی دستورات Tkinter فقط از حلقه‌ی اصلی (_ui_loop) انجام
می‌شود؛ بقیه‌ی threadها (hotkey/tray/ASR) از طریق صف ui_q درخواست می‌فرستند.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

# اجازه اجرای مستقیم: python app/main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from app import voice_commands
from app.asr import LiveTranscriber, load_engine
from app.config import Config, set_autostart
from app.control_window import ControlWindow
from app.overlay import Overlay
from app.paster import insert_text, send_key
from app.recorder import Recorder, detect_best_device

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_STARTING = "starting"
STATE_RECORDING = "recording"
STATE_TRANSCRIBING = "transcribing"

PARTIAL_INTERVAL = 0.8  # ثانیه بین ترنسکرایپ‌های زنده
SILENCE_RMS = 0.003     # آستانه سکوت برای توقف خودکار


def _beep(start: bool):
    """بوق کوتاه شروع/پایان ضبط — بی‌صدا در خطا (تنظیم sound_feedback)."""
    try:
        import winsound
        winsound.Beep(880 if start else 660, 60)
    except Exception:
        pass


class App:
    def __init__(self):
        self.cfg = Config.load()
        self.state = STATE_LOADING
        self.recorder: Recorder | None = None
        self.device: int | None = self.cfg.get("input_device")
        self.engine = None
        self.live: LiveTranscriber | None = None
        self.overlay: Overlay | None = None
        self.ui_q: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._device_ready = threading.Event()
        self._running = True
        self._tray = None
        self._hotkey_registered = ""
        self._silence_t0 = None  # زمان شروع سکوت فعلی (برای توقف خودکار)

    # ---------- راه‌اندازی ----------
    def start(self):
        self.overlay = Overlay()
        self.control = ControlWindow(self.overlay.root)
        self.control.on_toggle = self.toggle_recording
        self.control.on_settings = lambda: self.ui_q.put(("settings", None))
        self.control.set_state("loading", self.cfg.get("hotkey"))
        # بستن پنجره کنترل = خروج از برنامه (tray جدا کار می‌کند)
        self.control.win.protocol("WM_DELETE_WINDOW", self.quit)
        self._start_tray()
        self.apply_hotkey()
        # لود مدل و تشخیص میکروفون — هر دو در پس‌زمینه و موازی
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._detect_device, daemon=True).start()
        # حلقه UI — تنها جایی که به Tkinter دست می‌زنیم (thread اصلی)
        self._ui_loop()

    def _load_model(self):
        for attempt in range(3):
            try:
                self.engine = load_engine(
                    num_threads=int(self.cfg.get("num_threads") or 4))
                self.live = LiveTranscriber(self.engine)
                with self._state_lock:
                    if self.state in (STATE_LOADING, STATE_STARTING):
                        self.state = STATE_IDLE
                self._ui_set_state(self.state)
                return
            except Exception as e:
                if attempt == 2:
                    self.ui_q.put(("error", f"لود مدل ناموفق: {str(e)[:40]}"))
                    return
                self._notify(f"لود مدل ناموفق؛ تلاش مجدد ({attempt + 2}/3)…")
                time.sleep(5)

    def _detect_device(self):
        if self.device is not None:
            self._device_ready.set()
            return
        self.device = detect_best_device()
        self._device_ready.set()

    # ---------- تری ----------
    def _tray_icon(self):
        # لوگوی برنامه — در نبود asset، همان شکل ساده‌ی قبلی
        logo = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
        if logo.exists():
            try:
                return Image.open(logo)
            except Exception:
                pass
        img = Image.new("RGB", (64, 64), (30, 30, 40))
        d = ImageDraw.Draw(img)
        d.ellipse((18, 14, 46, 42), fill=(34, 197, 94))
        d.rectangle((29, 38, 35, 54), fill=(34, 197, 94))
        return img

    def _start_tray(self):
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(
                "نمایش پنجره", lambda *_: self.ui_q.put(("show_window", None)),
                default=True),
            pystray.MenuItem("شروع/توقف ضبط", lambda *_: self.toggle_recording()),
            pystray.MenuItem("تنظیمات", lambda *_: self.ui_q.put(("settings", None))),
            pystray.MenuItem("خروج", lambda *_: self.quit()),
        )
        self._tray = pystray.Icon(
            "WhisperFlowFarsi", self._tray_icon(), "وِیس‌فلو فارسی (در حال بارگذاری)", menu
        )
        threading.Thread(target=self._tray.run, daemon=True).start()

    def _tray_set_state(self, text: str):
        if self._tray:
            try:
                self._tray.title = text[:63]
            except Exception:
                pass

    def _ui_set_state(self, state: str):
        """به‌روزرسانی پنجره کنترل + tray — از هر thread، UI از طریق صف."""
        self.ui_q.put(("cstate", state))
        titles = {
            "loading": "وِیس‌فلو فارسی (در حال بارگذاری)",
            "idle": "وِیس‌فلو فارسی — آماده",
            "recording": "وِیس‌فلو فارسی — در حال ضبط",
            "transcribing": "وِیس‌فلو فارسی — در حال تشخیص",
            "starting": "وِیس‌فلو فارسی",
        }
        self._tray_set_state(titles.get(state, "وِیس‌فلو فارسی"))

    def _notify(self, text: str):
        if self._tray:
            try:
                self._tray.notify(text, "وِیس‌فلو فارسی")
            except Exception:
                pass

    # ---------- کلید میانبر ----------
    def apply_hotkey(self):
        """ثبت/به‌روزرسانی کلید میانبر — در هر thread قابل فراخوانی."""
        import keyboard

        self._unregister_hotkey()
        combo = self.cfg.get("hotkey")
        keyboard.add_hotkey(combo, self.toggle_recording, suppress=True,
                            trigger_on_release=False)
        self._hotkey_registered = combo

    def _unregister_hotkey(self):
        """حذف هات‌کی فعلی (برای تعلیق حین دیالوگ capture تنظیمات)."""
        import keyboard

        if self._hotkey_registered:
            try:
                keyboard.remove_hotkey(self._hotkey_registered)
            except (KeyError, ValueError):
                pass
            self._hotkey_registered = ""

    def suspend_hotkey(self):
        """تعلیق موقت هات‌کی (حین capture در تنظیمات)."""
        self._unregister_hotkey()

    def resume_hotkey(self):
        """بازگردانی هات‌کی بعد از تعلیق (پس از بستن تنظیمات)."""
        self.apply_hotkey()

    # ---------- ضبط (thread-safe) ----------
    def toggle_recording(self):
        """کالبک hotkey/tray — فقط یک worker می‌سازد و فوراً برمی‌گردد."""
        threading.Thread(target=self._toggle_worker, daemon=True).start()

    def _toggle_worker(self):
        with self._state_lock:
            if self.state in (STATE_LOADING, STATE_STARTING, STATE_TRANSCRIBING):
                return
            if self.state == STATE_RECORDING:
                self.state = STATE_TRANSCRIBING
                stopping = True
            else:
                self.state = STATE_STARTING
                stopping = False
        if stopping:
            self._stop_impl()
        else:
            self._start_impl()

    def _start_impl(self):
        if self.engine is None:
            with self._state_lock:
                self.state = STATE_IDLE if self.live else STATE_LOADING
            self._ui_set_state(self.state)
            return
        # صبر برای تشخیص میکروفون (معمولاً در استارتاپ تمام شده)
        if not self._device_ready.wait(timeout=8):
            pass  # با دستگاه پیش‌فرض ادامه می‌دهیم
        rec = Recorder(device=self.device)
        try:
            rec.start()
        except Exception as e:
            with self._state_lock:
                self.state = STATE_IDLE
            self._ui_set_state(STATE_IDLE)
            self._notify(f"میکروفون باز نشد: {str(e)[:40]}")
            return
        self.recorder = rec
        with self._state_lock:
            self.state = STATE_RECORDING
        self._silence_t0 = None
        self._ui_set_state(STATE_RECORDING)
        if self.cfg.get("sound_feedback"):
            _beep(start=True)
        if self.cfg.get("overlay_enabled") and self.overlay:
            self.ui_q.put(("show", None))
            self.ui_q.put(("font", int(self.cfg.get("overlay_font_size") or 15)))
        threading.Thread(target=self._partial_loop, daemon=True).start()

    def _stop_impl(self):
        rec = self.recorder
        self.recorder = None
        if rec is None:
            with self._state_lock:
                self.state = STATE_IDLE if self.live else STATE_LOADING
            self._ui_set_state(self.state)
            return
        rec.stop()
        if self.cfg.get("sound_feedback"):
            _beep(start=False)
        # overlay بلافاصله بسته نمی‌شود؛ در حالت «در حال تشخیص» می‌ماند
        self.ui_q.put(("processing", None))
        self._ui_set_state(STATE_TRANSCRIBING)
        threading.Thread(target=self._finish, args=(rec,), daemon=True).start()

    def _partial_loop(self):
        """ترنسکرایپ زنده — در thread خودش، نتیجه از طریق صف به UI می‌رود."""
        auto_stop = float(self.cfg.get("auto_stop_sec") or 0)
        speech_seen = False
        while self._running:
            with self._state_lock:
                if self.state != STATE_RECORDING:
                    return
            t0 = time.perf_counter()
            rec = self.recorder
            if rec is not None and self.live is not None:
                try:
                    buf = rec.get_tail_16k(self.live.window_sec)
                    text = self.live.partial(buf)
                    self.ui_q.put(("text", text))
                    # توقف خودکار پس از سکوت — فقط اگر قبلاً صدایی شنیده شده
                    if auto_stop > 0:
                        recent = buf[-int(1.5 * 16000):]
                        rms = (float(np.sqrt((recent.astype(np.float64) ** 2).mean()))
                               if recent.size else 0.0)
                        now = time.perf_counter()
                        if rms > SILENCE_RMS:
                            speech_seen = True
                            self._silence_t0 = None
                        elif speech_seen:
                            if self._silence_t0 is None:
                                self._silence_t0 = now
                            elif now - self._silence_t0 >= auto_stop:
                                with self._state_lock:
                                    if self.state != STATE_RECORDING:
                                        return
                                    self.state = STATE_TRANSCRIBING
                                self._stop_impl()
                                return
                except Exception:
                    pass
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.05, PARTIAL_INTERVAL - elapsed))

    def _finish(self, rec: Recorder):
        try:
            buf = rec.get_buffer_16k()
            text = self.live.final(buf) if self.live else ""
        except Exception:
            text = ""
        with self._state_lock:
            self.state = STATE_IDLE if self.live else STATE_LOADING
        self._ui_set_state(self.state)
        # بستن overlay بعد از ترنسکرایپ نهایی (قبل از درج)
        self.ui_q.put(("hide", None))
        if not text:
            self._notify("صدایی تشخیص داده نشد — میکروفون یا سطح ورودی را بررسی کنید")
            return
        time.sleep(0.15)  # فرصت برای بازگشت فوکوس به برنامه مقصد
        self._insert(text)

    def _insert(self, text: str):
        method = self.cfg.get("paste_method")
        restore = bool(self.cfg.get("restore_clipboard"))
        if self.cfg.get("voice_commands"):
            segments = voice_commands.parse(text)
            voice_commands.execute(
                segments,
                lambda t: insert_text(t, method, restore),
                self._send_key,
            )
        else:
            insert_text(text, method, restore)

    def _send_key(self, name: str):
        if name == "enter":
            send_key("enter")
        elif name == "delete_word":
            send_key("ctrl+backspace")

    # ---------- حلقه UI (فقط thread اصلی) ----------
    def _ui_loop(self):
        last_level_t = 0.0
        last_cstate = ""
        while self._running:
            # ۱) اجرای درخواست‌های threadهای دیگر
            while True:
                try:
                    op, arg = self.ui_q.get_nowait()
                except queue.Empty:
                    break
                if not self._running:
                    break
                try:
                    if op == "show":
                        self.overlay.show()
                    elif op == "font":
                        self.overlay.set_font_size(arg)
                    elif op == "show_window":
                        self.control.show_window()
                    elif op == "hide":
                        self.overlay.hide()
                    elif op == "processing":
                        self.overlay.set_processing()
                    elif op == "text":
                        self.overlay.update_text(arg)
                    elif op == "settings":
                        self._open_settings_ui()
                    elif op == "cstate":
                        self.control.set_state(arg, self.cfg.get("hotkey"))
                    elif op == "error":
                        self.control.set_error(arg)
                except Exception:
                    pass
            # ۲) نوار سطح صدا (هر ۱۰۰ms)
            now = time.perf_counter()
            if now - last_level_t > 0.1:
                if self.state == STATE_RECORDING:
                    self.overlay.update_level(self._recent_rms())
                else:
                    self.overlay.update_level(0.0)
                last_level_t = now
            # ۳) پمپ رویدادهای Tkinter
            try:
                self.overlay.tick()
                # sync وضعیت پنجره کنترل (اگر چیزی از صف جا مانده)
                with self._state_lock:
                    st = self.state
                if st != last_cstate and st in ("loading", "idle", "recording", "transcribing"):
                    self.control.set_state(st, self.cfg.get("hotkey"))
                    last_cstate = st
            except Exception:
                pass
            time.sleep(0.05)

    def _recent_rms(self) -> float:
        rec = self.recorder
        if rec is None:
            return 0.0
        return rec.recent_rms()

    def _open_settings_ui(self):
        from app.settings_ui import open_settings

        open_settings(self.overlay.root, self)

    def apply_config(self):
        """بعد از ذخیره‌ی تنظیمات — hotkey و autostart را اعمال کن."""
        # پنجره تنظیمات کپی خودش را روی دیسک می‌نویسد؛ تنظیمات تازه باید از دیسک خوانده شود
        self.cfg = Config.load()
        self.apply_hotkey()
        set_autostart(bool(self.cfg.get("autostart")))
        dev = self.cfg.get("input_device")
        if dev is None:
            # بازگشت به «خودکار» — تشخیص مجدد دستگاه در پس‌زمینه
            self.device = None
            self._device_ready.clear()
            threading.Thread(target=self._detect_device, daemon=True).start()
        else:
            self.device = int(dev)
            self._device_ready.set()
        # پنجره کنترل hint کلید میانبر را تازه کند
        self._ui_set_state(self.state)

    def quit(self):
        self._running = False
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception:
                pass
        if self._tray:
            try:
                self._tray.stop()
            except Exception:
                pass


def main():
    app = App()
    try:
        app.start()
    finally:
        # بستن Tkinter فقط در thread اصلی
        if app.overlay:
            try:
                app.overlay.root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
