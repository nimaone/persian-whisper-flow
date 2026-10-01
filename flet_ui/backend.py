"""بک‌اند ضبط/ترنسکرایب برای UI فلت — قرینه‌ی App در app/main.py بدون Tk.

منطق دقیقاً همان نسخه‌ی CTk است: لود مدل در پس‌زمینه، هات‌کی سراسری،
ضبط با Recorder، ترنسکرایب زنده، توقف خودکار با سکوت، درج متن با
کلیپ‌بورد/تایپ + فرمان‌های صوتی. تنها تفاوت: به‌جای صف UI و Tkinter،
خروجی‌ها با صدا زدن متدهای پنجره (set_state/set_status/set_error)
به UI می‌روند — تغییر props از هر تردی امن است، اما ارسال پچ به
کلاینت باید از ایونت‌لوپ فلت انجام شود که ControlWindow خودش با
page.run_task هندل می‌کند (پچ از ترد فرعی به صف ارسال نمی‌رسد).

واحدهای بیرونی (engine/recorder/insert) تزریق‌پذیرند تا تست خودکار
بدون مدل واقعی و میکروفون ممکن باشد.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app import fa_post, persian_itn, voice_commands
from app.asr import DirectCtcAsrEngine, LiveTranscriber, SpeechGate, load_engine
from app.config import Config, model_dir, set_autostart
from app.paster import insert_text, send_key
from app.recorder import Recorder, detect_best_device, resolve_pinned_device

STATE_LOADING = "loading"
STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_RECORDING = "recording"
STATE_TRANSCRIBING = "transcribing"

PARTIAL_INTERVAL = 0.8  # ثانیه بین ترنسکرایپ‌های زنده
SILENCE_RMS = 0.003     # آستانه سکوت برای توقف خودکار
SPEECH_GATE_HANGOVER = 1.5  # ثانیه decode اضافه پس از آخرین صدا

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _device_fallback_chain(device: int | None) -> list[int | None]:
    """زنجیره‌ی تلاش برای باز کردن میکروفون: دستگاه انتخابی → مسیرهای دیگر
    Host API همان میکروفون فیزیکی → پیش‌فرض سیستم. (قرینه‌ی app/main.py)

    بعضی مسیرها (WDM-KS، میکروفون مجازی بی‌برنامه) باز نمی‌شوند؛
    شکست یکی نباید دیکته را کلاً بیندازد.
    """
    devs: list[int | None] = []
    if device is not None:
        devs.append(device)
        try:
            from app.recorder import current_input_devices, device_siblings
            devs.extend(e["index"] for e in
                        device_siblings(current_input_devices(), device))
        except Exception:
            pass
    if None not in devs:
        devs.append(None)  # پیش‌فرض سیستم — آخرین جان پناه
    return list(dict.fromkeys(devs))


def _beep(start: bool):
    """بوق کوتاه شروع/پایان ضبط — بی‌صدا در خطا (تنظیم sound_feedback)."""
    try:
        import winsound
        winsound.Beep(880 if start else 660, 60)
    except Exception:
        pass


class DictationApp:
    """همان چرخه‌ی حیات App نسخه CTk — UI فلت به‌جای Tkinter."""

    def __init__(self, win, cfg: Config | None = None,
                 engine_loader=None, recorder_factory=None,
                 insert_fn=None, send_key_fn=None, device_probe=None,
                 register_hotkey: bool = True, with_overlay: bool = True):
        self.win = win
        self.cfg = cfg or Config.load()
        self.state = STATE_LOADING
        self.device = self.cfg.get("input_device")
        self.engine = None
        self.live: LiveTranscriber | None = None
        self.recorder = None

        # تزریق وابستگی — برای تست با فیک
        self._engine_loader = engine_loader or self._default_engine_loader
        self._recorder_factory = recorder_factory or (lambda device: Recorder(device=device))
        self._insert_fn = insert_fn or insert_text
        self._send_key_fn = send_key_fn or send_key
        self._device_probe = device_probe or detect_best_device
        self._register_hotkey_enabled = register_hotkey

        self._state_lock = threading.Lock()
        self._device_ready = threading.Event()
        self._running = True
        self._silence_t0 = None  # زمان شروع سکوت فعلی (برای توقف خودکار)
        self._engine_dirty = False  # هات‌وورد عوض شده — پس از ضبط rebuild شود
        self._hotkey_registered = ""
        self._settings_proc: subprocess.Popen | None = None

        # پنجره زنده (Overlay) — Tk فقط از thread خودش؛ فرمان‌ها با صف
        self._ov_q: queue.Queue = queue.Queue()
        self._overlay = None
        self._overlay_ready = threading.Event()
        if with_overlay:
            threading.Thread(target=self._overlay_loop, daemon=True).start()

    # ---------- پنجره زنده (thread اختصاصی Tk) ----------
    def _overlay_loop(self):
        """قرینه‌ی _ui_loop در app/main.py — همه‌ی دستورات Tk از همین thread."""
        try:
            from app.overlay import Overlay
            self._overlay = Overlay()
        except Exception:
            self._overlay_ready.set()
            return  # بدون overlay هم برنامه کار می‌کند
        self._overlay_ready.set()
        tail_n = 0  # واژه‌های قطعی‌نشده — «tail» همیشه پیش از «text» می‌آید
        while self._running:
            try:
                while True:
                    op, arg = self._ov_q.get_nowait()
                    if op == "show":
                        tail_n = 0
                        self._overlay.set_font_size(
                            int(self.cfg.get("overlay_font_size") or 15))
                        self._overlay.show()
                    elif op == "tail":
                        tail_n = int(arg or 0)
                    elif op == "text":
                        self._overlay.update_text(arg, tail_n)
                    elif op == "processing":
                        self._overlay.set_processing()
                    elif op == "hide":
                        self._overlay.hide()
                    elif op == "level":
                        self._overlay.update_level(arg)
            except queue.Empty:
                pass
            try:
                self._overlay.tick()
            except Exception:
                pass
            time.sleep(0.05)

    def _ov(self, op, arg=None):
        """فرمان به پنجره زنده — بی‌صدا اگر thread آن بالا نیامده."""
        if not self._overlay_ready.wait(timeout=3):
            return
        if self._overlay is not None and self._running:
            self._ov_q.put((op, arg))

    # ---------- راه‌اندازی ----------
    def start(self):
        self._ui_set_state(self.state)
        if self._register_hotkey_enabled:
            self.apply_hotkey()
        # لود مدل و تشخیص میکروفون — هر دو در پس‌زمینه و موازی
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._detect_device, daemon=True).start()

    def _hotwords(self) -> list[str]:
        try:
            words = list(self.cfg.get("hotwords") or [])
        except Exception:
            words = []
        return [str(w) for w in words if len(str(w).strip()) >= 2]

    def _default_engine_loader(self):
        """موتور متناسب با تنظیمات: هات‌وورد (beam) یا عادی — قرینه‌ی App."""
        if self.cfg.get("hotword_boost") and self._hotwords():
            try:
                from app.hotword_asr import load_hotword_engine
                return load_hotword_engine(
                    model_dir=model_dir(),
                    num_threads=int(self.cfg.get("num_threads") or 4),
                    hotwords=self._hotwords(),
                )
            except Exception as e:
                self._notify(f"حالت واژه‌های حساس فعال نشد؛ موتور عادی: {str(e)[:60]}")
        # اگر متن پایدار خواسته شده، به‌جای sherpa + session دوم،
        # یک session ONNX برای هم transcribe و هم امتیازدهی لود می‌کنیم.
        if bool(self.cfg.get("stable_live")):
            try:
                return DirectCtcAsrEngine(
                    model_dir=model_dir(),
                    num_threads=int(self.cfg.get("num_threads") or 4),
                    beam_width=2,
                )
            except Exception as e:
                self._notify(f"موتور متن پایدار فعال نشد؛ موتور عادی: {str(e)[:60]}")
        return load_engine(model_dir=model_dir(),
                           num_threads=int(self.cfg.get("num_threads") or 4))

    def _load_model(self):
        for attempt in range(3):
            try:
                self.engine = self._engine_loader()
                self.live = LiveTranscriber(
                    self.engine, stable_live=bool(self.cfg.get("stable_live")))
                with self._state_lock:
                    if self.state in (STATE_LOADING, STATE_STARTING):
                        self.state = STATE_IDLE
                self._ui_set_state(self.state)
                return
            except Exception as e:
                if attempt == 2:
                    self._set_error(f"لود مدل ناموفق: {str(e)[:40]}")
                    return
                self._notify(f"لود مدل ناموفق؛ تلاش مجدد ({attempt + 2}/3)…")
                time.sleep(5)

    def _detect_device(self):
        if self.device is not None:
            self._device_ready.set()
            return
        # دستگاه پین‌شده با کلید پایدار اعتبارسنجی می‌شود — ایندکس خام
        # بین بوت‌ها جابه‌جا می‌شود؛ اگر دستگاه پیدا نبود، خودکار
        resolved = resolve_pinned_device(
            self.cfg.get("input_device"), self.cfg.get("input_device_key"))
        self.device = resolved if resolved is not None else self._device_probe()
        self._device_ready.set()

    # ---------- اتصال به UI (آپدیت Flet از هر thread امن است) ----------
    def _ui_set_state(self, state: str):
        try:
            self.win.set_state(state, self.cfg.get("hotkey"))
        except Exception:
            pass  # پنجره بسته شده

    def _set_status(self, text: str):
        try:
            self.win.set_status(text)
        except Exception:
            pass

    def _set_error(self, msg: str):
        try:
            self.win.set_error(msg)
        except Exception:
            pass

    def _notify(self, text: str):
        # معادل tray notify — فعلاً در نوار وضعیت پنجره کنترل
        self._set_status(text)

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
        import keyboard

        if self._hotkey_registered:
            try:
                keyboard.remove_hotkey(self._hotkey_registered)
            except (KeyError, ValueError):
                pass
            self._hotkey_registered = ""

    def suspend_hotkey(self):
        """تعلیق موقت هات‌کی (حین باز بودن تنظیمات / capture کلید)."""
        self._unregister_hotkey()

    def resume_hotkey(self):
        """بازگردانی هات‌کی بعد از تعلیق."""
        self.apply_hotkey()

    # ---------- ضبط (thread-safe) ----------
    def toggle_recording(self):
        """کالبک hotkey/دکمه — فقط یک worker می‌سازد و فوراً برمی‌گردد."""
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
        if self.device is None:
            # تشخیص پس‌زمینه هنوز تمام نشده — همین‌جا حل می‌کنیم
            try:
                self.device = self._device_probe()
            except Exception:
                self.device = None
        # زنجیره‌ی جایگزین: شکست یک مسیر نباید دیکته را کلاً بیندازد —
        # بعضی مسیرها (WDM-KS، میکروفون مجازی بی‌برنامه) باز نمی‌شوند
        rec = None
        last_err: Exception | None = None
        for dev in _device_fallback_chain(self.device):
            r = self._recorder_factory(dev)
            try:
                r.start()
                rec = r
                self.device = dev  # دستگاهی که واقعاً باز شد
                break
            except Exception as e:
                last_err = e
        if rec is None:
            with self._state_lock:
                self.state = STATE_IDLE
            self._ui_set_state(STATE_IDLE)
            msg = str(last_err)[:40] if last_err else "دستگاهی باز نشد"
            self._notify(f"میکروفون باز نشد: {msg}")
            return
        self.recorder = rec
        with self._state_lock:
            self.state = STATE_RECORDING
        self._silence_t0 = None
        if self.live is not None:
            # گزینه‌ی متن پایدار در شروع هر ضبط از تنظیمات تازه خوانده می‌شود
            self.live.configure(bool(self.cfg.get("stable_live")))
            self.live.reset()
        self._ui_set_state(STATE_RECORDING)
        if self.cfg.get("sound_feedback"):
            _beep(start=True)
        if self.cfg.get("overlay_enabled"):
            self._ov("show")
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
        self._ov("processing")
        self._ui_set_state(STATE_TRANSCRIBING)
        threading.Thread(target=self._finish, args=(rec,), daemon=True).start()

    def _partial_loop(self):
        """ترنسکرایپ زنده — نتیجه با set_status در پنجره کنترل دیده می‌شود."""
        auto_stop = float(self.cfg.get("auto_stop_sec") or 0)
        speech_seen = False
        gate = SpeechGate(SILENCE_RMS, hangover=SPEECH_GATE_HANGOVER)

        def post(t: str) -> str:
            """پس‌پردازش متن زنده — همان چیزی که درج نهایی هم می‌بیند."""
            if self.cfg.get("persian_itn"):
                t = persian_itn.normalize_text(t, min_tokens=2)
            if self.cfg.get("rejoin_prefixes"):
                t = fa_post.rejoin_prefixes(t)
            return t

        while self._running:
            with self._state_lock:
                if self.state != STATE_RECORDING:
                    return
            t0 = time.perf_counter()
            rec = self.recorder
            if rec is not None and self.live is not None:
                try:
                    buf = rec.get_tail_16k(self.live.window_sec)
                    gate_rms = float(np.sqrt(
                        (buf[-int(0.25 * 16000):].astype(np.float64) ** 2).mean()
                    )) if buf.size else 0.0
                    if gate.should_decode(gate_rms, now=t0):
                        res = self.live.partial_result(
                            buf, t_offset=max(
                                0.0, rec.duration_sec() - buf.size / 16000.0))
                        text = post(res.text)
                        # دنباله‌ی قطعی‌نشده: واژه‌های پنجره‌ی جاری — overlay
                        # این‌ها را کم‌رنگ نشان می‌دهد. تا وقتی پیشوندی قفل
                        # نشده کل نمایش یکدست می‌ماند. شمارش بعد از همان
                        # پس‌پردازشی است که روی کل متن خورد (ITN/rejoin
                        # واژه می‌چسبانند و تعداد را جابه‌جا می‌کنند).
                        tail_n = 0
                        words = res.text.split()
                        if 0 < len(res.words) < len(words):
                            tail_n = len(post(" ".join(
                                words[-len(res.words):])).split())
                        self._ov("tail", tail_n)
                        self._ov("text", text)
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
            self._ov("level", rec.recent_rms() if rec is not None else 0.0)
            time.sleep(max(0.05, PARTIAL_INTERVAL - elapsed))

    def _finish(self, rec):
        try:
            buf = rec.get_buffer_16k()
            text = self.live.final(buf) if self.live else ""
        except Exception:
            text = ""
        with self._state_lock:
            self.state = STATE_IDLE if self.live else STATE_LOADING
        self._ui_set_state(self.state)
        # اگر تنظیمات هات‌وورد وسط ضبط عوض شده بود، الان جای امن برای تعویض موتور است
        if self._engine_dirty:
            threading.Thread(target=self._rebuild_engine, daemon=True).start()
        # بستن overlay بعد از ترنسکرایب نهایی (قبل از درج)
        self._ov("hide")
        if not text:
            self._notify("صدایی تشخیص داده نشد — میکروفون یا سطح ورودی را بررسی کنید")
            return
        time.sleep(0.15)  # فرصت برای بازگشت فوکوس به برنامه مقصد
        self._insert(text)

    def _insert(self, text: str):
        if self.cfg.get("persian_itn"):
            text = persian_itn.normalize_text(text, min_tokens=2)
        if self.cfg.get("rejoin_prefixes"):
            text = fa_post.rejoin_prefixes(text)
        method = self.cfg.get("paste_method")
        restore = bool(self.cfg.get("restore_clipboard"))
        if self.cfg.get("voice_commands"):
            segments = voice_commands.parse(text)
            voice_commands.execute(
                segments,
                lambda t: self._insert_fn(t, method, restore),
                self._send_key,
            )
        else:
            self._insert_fn(text, method, restore)

    def _send_key(self, name: str):
        if name == "enter":
            self._send_key_fn("enter")
        elif name == "delete_word":
            self._send_key_fn("ctrl+backspace")

    # ---------- تنظیمات (پنجره فلت مالتی‌ویندوز در یک پروسه ندارد) ----------
    def open_settings(self):
        """تنظیمات به‌صورت پروسه‌ی جدا؛ پس از بستن، تنظیمات از دیسک اعمال می‌شود.

        هات‌کی حین باز بودن تنظیمات تعلیق می‌شود تا فشردن ترکیب میان‌بر،
        ضبط را شروع نکند (قرینه‌ی suspend/resume در open_settings نسخه CTk).
        """
        if self._settings_proc is not None and self._settings_proc.poll() is None:
            return  # از قبل باز است
        self.suspend_hotkey()
        # دستگاه فعلی به پروسه‌ی تنظیمات می‌رود — برچسب «خودکار — نام»
        # و تست صدا روی همان دستگاه (قرینه‌ی appِ در دسترس CTk)
        env = {**os.environ,
               "DIKTEYAR_DEVICE": str(self.device) if self.device is not None else ""}
        self._settings_proc = subprocess.Popen(
            [sys.executable, "-m", "flet_ui.run", "settings"],
            cwd=str(_PROJECT_ROOT), env=env,
        )
        threading.Thread(target=self._wait_settings, daemon=True).start()

    def _wait_settings(self):
        if self._settings_proc is not None:
            self._settings_proc.wait()
        if self._running:
            self.apply_config()  # هات‌کی جدید هم همین‌جا دوباره ثبت می‌شود

    def apply_config(self):
        """بعد از ذخیره‌ی تنظیمات — hotkey و autostart و دستگاه و موتور را هماهنگ کن."""
        old_key = self._engine_key  # قبل از خواندن تنظیمات جدید
        # پنجره تنظیمات کپی خودش را روی دیسک می‌نویسد؛ تنظیمات تازه باید از دیسک خوانده شود
        self.cfg = Config.load()
        new_key = self._engine_key
        if self._register_hotkey_enabled:
            self.apply_hotkey()
        set_autostart(bool(self.cfg.get("autostart")))
        dev = self.cfg.get("input_device")
        if dev is None:
            # بازگشت به «خودکار» — تشخیص مجدد دستگاه در پس‌زمینه
            self.device = None
            self._device_ready.clear()
            threading.Thread(target=self._detect_device, daemon=True).start()
        else:
            resolved = resolve_pinned_device(
                int(dev), self.cfg.get("input_device_key"))
            if resolved is None:
                # دستگاه پین‌شده دیگر موجود نیست — تشخیص خودکار
                self.device = None
                self._device_ready.clear()
                threading.Thread(target=self._detect_device, daemon=True).start()
            else:
                self.device = resolved
                self._device_ready.set()
        # تغییر حالت/لیست هات‌وورد → موتور باید عوض شود؛ وسط ضبط ممنوع، بعداً در _finish
        if new_key != old_key and self.state in (STATE_RECORDING, STATE_TRANSCRIBING):
            self._engine_dirty = True
        elif new_key != old_key and self.live is not None:
            threading.Thread(target=self._rebuild_engine, daemon=True).start()
        self._ui_set_state(self.state)

    @property
    def _engine_key(self):
        """امضای تنظیماتی که نوع موتور را تعیین می‌کند."""
        return (bool(self.cfg.get("hotword_boost")), tuple(self._hotwords()),
                bool(self.cfg.get("stable_live")))

    def _rebuild_engine(self):
        """تعویض موتور در thread پس‌زمینه — مثل استارتاپ: LOADING → IDLE."""
        with self._state_lock:
            if self.state in (STATE_RECORDING, STATE_TRANSCRIBING):
                self._engine_dirty = True
                return
            if self.state != STATE_IDLE:
                return  # در حال لود اولیه — دست نزنیم
            self.state = STATE_LOADING
        self._ui_set_state(self.state)
        try:
            engine = self._engine_loader()
            live = LiveTranscriber(
                engine, stable_live=bool(self.cfg.get("stable_live")))
        except Exception:
            with self._state_lock:
                self.state = STATE_IDLE if self.live else STATE_LOADING
            self._ui_set_state(self.state)
            return
        self.engine = engine
        self.live = live
        self._engine_dirty = False
        with self._state_lock:
            self.state = STATE_IDLE
        self._ui_set_state(self.state)

    def quit(self):
        self._running = False
        self._unregister_hotkey()
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception:
                pass