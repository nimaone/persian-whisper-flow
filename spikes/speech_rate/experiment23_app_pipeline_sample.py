"""تست نهایی پایپ‌لاین خود اپ روی sample — بدون میکروفون، بدون درج سیستم.

به‌جای شبیه‌سازی LiveTranscriber، خودِ App اپ اجرا می‌شود:
- همان _make_engine، همان LiveTranscriber با تنظیمات واقعی config
- همان _partial_loop (PARTIAL_INTERVAL=0.8، SpeechGate، ITN، rejoin)
- همان _finish → live.final روی کل بافر + پس‌پردازش اپ
تنها تفاوت‌ها:
- Recorder با FakeRecorder عوض می‌شود: صدا از sample_video16k.wav
  بلوک‌بلوک (۱۰۰ms) با زمان واقعی تزریق می‌شود — همان اینترفیس.
- _insert بجای تایپ در پنجره‌ی فعال، متن را در خروجی جمع می‌کند.
- UI/tray/hotkey وجود ندارد (headless) — اینها فقط رابط‌اند نه پایپ‌لاین.

اجرا (حالت صریح می‌خواهد، وگرنه هر چه در تنظیمات کاربر است اجرا می‌شود):
  python experiment23_app_pipeline_sample.py stable   # حالت پایدار (فقط در حافظه)
  python experiment23_app_pipeline_sample.py plain    # حالت معمولی
  python experiment23_app_pipeline_sample.py          # تنظیمات واقعی کاربر

سنجه‌ها: متن زنده‌ی هر tick (نمایشی که کاربر می‌دید)، متن نهایی
(آنچه درج می‌شد)، و WER هر دو نسبت به sample/reference.txt.
"""
from __future__ import annotations

import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REF = ROOT / "sample" / "reference.txt"


def norm_words(text: str) -> list[str]:
    out = []
    for w in text.replace("\u200c", "").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if w:
            out.append(w)
    return out


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    m = SequenceMatcher(None, hyp, ref, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    return len(ref) - matched, len(ref)


class FakeRecorder:
    """اینترفیس مشابه app.recorder.Recorder — صدا از فایل، بلوک‌بلوک."""

    def __init__(self, wav: np.ndarray, block_sec: float = 0.1):
        self.wav = wav
        self.block = int(block_sec * 16000)
        self._pos = 0
        self._out_parts: list[np.ndarray] = []
        self._t0 = time.perf_counter()
        self._stopped = threading.Event()
        # تزریق بلوک‌ها با زمان واقعی — مثل callback sounddevice
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped.is_set() and self._pos < self.wav.size:
            now = time.perf_counter()
            want = int((now - self._t0) * 16000)
            while self._pos < min(want, self.wav.size):
                chunk = self.wav[self._pos:self._pos + self.block]
                self._out_parts.append(chunk.astype(np.float32))
                self._pos += chunk.size
            time.sleep(0.02)

    # ---- اینترفیس Recorder ----
    def get_tail_16k(self, window_sec: float) -> np.ndarray:
        need = int(window_sec * 16000)
        parts = self._out_parts
        total = 0
        idx = len(parts)
        while idx > 0 and total < need:
            idx -= 1
            total += parts[idx].size
        return np.concatenate(parts[idx:]) if idx < len(parts) else np.zeros(0, np.float32)

    def get_buffer_16k(self) -> np.ndarray:
        return np.concatenate(self._out_parts) if self._out_parts \
            else np.zeros(0, np.float32)

    def duration_sec(self) -> float:
        return sum(p.size for p in self._out_parts) / 16000.0

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)


def main():
    import app.main as m

    # حالت صریح از خط فرمان — بی‌آرگومان = همان تنظیمات واقعی کاربر.
    # قبلاً راهی برای روشن‌کردن اجباری حالت پایدار نبود و تست روی
    # تنظیمات کاربر (خاموش) می‌افتاد و کسی متوجه نمی‌شد.
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ""
    if mode not in ("", "stable", "plain"):
        print("آرگومان نامعتبر. درست: stable | plain | بدون آرگومان")
        return

    wav, sr = sf.read(Path(__file__).resolve().parent / "sample_video16k.wav",
                      dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000
    dur = len(wav) / 16000

    app = m.App()   # __init__ فقط config/queue — بدون UI
    if mode:
        app.cfg.set("stable_live", mode == "stable")   # فقط در حافظه؛ settings.json دست‌نخورده
    source = "آرگومان خط فرمان" if mode else "تنظیمات کاربر"
    print(f"stable_live = {bool(app.cfg.get('stable_live'))} ({source}) | "
          f"persian_itn = {bool(app.cfg.get('persian_itn'))} | "
          f"rejoin_prefixes = {bool(app.cfg.get('rejoin_prefixes'))}")

    # موتور و live با همان مسیر اپ
    app.engine = app._make_engine()
    app.live = m.LiveTranscriber(
        app.engine, stable_live=bool(app.cfg.get("stable_live")))
    app.live.configure(bool(app.cfg.get("stable_live")))
    app.live.reset()

    # مهار درج در سیستم — فقط جمع‌آوری
    captured = {"live": [], "final": None}

    def fake_insert(text, *a, **k):
        captured["final"] = text
        print(f"\n[درج نهایی] {text}")

    m.insert_text = fake_insert
    app._send_key = lambda name: None

    rec = FakeRecorder(wav)
    app.recorder = rec
    with app._state_lock:
        app.state = m.STATE_RECORDING

    # همان حلقه‌ی زنده‌ی اپ — بدون تغییر
    live_thread = threading.Thread(target=app._partial_loop, daemon=True)

    # ربایت متن زنده از صف UI (همان مسیری که overlay می‌رفت)
    texts = []

    def ui_pump():
        while True:
            try:
                op, arg = app.ui_q.get(timeout=0.1)
            except Exception:
                if threading.current_thread() is not pump_thread:
                    return
                continue
            if op == "text":
                texts.append(arg)
            if op == "hide":
                return

    pump_thread = threading.Thread(target=ui_pump, daemon=True)

    live_thread.start()
    pump_thread.start()

    # اجرای ضبط به اندازه‌ی طول فایل + ۲ ثانیه
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur + 2.0:
        time.sleep(0.5)

    # همان مسیر توقف اپ
    app._stop_impl()
    # ترنسکرایپ نهایی چند ثانیه طول می‌کشد — تا درج کامل صبر کن
    t_stop = time.perf_counter()
    while captured["final"] is None and time.perf_counter() - t_stop < 60:
        time.sleep(0.5)
    pump_thread.join(timeout=3.0)

    ref = norm_words(REF.read_text(encoding="utf-8"))
    print(f"\n{'=' * 60}")
    print(f"حالت: {'پایدار' if app.cfg.get('stable_live') else 'معمولی'} ({source})")
    print(f"تعداد متن‌های زنده‌ی نمایش‌داده‌شده: {len(texts)}")
    if texts:
        print(f"متن زنده‌ی آخر (چیزی که کاربر لحظه‌ی توقف می‌دید):")
        print(f"  {texts[-1]}")
        e, n = wer(ref, norm_words(texts[-1]))
        print(f"  WER زنده‌ی آخر نسبت به مرجع: {100 * e / max(1, n):.1f}% ({e}/{n})")
    if captured["final"]:
        e, n = wer(ref, norm_words(captured["final"]))
        print(f"متن نهایی درج‌شده: WER {100 * e / max(1, n):.1f}% ({e}/{n})")


if __name__ == "__main__":
    main()
