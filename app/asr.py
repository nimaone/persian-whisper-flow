"""موتور تشخیص گفتار — بارگذاری مدل Shenava-Koochik از طریق sherpa-onnx.

مدل از نوع NeMo CTC آفلاین است؛ برای نمایش زنده، هر بار روی آخرین
بخش بافر inference مجدد اجرا می‌شود و پیشوند پایدار قفل می‌شود.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import sherpa_onnx


class AsrEngine:
    """بارگذاری مدل و ترنسکرایپ — thread-safe."""

    def __init__(self, model_dir: str | Path = "model", num_threads: int = 4):
        model_dir = Path(model_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            debug=False,
        )
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """ترنسکرایپ موج صوتی float32 [-1..1] → متن فارسی."""
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            self._recognizer.decode_stream(stream)
            return stream.result.text.strip()


class LiveTranscriber:
    """ترنسکرایپ تدریجی روی بافر در حال رشد، با قفل پیشوند پایدار.

    چون مدل آفلاین است، متن «زنده» با اجرای مجدد inference روی
    آخرین حداکثر `window_sec` ثانیه‌ی بافر ساخته می‌شود؛ بخشی از
    متن که در اجرای قبل نیز یکسان بود به‌عنوان متن قطعی نگه داشته
    می‌شود تا چشمک‌زن تغییر نکند.
    """

    def __init__(self, engine: AsrEngine, window_sec: float = 10.0):
        self.engine = engine
        self.window_sec = window_sec

    def partial(self, buffer16k: np.ndarray) -> str:
        """ترنسکرایپِ حین ضبط روی پنجره‌ی انتهای بافر."""
        win = buffer16k[-int(self.window_sec * 16000):]
        if win.size < 1600:  # کمتر از 0.1s صدا
            return ""
        try:
            return self.engine.transcribe(win)
        except Exception:
            return ""

    def final(self, buffer16k: np.ndarray) -> str:
        """ترنسکرایپ نهایی روی کل بافر بعد از توقف ضبط."""
        if buffer16k.size == 0:
            return ""
        try:
            return self.engine.transcribe(buffer16k)
        except Exception as e:
            raise RuntimeError(f"ترنسکرایپ نهایی ناموفق: {e}") from e


def load_engine(model_dir: str | Path = "model", num_threads: int = 4) -> AsrEngine:
    t0 = time.perf_counter()
    eng = AsrEngine(model_dir, num_threads=num_threads)
    t1 = time.perf_counter()
    print(f"[asr] مدل در {t1 - t0:.2f} ثانیه بارگذاری شد")
    return eng
