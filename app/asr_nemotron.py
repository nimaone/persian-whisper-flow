# -*- coding: utf-8 -*-
"""موتور Nemotron ASR — مدل transducer استریمینگ فارسی (۶۰۰M).

فاین‌تیون `nemotron-3.5-asr-streaming-0.6b` روی فارسی، تبدیل‌شده به ONNX
با اسکریپت رسمی sherpa-onnx. فایل‌های ONNX (int8) از مخزن
`Nimaone/nemotron-asr-streaming-farsi-onnx` دانلود می‌شوند
(کاتالوگ: `app/models_catalog.py`).

دقت: ‏~۵ امتیاز WER بهتر از شنوا کوچیک روی صوت محاوره‌ای؛
سرعت: ‏~۴ برابر کندتر (RTF ‏~0.32 روی ‏۴ تاپ) — برای دیکته قابل استفاده.

دو تفاوت ساختاری با AsrEngine (CTC):
- pyctcdecode/هات‌وورد روی transducer کار نمی‌کند → `transcribe_with_details`
  فقط متن ساده برمی‌گرداند.
- ورودی فیچر ‏۱۲۸ mel است (نه ۸۰) و زبان باید با `fa-IR` ست شود.

استریم پیوسته: `begin_stream()` یک نشست دائمی می‌سازد که کش‌های انکودر را
بین feedها حمل می‌کند (منطق استریم cache-aware کارت مدل) — هر پنجرهٔ
~۱.۱۲s فقط یک بار انکد می‌شود؛ `transcribe` همان مسیر یک‌ضربِ هر بار از صفر است.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from .asr import AsrHypothesis


def _resolve_export_dir() -> Path:
    """پوشه‌ی فایل‌های ONNX — env override، وگرنه پوشه‌ی مدل فعال."""
    env = os.environ.get("DICTEYAR_NEMOTRON_DIR")
    if env:
        return Path(env)
    from app.config import model_dir
    from app.models_catalog import get as get_model

    return get_model("nemotron-farsi").resolve_dir(model_dir())


class NemotronAsrEngine:
    """موتور transducer استریمینگ — هم‌رابط با AsrEngine (duck-typed)."""

    def __init__(self, export_dir: str | Path | None = None, num_threads: int = 4):
        import sherpa_onnx

        self._dir = Path(export_dir) if export_dir else _resolve_export_dir()
        self._num_threads = num_threads

        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self._dir / "tokens.txt"),
            encoder=self._pick("encoder"),
            decoder=self._pick("decoder"),
            joiner=self._pick("joiner"),
            num_threads=num_threads,
            feature_dim=128,
        )
        self._lock = threading.Lock()
        print(
            f"[asr] Nemotron ({self._dir.name}) در "
            f"{time.perf_counter() - t0:.2f} ثانیه بارگذاری شد"
        )

    def _pick(self, name: str) -> str:
        """int8 پیش‌فرض است (لود سریع‌تر، حافظه کمتر)؛ fp32 با env فعال می‌شود."""
        fp32 = self._dir / f"{name}.onnx"
        int8 = self._dir / f"{name}.int8.onnx"
        if os.environ.get("DICTEYAR_NEMOTRON_PRECISION", "").lower() != "fp32" and int8.is_file():
            return str(int8)
        if fp32.is_file():
            return str(fp32)
        raise FileNotFoundError(f"{name}.onnx در {self._dir} یافت نشد")

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """ترنسکرایپ موج float32 [-1..1] → متن فارسی (دیکد استریم cache-aware)."""
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            try:
                stream.set_option("language", "fa-IR")
            except Exception:
                pass  # بدون option هم پرامپت auto استفاده می‌شود
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            stream.input_finished()
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            res = self._recognizer.get_result(stream)
            text = res.text if hasattr(res, "text") else str(res)
            return text.strip()

    def transcribe_with_details(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        previous_text: str | None = None,
    ) -> AsrHypothesis:
        # transducer: امتیاز per-frame مانند CTC موجود نیست → متن ساده
        return AsrHypothesis(self.transcribe(samples, sample_rate))

    def begin_stream(self) -> "NemotronStreamSession":
        """نشست استریم پیوستهٔ تازه — برای هر دیکته یک نشست بسازید.

        در نشست، کش‌های توجه انکودر و وضعیت دیکدر بین feedها زنده می‌مانند؛
        متن فقط اضافه‌شونده است و پس از end() نشست دیگر قابل استفاده نیست.
        """
        return NemotronStreamSession(self)

    def release_detail_decoder(self):
        """برای هم‌رابطی با AsrEngine — این موتور دیکدر دومی ندارد."""

    def release(self):
        with self._lock:
            self._recognizer = None


class NemotronStreamSession:
    """نشست استریم پیوسته — یک stream دائمی sherpa برای یک دیکته.

    برخلاف `transcribe` (stream تازه در هر فراخوانی)، کش‌های توجه انکودر
    و وضعیت دیکدر RNN-T بین feedها زنده می‌مانند: هر پنجرهٔ ~۱.۱۲ ثانیه‌ای
    فقط یک بار انکد می‌شود. هزینهٔ CPU به‌جای O(تعداد tick × طول پنجره)
    به O(طول صوت) می‌رسد و متن زنده ~۲.۲ ثانیه پس از هر جمله (چانک ۱.۱۲ +
    look-ahead ۱.۱۲) آماده است. متن فقط اضافه‌شونده است.
    """

    def __init__(self, engine: NemotronAsrEngine):
        self._engine = engine
        with engine._lock:
            stream = engine._recognizer.create_stream()
            try:
                stream.set_option("language", "fa-IR")
            except Exception:
                pass  # بدون option هم پرامپت auto استفاده می‌شود
        self._stream = stream
        self._done = False

    def feed(self, samples: np.ndarray, decode: bool = True) -> str | None:
        """صوت 16k float32 جدید را به جریان اضافه می‌کند.

        decode=False (دوران سکوت): صوت فقط بافر می‌شود و None برمی‌گردد؛
        پنجره‌های انباشته در اولین feed بعدی با decode=True یکجا دیکد
        می‌شوند (پیوستگی کش حفظ می‌شود، CPU سکوت صفر است).
        خروجی decode=True: متن کل نشست تا این لحظه (ممکن است "").
        """
        eng = self._engine
        with eng._lock:
            if self._stream is None or self._done:
                return None
            samples = np.asarray(samples, dtype=np.float32)
            if samples.size:
                self._stream.accept_waveform(16000, samples)
            if not decode:
                return None
            rec = eng._recognizer
            while rec.is_ready(self._stream):
                rec.decode_stream(self._stream)
            return self._text()

    def has_pending(self) -> bool:
        """پنجرهٔ کاملِ آمادهٔ دیکد در جریان باقی مانده؟ (برای فلاش انتهای جمله)

        پنجرهٔ ناقص آخر (کمتر از یک چانک) اینجا حساب نمی‌شود — آن فقط با
        end() فلاش می‌شود؛ ولی سکوتِ بعدش توکن تولید نمی‌کند پس متن کامل است.
        """
        eng = self._engine
        with eng._lock:
            if self._stream is None or self._done:
                return False
            return eng._recognizer.is_ready(self._stream)

    def end(self) -> str:
        """فلاش انتهایی (پنجرهٔ ناقص آخر با صفر پد و دیکد می‌شود) و متن نهایی."""
        eng = self._engine
        with eng._lock:
            if self._stream is None:
                return ""
            if not self._done:
                self._stream.input_finished()
                rec = eng._recognizer
                while rec.is_ready(self._stream):
                    rec.decode_stream(self._stream)
                self._done = True
            return self._text()

    def close(self):
        """رهاکردن نشست بدون فلاش — برای لغو دیکته."""
        with self._engine._lock:
            self._stream = None

    def _text(self) -> str:
        res = self._engine._recognizer.get_result(self._stream)
        text = res.text if hasattr(res, "text") else str(res)
        return text.strip()
