"""موتور تشخیص با beam search و هات‌وورد — جایگزین اختیاری برای sherpa-onnx.

همان مدل NeMo CTC است، ولی به‌جای گری‌دی sherpa-onnx، خروجی log-probs با
onnxruntime استخراج و با pyctcdecode دیکد می‌شود تا واژه‌های حساسِ کاربر
(لیست هات‌وورد) در beam search تقویت شوند. اینترفیس با AsrEngine یکسان است
(duck-typed: transcribe(samples, sample_rate) -> str).

هشدار: sherpa-onnx 1.13.7 برای مدل‌های NeMo CTC فقط greedy_search پشتیبانی
می‌کند و پارامتر هات‌وورد آن مخصوص مدل‌های transducer است؛ بنابراین مسیر
فعلی (session مستقیم ONNX + fbank کالدی‌استایل + pyctcdecode) تنها راه
هات‌وورد روی این مدل است.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

# kenlm (وابستگی اختیاری pyctcdecode) نصب نیست و برای هات‌ووردِ بدون LM هم
# لازم نیست؛ هشدار import آن را بی‌صدا می‌کنیم تا در کنسول اپ اسپم نشود
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

from pyctcdecode import Alphabet, BeamSearchDecoderCTC

# وزن ۱۰: در تست‌ها همه واژه‌های قابل‌اصلاح را درست کرد و کنترل‌های
# false-positive (مثل «آب می‌خورم» وقتی هات‌وورد «اپ» است) تا وزن ۳۰ سالم ماندند
DEFAULT_BEAM_WIDTH = 100
DEFAULT_HOTWORD_WEIGHT = 10.0


def load_labels(tokens_path: str | Path) -> list[str]:
    """tokens.txt (نماد شناسه) → لیست ترتیبی؛ blank مدل به رشته خالی
    تبدیل می‌شود چون pyctcdecode رشته خالی را به‌عنوان blank CTC می‌شناسد."""
    toks: dict[int, str] = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2:
                toks[int(parts[1])] = parts[0]
    return ["" if toks[i] == "<blk>" else toks[i] for i in range(len(toks))]


def kaldi_fbank(x: np.ndarray, sr: int = 16000, num_mel: int = 80) -> np.ndarray:
    """ویژگی fbank کالدی‌استایل — همان تنظیمات پیشفرض kaldi-native-fbank
    که sherpa-onnx برای مدل‌های NeMo استفاده می‌کند:
    پنجره پووی ۲۵ms، گام ۱۰ms، پیش‌تأکید ۰.۹۷، حذف DC، طیف توان،
    ۸۰ پوشش مل ۲۰Hz..نیکوییست، لگاریتم با کف ۱.۱۹e-7. خروجی [T, 80]."""
    frame_len, frame_shift = 400, 160
    if sr != 16000:
        n_out = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), n_out, endpoint=False),
                      np.arange(len(x)), x).astype(np.float64)
    x = x.astype(np.float64)
    n_frames = 1 + max(0, (len(x) - frame_len) // frame_shift)
    idx = np.arange(frame_len)[None, :] + frame_shift * np.arange(n_frames)[:, None]
    frames = x[idx]
    pre = np.concatenate([frames[:, :1], frames], axis=1)
    frames = pre[:, 1:] - 0.97 * pre[:, :-1]
    frames = frames - frames.mean(axis=1, keepdims=True)
    n = np.arange(frame_len)
    povey = np.power(0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_len - 1)), 0.85)
    frames = frames * povey
    fft = np.fft.rfft(frames, n=512)
    power = (fft.real ** 2 + fft.imag ** 2)[:, :257]

    def hz2mel(f):
        return 1127.0 * np.log1p(np.asarray(f) / 700.0)

    mel_pts = np.linspace(hz2mel(20.0), hz2mel(16000 / 2), num_mel + 2)
    hz_pts = 700.0 * np.expm1(mel_pts / 1127.0)
    bins = np.floor((512 + 1) * hz_pts / 16000.0).astype(int)
    fbank = np.zeros((num_mel, 257))
    for m in range(1, num_mel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c += 1
        if r == c:
            r += 1
        for k in range(l, c):
            fbank[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            fbank[m - 1, k] = (r - k) / (r - c)
    feat = power @ fbank.T
    return np.log(np.maximum(feat, 1.19e-7))


class HotwordAsrEngine:
    """موتور CTC با beam search و تقویت هات‌وورد — thread-safe، هم‌اینترفیس AsrEngine."""

    def __init__(
        self,
        model_dir: str | Path = "model",
        num_threads: int = 4,
        hotwords: list[str] | None = None,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ):
        model_dir = Path(model_dir)
        self._labels = load_labels(model_dir / "tokens.txt")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._decoder = BeamSearchDecoderCTC(Alphabet(self._labels, is_bpe=True))
        self._beam_width = beam_width
        self._hotword_weight = hotword_weight
        self.hotwords = list(hotwords or [])
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """ترنسکرایپ موج صوتی float32 [-1..1] → متن فارسی (با تقویت هات‌وورد)."""
        if samples.size == 0:
            return ""
        with self._lock:
            feat = kaldi_fbank(samples, sample_rate)
            log_probs, _ = self._session.run(
                None,
                {"audio_signal": feat.T[None].astype(np.float32),
                 "length": np.array([feat.shape[0]], dtype=np.int64)},
            )
            return self._decoder.decode(
                log_probs[0],
                beam_width=self._beam_width,
                hotwords=self.hotwords or None,
                hotword_weight=self._hotword_weight,
            ).strip()


def load_hotword_engine(
    model_dir: str | Path = "model",
    num_threads: int = 4,
    hotwords: list[str] | None = None,
) -> HotwordAsrEngine:
    import time

    t0 = time.perf_counter()
    eng = HotwordAsrEngine(model_dir, num_threads=num_threads, hotwords=hotwords)
    t1 = time.perf_counter()
    print(f"[asr] موتور هات‌وورد در {t1 - t0:.2f} ثانیه بارگذاری شد")
    return eng
