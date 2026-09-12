"""آزمایش ۱۵ — متن زنده پایدار روی صدای واقعی کاربر (sample/*.mp4).

شبیه‌سازی دقیق اپ: بافر رشدکننده، partial هر ۰٫۸s روی پنجره‌ی ۱۶s،
نمایش: طول نمایش در طول زمان (تشخیص فروپاشی/فریز)، متن نمایش نهایی،
بدترین tick. مقایسه با خروجی مسیر نهایی.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

from app.config import model_dir
from app.asr import DirectCtcAsrEngine, LiveTranscriber

WAV = Path(__file__).parent / "sample_voice16k.wav"


def main():
    wav, _ = sf.read(WAV, dtype="float32")
    print(f"طول صدا: {len(wav) / 16000:.1f}s")

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    live = LiveTranscriber(eng, stable_live=True)
    step = int(0.8 * 16000)
    end = step
    display = ""
    worst = 0.0
    marks = []
    while end <= len(wav):
        buf = wav[max(0, end - int(live.window_sec * 16000)):end]
        t_offset = max(0.0, end / 16000.0 - buf.size / 16000.0)
        t0 = time.perf_counter()
        t = live.partial(buf, t_offset=t_offset)
        worst = max(worst, time.perf_counter() - t0)
        if t:
            display = t
        if (end // 16000) % 10 == 0 and (end % 16000) < step:
            marks.append((end // 16000, len(display.split())))
        end += step

    print("طول نمایش (ثانیه: واژه):", marks)
    n = len(display.split())
    print(f"نمایش نهایی: {n} واژه | بدترین tick: {worst * 1000:.0f}ms")
    print("متن نمایش نهایی:")
    print(display)


if __name__ == "__main__":
    main()
