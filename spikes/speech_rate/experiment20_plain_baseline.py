"""تکمیل آزمایش ۱۹ — live1 بعد از رفع باگ قفل اول + پایه‌ی «حالت معمولی».

پرسش نهایی کاربر: حالت زنده پایدار از معمولی بهتر است یا نه؟
این اسکریپت برای هر منبعِ manifest، حالت معمولی (پنجره ۱۰ ثانیه،
بدون مکانیزم پایدار — همان رفتار قبل از حالت پایدار) را می‌سنجد و
live1 را با حالت پایدارِ اصلاح‌شده دوباره اجرا می‌کند.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import LiveTranscriber
from experiment19_training_raw_window import (
    RAW, STEP_SEC, load_source_refs, norm_words, wer, word_times,
)

SEG_START, SEG_END = 120.0, 480.0


def simulate_generic(wav, live, window, ref_words, ref_times):
    step = int(STEP_SEC * 16000)
    end = step
    err = nref = 0
    while end <= len(wav):
        buf = wav[max(0, end - int(window * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        text = live.partial(buf, t_offset=t_offset)
        if text.strip():
            buf_end = end / 16000.0
            t_start = max(0.0, buf_end - window)
            expected = [w for w, t in zip(ref_words, ref_times)
                        if t_start <= t < buf_end]
            if expected:
                e, n = wer(expected, norm_words(text))
                err += e
                nref += n
        end += step
    return err, nref


def main():
    refs = load_source_refs()
    eng = __import__("app.asr", fromlist=["DirectCtcAsrEngine"]) \
        .DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)

    print("=== حالت معمولی (بدون پایدار، پنجره ۱۰s) ===")
    tot_e = tot_n = 0
    for source, clips in sorted(refs.items()):
        wav_path = RAW / f"{source}_16k.wav"
        if not wav_path.exists():
            continue
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        seg = wav[int(SEG_START * 16000):int((SEG_END + 1.0) * 16000)]
        ref_words, ref_times = word_times(clips)
        live = LiveTranscriber(eng, stable_live=False, window_sec=10.0)
        e, n = simulate_generic(seg, live, 10.0, ref_words, ref_times)
        tot_e += e
        tot_n += n
        print(f"  {source:<24} WER {100 * e / max(1, n):5.1f}% ({e}/{n})")
    print(f"  {'جمع':<24} WER {100 * tot_e / max(1, tot_n):5.1f}% ({tot_e}/{tot_n})")

    print("\n=== live1 با حالت پایدارِ اصلاح‌شده ===")
    clips = refs["kooshiar-live1"]
    wav, sr = sf.read(RAW / "kooshiar-live1_16k.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    seg = wav[int(SEG_START * 16000):int((SEG_END + 1.0) * 16000)]
    ref_words, ref_times = word_times(clips)
    for window in (12.0, 16.0):
        live = LiveTranscriber(eng, stable_live=True, window_sec=window)
        e, n = simulate_generic(seg, live, window, ref_words, ref_times)
        print(f"  پایدار {window:.0f}s → WER {100 * e / max(1, n):5.1f}% ({e}/{n})")


if __name__ == "__main__":
    main()
