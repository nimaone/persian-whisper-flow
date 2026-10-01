"""عیب‌یابی آزمایش ۱۹ — چرا WER حالت زنده روی training/raw این‌قدر بالاست؟

سه پرسش:
1. نمایش زنده در هر tick چه شکلی است؟ (چاپ کنار هم با مرجع)
2. نمایش چند ثانیه عقب است؟ — WER در برابر پنجره‌ی مرجعِ جابجاشده با
   lagهای مختلف سنجیده می‌شود؛ اگر کمینه‌ی WER در lag ≈ طول پنجره
   باشد یعنی نمایش یک پنجره‌ی کامل عقب است (فریز).
3. حالت زنده‌ی معمولی (بدون مکانیزم پایدار، پنجره ۱۰ ثانیه) روی همین
   صدا چه WER ای دارد؟ — پایه‌ی «قبل از حالت پایدار».
"""
from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "training" / "raw"
MANIFEST = ROOT / "training" / "all" / "manifest.jsonl"

SOURCE = "kooshiar-aug23-p1"
SEG_START, SEG_LEN = 120.0, 120.0   # ۱۲۰ ثانیه برای سرعت عیب‌یابی
WINDOW = 12.0
STEP_SEC = 1.6


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


def load_ref() -> tuple[list[str], list[float]]:
    words, times = [], []
    for line in open(MANIFEST, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] != SOURCE or not (SEG_START <= r["start"] < SEG_START + SEG_LEN):
            continue
        ws = norm_words(r["text"])
        t0, t1 = r["start"] - SEG_START, r["end"] - SEG_START
        span = max(0.1, t1 - t0)
        for k, w in enumerate(ws):
            words.append(w)
            times.append(t0 + span * k / len(ws))
    return words, times


def run(wav, eng, ref_words, ref_times, stable: bool, window: float):
    live = LiveTranscriber(eng, stable_live=stable, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    ticks: list[tuple[float, str]] = []   # (buf_end, display)
    while end <= len(wav):
        buf = wav[max(0, end - int(window * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        text = live.partial(buf, t_offset=t_offset)
        ticks.append((end / 16000.0, text or ""))
        end += step
    return ticks


def main():
    wav, sr = sf.read(RAW / f"{SOURCE}_16k.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    seg = wav[int(SEG_START * 16000):int((SEG_START + SEG_LEN + 1.0) * 16000)]
    ref_words, ref_times = load_ref()
    print(f"{len(ref_words)} واژه‌ی مرجع در {SEG_LEN:.0f} ثانیه‌ی {SOURCE}")

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)

    # --- سؤال ۲ و ۱: پایدار ۱۲ ثانیه ---
    ticks = run(seg, eng, ref_words, ref_times, stable=True, window=WINDOW)
    print(f"\n--- پایدار {WINDOW:.0f}s: {len(ticks)} tick ---")
    print("نمونه‌ی tickها (نمایش در برابر مرجع همان بازه):")
    for i in (10, 30, 50, 70):
        if i >= len(ticks):
            continue
        buf_end, display = ticks[i]
        t_start = max(0.0, buf_end - WINDOW)
        expected = [w for w, t in zip(ref_words, ref_times) if t_start <= t < buf_end]
        print(f"\n[tick {i}] buf_end={buf_end:.1f}s — نمایش ({len(display.split())} واژه):")
        print(f"  {display[:150]!r}")
        print(f"  مرجع بازه ({len(expected)} واژه): {' '.join(expected)[:150]!r}")

    # WER در برابر پنجره‌ی جابجاشده با lagهای مختلف
    print("\nWER در برابر پنجره‌ی مرجعِ جابجاشده (lag ثانیه):")
    for lag in (0, 2, 4, 6, 8, 10, 12, 14):
        err = nref = 0
        for buf_end, display in ticks:
            if not display.strip():
                continue
            t_start = max(0.0, buf_end - WINDOW - lag)
            t_end = buf_end - lag
            if t_end <= 0:
                continue
            expected = [w for w, t in zip(ref_words, ref_times) if t_start <= t < t_end]
            if expected:
                e, n = wer(expected, norm_words(display))
                err += e
                nref += n
        print(f"  lag {lag:>2}s → WER {100 * err / max(1, nref):5.1f}% ({err}/{nref})")

    # --- سؤال ۳: زنده‌ی معمولی (قبل از حالت پایدار) ---
    ticks_plain = run(seg, eng, ref_words, ref_times, stable=False, window=10.0)
    err = nref = 0
    for buf_end, display in ticks_plain:
        if not display.strip():
            continue
        t_start = max(0.0, buf_end - 10.0)
        expected = [w for w, t in zip(ref_words, ref_times) if t_start <= t < buf_end]
        if expected:
            e, n = wer(expected, norm_words(display))
            err += e
            nref += n
    print(f"\n--- زنده‌ی معمولی (بدون پایدار، پنجره ۱۰s) → "
          f"WER {100 * err / max(1, nref):.1f}% ({err}/{nref}) ---")


if __name__ == "__main__":
    main()
