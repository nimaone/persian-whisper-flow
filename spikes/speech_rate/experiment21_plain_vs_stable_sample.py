"""آزمایش ۲۱ — حالت معمولی در برابر پایدار روی صدای خود کاربر (sample).

همان صدا و مرجع آزمایش ۱۷ (sample/reference.txt، متن دقیق گوینده)،
این بار هر دو حالت با ۵ پنجره:
- معمولی: بدون مکانیزم پایدار — هر tick دیکدِ خام پنجره انتهایی.
- پایدار: پیشوند قفل‌شده با قوانین رأی/رقیب (کد فعلی اپ).

WER تجمعی tickبه‌tick نسبت به مرجع همان بازه + تفکیک تخصصی/معمولی.
گام tick ۰٫۸ ثانیه (PARTIAL_INTERVAL اپ).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber
from experiment17_reference_window_sweep import (
    STEP_SEC, build_ref_timeline, extract_audio, load_specialist,
    norm_words, wer_by_class,
)

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (8.0, 10.0, 12.0, 14.0, 16.0)
TICK_BUDGET_SEC = 0.8


def simulate(wav, eng, window, stable, ref_words, ref_times, specialist):
    live = LiveTranscriber(eng, stable_live=stable, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    display = ""
    prev_keys: list[str] = []
    rewrites = shrinks = 0
    latencies: list[float] = []
    cls_errs = {"spec": 0, "common": 0}
    cls_nref = {"spec": 0, "common": 0}
    while end <= len(wav):
        buf = wav[max(0, end - int(window * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        t0 = time.perf_counter()
        text = live.partial(buf, t_offset=t_offset)
        latencies.append(time.perf_counter() - t0)
        if text.strip():
            display = text
            keys = [_cmp_key(w) for w in display.split()]
            if prev_keys:
                if len(keys) < len(prev_keys):
                    shrinks += 1
                elif keys[:len(prev_keys)] != prev_keys:
                    rewrites += 1
            prev_keys = keys
            buf_end = end / 16000.0
            t_start = max(0.0, buf_end - window)
            expected = [w for w, t in zip(ref_words, ref_times)
                        if t_start <= t < buf_end]
            if expected:
                by = wer_by_class(expected, norm_words(text), specialist)
                for cls in cls_errs:
                    cls_errs[cls] += by[cls][0]
                    cls_nref[cls] += by[cls][1]
        end += step
    return {
        "display": display,
        "cls_errs": cls_errs, "cls_nref": cls_nref,
        "mean_tick": sum(latencies) / len(latencies),
        "max_tick": max(latencies),
        "over_budget": sum(1 for x in latencies if x > TICK_BUDGET_SEC),
        "rewrites": rewrites, "shrinks": shrinks,
    }


def main():
    wav, sr = sf.read(extract_audio(), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    ref_words, ref_times, _ = build_ref_timeline(wav, eng)
    specialist = load_specialist()

    print(f"{'حالت':>8} | {'پنجره':>5} | {'WER کل':>7} | {'WER تخصصی':>10} | "
          f"{'WER معمولی':>10} | {'tick':>6} | {'بازنویسی':>8}")
    print("-" * 76)
    for stable, mode in ((False, "معمولی"), (True, "پایدار")):
        for window in WINDOWS:
            r = simulate(wav, eng, window, stable, ref_words, ref_times, specialist)
            te = sum(r["cls_errs"].values())
            tn = sum(r["cls_nref"].values())
            se, sn = r["cls_errs"]["spec"], r["cls_nref"]["spec"]
            ce, cn = r["cls_errs"]["common"], r["cls_nref"]["common"]
            extra = (f" | {r['rewrites']:>3d} بازنویسی, {r['shrinks']:>2d} کوچک‌شدن"
                     if stable else "")
            print(f"{mode:>8} | {window:>4.0f}s | {100 * te / max(1, tn):>6.1f}% | "
                  f"{se}/{sn} ({100 * se / max(1, sn):.0f}%) | "
                  f"{ce}/{cn} ({100 * ce / max(1, cn):.0f}%) | "
                  f"{r['mean_tick'] * 1000:>4.0f}ms | {r['over_budget']:>3d} بالای بودجه{extra}")


if __name__ == "__main__":
    main()
