"""آزمایش ۲۲ — تجربه‌ی کاربر، نه WER تجمعی: نرخ پس‌گرفتنِ واژه در دو حالت.

WER تجمعی tickبه‌tick هر دو حالت را «شکل نهایی» می‌سنجد و تفاوت
تجربه را پنهان می‌کند. برای کاربر، آنچه رنج‌آور است:
- واژه‌ای که دیده و بعد پاک شده (پس‌گرفته شدن)
- واژه‌ای که دو شکل مختلف دیده (نوسان نهایی آن)

این آزمایش روی صدای خود کاربر (sample) می‌شمارد:
1. چمپ (churn): در هر tick، چند واژه‌ی نمایشِ tick قبل در نمایش جدید
   نیست — یعنی واژه‌های «پس‌گرفته‌شده» یا جابه‌جا. برای کاربر یعنی
   متن زیر دستش تکان می‌خورد.
2. سرعت پوشش: در هر لحظه‌ی زمانی، چند درصد از واژه‌های مرجعِ تا آن
   لحظه در نمایش هست (فرقی نمی‌کند کدام پنجره) — مقایسه‌ی «تأخیر
   تجربی» دو حالت.
3. پایان کار: نمایش لحظه‌ی آخر در برابر مرجع (کاربر موقع توقف ضبط
   همین را می‌بیند و بعدش متن نهایی جایگزینش می‌شود).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber
from experiment17_reference_window_sweep import (
    STEP_SEC, build_ref_timeline, extract_audio, norm_words,
)

WINDOWS = (8.0, 12.0, 16.0)


def simulate_ticks(wav, eng, window, stable, ref_words, ref_times):
    live = LiveTranscriber(eng, stable_live=stable, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    ticks = []          # (buf_end, keys)
    while end <= len(wav):
        buf = wav[max(0, end - int(window * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        text = live.partial(buf, t_offset=t_offset)
        if text.strip():
            ticks.append((end / 16000.0, [_cmp_key(w) for w in text.split()]))
        end += step
    return ticks


def main():
    wav, sr = sf.read(extract_audio(), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    ref_words, ref_times, _ = build_ref_timeline(wav, eng)

    print(f"{'حالت':>7} | {'پنجره':>5} | {'چمپ/tick':>9} | {'سرعت پوشش':>10} | "
          f"{'WER پایان':>9} | {'واژه‌ی زنده‌ی مرده':>18}")
    print("-" * 78)
    for window in WINDOWS:
        for stable, mode in ((False, "معمولی"), (True, "پایدار")):
            ticks = simulate_ticks(wav, eng, window, stable, ref_words, ref_times)
            # چمپ: واژه‌های tick قبل که در tick جدید نیستند
            churn_total = churn_ticks = 0
            prev = None
            for _, keys in ticks:
                if prev is not None:
                    gone = sum(1 for k in prev if k not in keys)
                    if gone:
                        churn_total += gone
                        churn_ticks += 1
                prev = keys
            # سرعت پوشش: در هر tick چند درصد از مرجعِ تا آن لحظه حاضر است
            covs = []
            for buf_end, keys in ticks:
                ref_upto = [w for w, t in zip(ref_words, ref_times) if t <= buf_end]
                if not ref_upto:
                    continue
                have = sum(1 for w in ref_upto if _cmp_key(w) in keys)
                covs.append(have / len(ref_upto))
            # پایان کار
            from difflib import SequenceMatcher
            final_keys = ticks[-1][1] if ticks else []
            ref_all = [_cmp_key(w) for w in ref_words]
            m = SequenceMatcher(None, final_keys, ref_all, autojunk=False)
            matched = sum(b.size for b in m.get_matching_blocks())
            final_wer = 100 * (len(ref_all) - matched) / max(1, len(ref_all))
            print(f"{mode:>7} | {window:>4.0f}s | {churn_total / max(1, len(ticks)):>8.1f} | "
                  f"{100 * sum(covs) / max(1, len(covs)):>9.1f}% | "
                  f"{final_wer:>8.1f}% | "
                  f"{churn_ticks:>3d} از {len(ticks)} tick")


if __name__ == "__main__":
    main()
