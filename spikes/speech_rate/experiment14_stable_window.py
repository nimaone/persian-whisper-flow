"""آزمایش ۱۴ — متن زنده پایدار برای دیکته‌های ۱۰-۱۵ ثانیه‌ای: پنجره ۱۰ در برابر ۱۶ ثانیه.

فرضیه: پنجره‌ی زنده ۱۰ ثانیه‌ای برای دیکته‌های بلندتر از ۱۰ ثانیه، ابتدای
صحت را هرگز نمی‌بیند (پوشش ناقص) و اسکرول پنجره بازآرایش پیشوند را
تحمیل می‌کند. پنجره‌ی ۱۶ ثانیه‌ای باید پوشش و کیفیت را بالا ببرد — به
شرطی که تأخیر هر tick زیر PARTIAL_INTERVAL (۰٫۸s) بماند.

شبیه‌سازی دقیق اپ: بافر رشدکننده، partial هر ۰٫۸s روی پنجره‌ی انتهایی،
متن نمایش نهایی (آخرین non-empty) در برابر متن مرجع manifest.
"""
from __future__ import annotations

import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import soundfile as sf

from app.config import model_dir
from app.asr import DirectCtcAsrEngine, LiveTranscriber


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


def main():
    rows = [json.loads(l) for l in
            open("training/all/manifest.jsonl", encoding="utf-8")]
    clips = [r for r in rows if 10.0 <= r.get("duration", 0) <= 15.0][:6]
    print(f"کلیپ‌ها: {[r['id'] for r in clips]}")

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)

    for window in (10.0, 16.0):
        tot_err = tot_ref = 0
        tot_cov_err = 0
        max_tick = 0.0
        for r in clips:
            wav, _ = sf.read("training/all/" + r["audio"], dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]
            live = LiveTranscriber(eng, stable_live=True)
            live.window_sec = window
            step = int(0.8 * 16000)
            end = step
            display = ""
            while end <= len(wav):
                buf = wav[max(0, end - int(window * 16000)):end]
                t0 = time.perf_counter()
                t = live.partial(buf)
                dt = time.perf_counter() - t0
                max_tick = max(max_tick, dt)
                if t:
                    display = t
                end += step
            ref = norm_words(r["text"])
            hyp = norm_words(display)
            err, nref = wer(ref, hyp)
            # پوشش: چند واژه‌ی مرجع اصلاً در نمایش نیستند (بریدگی ابتدای جمله)
            cov_err, _ = wer(ref, hyp)  # همان — چون hyp کوتاه‌تر = خطا
            tot_err += err
            tot_ref += nref
            tot_cov_err += max(0, len(ref) - len(hyp))
            print(f"  [{r['id']}] ref={len(ref)} نمایش={len(hyp)} "
                  f"خطا={err} | {display[:55]!r}")
        print(f"پنجره {window:.0f}s → خطای کل: {tot_err}/{tot_ref} "
              f"({100 * tot_err / max(1, tot_ref):.1f}%) | "
              f"واژه‌های گم‌شده‌ی ابتدای جمله: {tot_cov_err} | "
              f"بدترین tick: {max_tick * 1000:.0f}ms")


if __name__ == "__main__":
    main()
