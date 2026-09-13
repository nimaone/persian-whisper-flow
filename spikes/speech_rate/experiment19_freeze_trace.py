"""عیب‌یابی فریز نمایش پایدار — لاگ تصمیم‌های _accepts tickبه‌tick.

روی بازه‌ی کوتاه aug23-p1 (پنجره ۱۲ ثانیه)، متن خام هر tick، اطمینانش
و تصمیم‌گیری نهایی چاپ می‌شود تا مشخص شود کدام قانون نمایش را فریز
کرده است.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import DirectCtcAsrEngine, LiveTranscriber
from experiment19_diag import load_ref, norm_words

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "training" / "raw"
SOURCE = "kooshiar-aug23-p1"
SEG_START, SEG_LEN = 120.0, 80.0
WINDOW = 12.0
STEP_SEC = 1.6


def main():
    wav, sr = sf.read(RAW / f"{SOURCE}_16k.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    seg = wav[int(SEG_START * 16000):int((SEG_START + SEG_LEN + 1.0) * 16000)]

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    live = LiveTranscriber(eng, stable_live=True, window_sec=WINDOW)

    # لاگ‌کردن ورودی/خروجی _accepts بدون تغییر کد اپ
    orig_accepts = live._accepts
    log = []

    def logged_accepts(candidate, current, t_start=0.0, buf_end=0.0):
        ok = orig_accepts(candidate, current, t_start=t_start, buf_end=buf_end)
        log.append({
            "buf_end": round(buf_end, 1),
            "conf": round(candidate.confidence, 3),
            "n_words": len(candidate.words) if candidate.words else len(candidate.text.split()),
            "cand": candidate.text[:70],
            "cur": current.text[:70] if current else None,
            "ok": ok,
        })
        return ok

    live._accepts = logged_accepts

    # لاگ خطاهای مسیر details
    orig_details = eng.transcribe_with_details
    errors = []

    def logged_details(samples, sample_rate=16000, previous_text=None):
        try:
            return orig_details(samples, sample_rate, previous_text=previous_text)
        except Exception as e:
            errors.append(repr(e))
            raise

    eng.transcribe_with_details = logged_details

    step = int(STEP_SEC * 16000)
    end = step
    i = 0
    while end <= len(seg):
        buf = seg[max(0, end - int(WINDOW * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        text = live.partial(buf, t_offset=t_offset)
        if i % 3 == 0 and log:
            e = log[-1]
            print(f"tick {i:>2} buf_end={e['buf_end']:>5.1f} ok={str(e['ok']):>5} "
                  f"conf={e['conf']:.2f} n={e['n_words']:>2} | کاندید: {e['cand'][:55]!r}")
        end += step
        i += 1

    print(f"\nخطاهای transcribe_with_details: {len(errors)}")
    for e in errors[:3]:
        print(" ", e)

    # چند تصمیم ردشده با جزئیات
    print("\nنمونه‌ی ردشده‌ها (ok=False):")
    shown = 0
    for e in log:
        if not e["ok"] and shown < 8:
            print(f"  buf_end={e['buf_end']} conf={e['conf']} n={e['n_words']}")
            print(f"    کاندید: {e['cand']!r}")
            print(f"    فعلی:   {e['cur']!r}")
            shown += 1


if __name__ == "__main__":
    main()
