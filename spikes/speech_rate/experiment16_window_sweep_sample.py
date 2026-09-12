"""آزمایش ۱۶ — جاروب سایز پنجره‌ی متن زنده پایدار روی صدای کاربر (sample).

ورودی: sample/explorer_QQCtNcPon9.mp4 (۲:۲۶) → صدای 16k مونو در
spikes/speech_rate/sample_video16k.wav (با ffmpeg استخراج می‌شود اگر نباشد).

مرجع: decode کامل کل فایل با همان موتور (chunkهای ۱۶ ثانیه‌ای — همان
مسیر transcribe نهایی اپ). چون متن واقعی ویدیو در دسترس نیست، WER نسبت
به این مرجع فقط «فاصله‌ی نمایش زنده از decode کامل» را می‌سنجد؛ متن
نهایی هر پنجره هم چاپ می‌شود تا مقایسه‌ی چشمی ممکن باشد.

روش شبیه‌سازی مثل آزمایش ۱۴: بافر رشدکننده، partial هر ۰٫۸s
(PARTIAL_INTERVAL اپ) روی پنجره‌ی انتهایی.

سنجه‌ها برای هر سایز پنجره:
- WER tickبه‌tick در برابر «ایده‌آلِ همان برش»: در هر tick همان پنجره‌ی
  صوتی بدون قیودِ زنده decode می‌شود؛ اختلاف = هزینه‌ی کیفیتِ مکانیزم
  قفل/پایداری در آن سایز پنجره (مقایسه با مرجع کل فایل بی‌معناست چون
  نمایش زنده فقط متنِ پنجره‌ی انتهایی است).
- تأخیر هر tick (زنده): میانگین، بدترین، و تعداد tickهای بالای ۸۰۰ms
  (زیر ۸۰۰ms نماند یعنی در اپ real-time عقب می‌ماند)
- پایداری: تعداد «بازنویسی» — tickهایی که متن نمایش جدید، ادامه‌ی
  طبیعی متن قبلی نیست (پیشوند نوشته‌شده تغییر کرده) یا کوچک‌تر شده.
"""
from __future__ import annotations

import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "sample" / "explorer_QQCtNcPon9.mp4"
WAV = Path(__file__).resolve().parent / "sample_video16k.wav"

WINDOWS = (8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
STEP_SEC = 0.8          # PARTIAL_INTERVAL اپ
TICK_BUDGET_SEC = 0.8   # tick بالاتر از این یعنی عقب افتادن از real-time


def extract_audio() -> Path:
    if WAV.exists():
        return WAV
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(VIDEO), "-ac", "1", "-ar", "16000", str(WAV)],
        check=True, capture_output=True,
    )
    return WAV


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


def simulate(wav, window: float, eng, ref_text: str) -> dict:
    live = LiveTranscriber(eng, stable_live=True, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    display = ""
    prev_keys: list[str] = []
    rewrites = 0
    shrinks = 0
    latencies: list[float] = []
    while end <= len(wav):
        buf = wav[max(0, end - int(window * 16000)):end]
        # t_offset = جایگاه مطلق ابتدای بافر — مثل main.py؛ بدون آن
        # خط زمانی رأی/رقیب و تشخیص اسکرول پنجره غیرفعال می‌ماند.
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        t0 = time.perf_counter()
        text = live.partial(buf, t_offset=t_offset)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        if text.strip():
            display = text
            keys = [_cmp_key(w) for w in display.split()]
            if prev_keys:
                if len(keys) < len(prev_keys):
                    shrinks += 1
                elif keys[:len(prev_keys)] != prev_keys:
                    rewrites += 1
            prev_keys = keys
        end += step
    err, nref = wer(norm_words(ref_text), norm_words(display))
    return {
        "window": window,
        "display": display,
        "err": err,
        "ref": nref,
        "mean_tick": sum(latencies) / len(latencies),
        "max_tick": max(latencies),
        "over_budget": sum(1 for x in latencies if x > TICK_BUDGET_SEC),
        "rewrites": rewrites,
        "shrinks": shrinks,
    }


def main():
    wav_path = extract_audio()
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000
    dur = len(wav) / sr
    print(f"صدا: {wav_path.name} — {dur:.1f} ثانیه")

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)

    t0 = time.perf_counter()
    ref_text = eng.transcribe(wav)
    print(f"مرجع (decode کامل، {time.perf_counter() - t0:.1f}s):")
    print(f"  {ref_text}\n")
    ref_n = len(norm_words(ref_text))

    print(f"{'پنجره':>6} | {'WER':>6} | {'میانگین tick':>11} | "
          f"{'بدترین tick':>11} | {'>۸۰۰ms':>6} | {'بازنویسی':>8} | {'کوچک‌شدن':>8}")
    print("-" * 72)
    for window in WINDOWS:
        r = simulate(wav, window, eng, ref_text)
        print(f"{r['window']:>5.0f}s | {100 * r['err'] / max(1, r['ref']):>5.1f}% | "
              f"{r['mean_tick'] * 1000:>9.0f}ms | {r['max_tick'] * 1000:>9.0f}ms | "
              f"{r['over_budget']:>6d} | {r['rewrites']:>8d} | {r['shrinks']:>8d}")
        print(f"        نمایش: {r['display'][:100]!r}" + ("…" if len(r["display"]) > 100 else ""))

    print(f"\nمرجع {ref_n} واژه — WER پایین‌تر = نمایش زنده نزدیک‌تر به decode کامل؛")
    print("«بازنویسی» بالای صفر یعنی متن نوشته‌شده وسط کار عوض شده (نشت پایداری).")


if __name__ == "__main__":
    main()
