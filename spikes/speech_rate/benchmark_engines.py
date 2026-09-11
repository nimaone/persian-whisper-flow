"""سنجه‌ی مجدد جدول مقایسه پس از تغییر موتور پردازش.

سنجه‌ها (هر سه مسیر):
1. ترنسکرایپ نهایی روی sample (۱۴۲٫۸s) — همان فایل جدول قبلی
2. تأخیر مسیر زنده (پنجره‌ی 10s) — میانگین/median روی ۱۵ پنجره
3. حافظه (RSS پس از لود و پس از پنجره‌های زنده) — psutil
4. ترنسکرایپ نهایی روی کلیپ کوتاه ۶٫۴s

مسیرها:
  A) پیش‌فرض: sherpa greedy (stable_live=off)
  B) تک‌مدلی: DirectCtcAsrEngine pyctcdecode beam=2 (stable_live=on) — موتور جدید
  C) دو-سشنی قدیمی: sherpa + session دوم detail (رفتار stable_live قدیمی)
"""
from __future__ import annotations

import gc
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR = 16000
SAMPLE = ROOT / "spikes" / "speech_rate" / "sample_16k.wav"
CLIP = ROOT / "training" / "kooshiar-live1" / "clips" / "seg_00000.wav"


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def bench_engine(name, make):
    print(f"\n===== {name} =====", flush=True)
    base = rss_mb()
    t0 = time.perf_counter()
    eng = make()
    load_s = time.perf_counter() - t0
    after_load = rss_mb()
    print(f"load: {load_s:.2f}s | RSS: {base:.0f} → {after_load:.0f} MB", flush=True)

    samples = load_wav(SAMPLE)
    t0 = time.perf_counter()
    txt = eng.transcribe(samples)
    final_s = time.perf_counter() - t0
    print(f"final on {samples.size/SR:.1f}s sample: {final_s:.1f}s "
          f"(RTF={final_s/(samples.size/SR):.3f}) | words={len(txt.split())}",
          flush=True)

    win = samples[-int(10 * SR):]
    lat = []
    for _ in range(15):
        t0 = time.perf_counter()
        if hasattr(eng, "transcribe_with_details"):
            eng.transcribe_with_details(win, previous_text=None)
        else:
            eng.transcribe(win)
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat)
    print(f"live latency (10s win ×15): mean={lat.mean()*1000:.0f}ms "
          f"median={np.median(lat)*1000:.0f}ms max={lat.max()*1000:.0f}ms",
          flush=True)

    after_live = rss_mb()
    print(f"RSS after live: {after_live:.0f} MB (+{after_live-after_load:.0f})",
          flush=True)

    clip = load_wav(CLIP)
    t0 = time.perf_counter()
    eng.transcribe(clip)
    print(f"final on {clip.size/SR:.1f}s clip: {time.perf_counter()-t0:.2f}s",
          flush=True)

    if hasattr(eng, "release_detail_decoder"):
        eng.release_detail_decoder()
    del eng
    gc.collect()
    time.sleep(0.3)
    print(f"RSS after release: {rss_mb():.0f} MB", flush=True)


def main():
    from app.asr import DirectCtcAsrEngine, load_engine

    print(f"base RSS: {rss_mb():.0f} MB", flush=True)

    bench_engine("پیش‌فرض (sherpa, stable_live=off)",
                 lambda: load_engine(ROOT / "model", num_threads=4))

    bench_engine("تک‌مدلی DirectCtcAsrEngine (stable_live=on، موتور جدید)",
                 lambda: DirectCtcAsrEngine(ROOT / "model", num_threads=4,
                                            beam_width=2))

    def make_dual():
        eng = load_engine(ROOT / "model", num_threads=4)
        # session دوم detail را همان‌طور که transcribe_with_details می‌سازد
        eng.transcribe_with_details(load_wav(CLIP), previous_text=None)
        return eng
    bench_engine("دو-سشنی قدیمی (sherpa + detail session)", make_dual)


if __name__ == "__main__":
    main()
