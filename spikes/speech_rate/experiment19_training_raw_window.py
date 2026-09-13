"""آزمایش ۱۹ — پنجره‌ی ۱۲ در برابر ۱۶ ثانیه روی صداهای training/raw.

روش مرجع‌گیری همان آزمایش ۱۳ است: متن proofread‌شده‌ی manifest
(training/all/manifest.jsonl) که اسنپ‌شات‌های بازه‌ی [۱۲۰, ۴۸۰)s از
همین ویدیوهای kooshiar است؛ هر clip مرز زمانی start/end دارد و زمانِ
واژه‌هایش بین مرزها درون‌یابی می‌شود. عدد پایه‌ی مسیر فینال (دیکد
chunk ۱۶ ثانیه‌ای) در report_wer_final.json: ۱۶۳۲/۴۷۲۴ = ۳۴٫۵٪.

اینجا مسیر «متن زنده پایدار» سنجیده می‌شود: پنجره‌ی جدید ۱۲ ثانیه در
برابر پیش‌فرض قبلی ۱۶ ثانیه. گام tick ۱٫۶ ثانیه است (پنجره‌های
هم‌پوشان، پوشش کامل محتوا؛ حدود نیم‌زمان گام ۰٫۸).

سنجه هر منبع: WER تجمعی tickبه‌tick نسبت به مرجع همان بازه + تأخیر
tick + بازنویسی پیشوند. خروجی: report_window_training_raw.json
"""
from __future__ import annotations

import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber
from experiment17_reference_window_sweep import extract_audio  # noqa: F401 ( consistency)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "training" / "raw"
MANIFEST = ROOT / "training" / "all" / "manifest.jsonl"
REPORT = Path(__file__).resolve().parent / "report_window_training_raw.json"

SEG_START, SEG_END = 120.0, 480.0
WINDOWS = (12.0, 16.0)   # جدید / قبلی
STEP_SEC = 1.6
TICK_BUDGET_SEC = 0.8


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


def load_source_refs() -> dict[str, list[tuple[float, float, list[str]]]]:
    """منبع → [(t0, t1, واژه‌های متن)] برای clipهای بازه‌ی مرجع."""
    refs: dict[str, list] = {}
    for line in open(MANIFEST, encoding="utf-8"):
        r = json.loads(line)
        if not (SEG_START <= r["start"] < SEG_END):
            continue
        words = norm_words(r["text"])
        if not words:
            continue
        refs.setdefault(r["source"], []).append(
            (r["start"] - SEG_START, r["end"] - SEG_START, words)
        )
    return refs


def word_times(clips) -> tuple[list[str], list[float]]:
    """واژه‌های مرجع + زمان درون‌یابی‌شده در بازه‌ی هر clip."""
    words: list[str] = []
    times: list[float] = []
    for t0, t1, ws in clips:
        span = max(0.1, t1 - t0)
        for k, w in enumerate(ws):
            words.append(w)
            times.append(t0 + span * k / len(ws))
    return words, times


def simulate(wav, window: float, eng, ref_words, ref_times) -> dict:
    live = LiveTranscriber(eng, stable_live=True, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    display = ""
    prev_keys: list[str] = []
    rewrites = shrinks = 0
    latencies: list[float] = []
    err = nref = 0
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
                e, n = wer(expected, norm_words(text))
                err += e
                nref += n
        end += step
    return {
        "err": err, "ref": nref,
        "mean_tick": sum(latencies) / len(latencies),
        "max_tick": max(latencies),
        "over_budget": sum(1 for x in latencies if x > TICK_BUDGET_SEC),
        "rewrites": rewrites, "shrinks": shrinks,
        "display_tail": display[-120:],
    }


def main():
    refs = load_source_refs()
    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    report = {"baseline_final_path": "1632/4724 = 34.5% (report_wer_final.json)",
              "sources": {}, "total": {}}

    for source, clips in sorted(refs.items()):
        wav_path = RAW / f"{source}_16k.wav"
        if not wav_path.exists():
            print(f"پرش — صوت یافت نشد: {wav_path.name}")
            continue
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        seg = wav[int(SEG_START * 16000):int((SEG_END + 1.0) * 16000)]
        ref_words, ref_times = word_times(clips)
        print(f"\n=== {source} — {len(ref_words)} واژه‌ی مرجع ===")
        report["sources"][source] = {}
        for window in WINDOWS:
            r = simulate(seg, window, eng, ref_words, ref_times)
            report["sources"][source][f"{window:.0f}s"] = r
            print(f"  {window:>2.0f}s | WER {100 * r['err'] / max(1, r['ref']):5.1f}% "
                  f"({r['err']}/{r['ref']}) | tick {r['mean_tick'] * 1000:.0f}ms "
                  f"(بدترین {r['max_tick'] * 1000:.0f}ms، {r['over_budget']} بالای بودجه) "
                  f"| بازنویسی {r['rewrites']} | کوچک‌شدن {r['shrinks']}")

    # جمع کل
    print("\n=== جمع کل ===")
    for window in WINDOWS:
        te = tn = mr = 0
        ob = 0
        for s in report["sources"].values():
            r = s[f"{window:.0f}s"]
            te += r["err"]
            tn += r["ref"]
            mr += r["rewrites"]
            ob += r["over_budget"]
        report["total"][f"{window:.0f}s"] = {
            "err": te, "ref": tn, "rewrites": mr, "over_budget": ob}
        print(f"  {window:>2.0f}s | WER {100 * te / max(1, tn):5.1f}% ({te}/{tn}) "
              f"| بازنویسی {mr} | tick بالای بودجه {ob}")

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nگزارش: {REPORT}")


if __name__ == "__main__":
    main()
