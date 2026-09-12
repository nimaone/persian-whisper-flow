"""آزمایش ۱۷ — جاروب سایز پنجره‌ی متن زنده پایدار با مرجع واقعی کاربر.

مرجع: sample/reference.txt (متن دقیق ویدیو، تاییدشده توسط گوینده).

چون نمایش زنده فقط متنِ پنجره‌ی انتهایی است (نه کل ویدیو)، WER
tickبه‌tick سنجیده می‌شود: در هر tick، نمایش زنده در برابر واژه‌های
مرجعِ همان بازه‌ی زمانی پنجره مقایسه می‌شود. برای این کار واژه‌های
مرجع به زمان تطبیق داده می‌شوند: کل فایل با transcribe_with_details
(chunkهای ۱۶ ثانیه‌ای) decode می‌شود، زمانِ مطلق هر واژه‌ی decode‌شده
از فریم‌های مدل (۸۰ms) درمی‌آید و واژه‌های مرجع با SequenceMatcher
هم‌تراز و واژه‌های بی‌قرینه بین همسایه‌ها درون‌یابی می‌شوند.

سنجه‌ها برای هر سایز پنجره:
- WER تجمعی tickبه‌tick در برابر مرجع همان بازه — تفکیک‌شده برای
  واژه‌های تخصصی (sample/specialist_words.txt) و واژه‌های معمولی.
  خطای هر کلاس = حذف‌ها + جایگزینی‌های همان کلاس؛ درج‌های اضافی
  حساب نمی‌شوند تا دو کلاس جمع‌پذیر باشند.
- تأخیر هر tick (زنده): میانگین، بدترین، تعداد tickهای بالای ۸۰۰ms
- پایداری: تعداد «بازنویسی» (تغییر پیشوند نوشته‌شده) و «کوچک‌شدن»

در ابتدا WER دیکدِ کامل آفلاین (بدون قیود زنده) هم با همین تفکیک
چاپ می‌شود تا معلوم شود خطای واژه‌های تخصصی ذاتی مدل است یا محصول
مکانیزم زنده.
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
REF = ROOT / "sample" / "reference.txt"

WINDOWS = (8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
STEP_SEC = 0.8          # PARTIAL_INTERVAL اپ
TICK_BUDGET_SEC = 0.8   # tick بالاتر از این یعنی عقب افتادن از real-time
SEC_PER_FRAME = 0.08    # MS_PER_MODEL_FRAME در asr.py


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


SPEC_LEXICON_PATH = ROOT / "sample" / "specialist_words.txt"


def load_specialist() -> set[str]:
    if not SPEC_LEXICON_PATH.exists():
        return set()
    out = set()
    for line in SPEC_LEXICON_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(_cmp_key(line))
    return out


def wer_by_class(
    expected: list[str],
    hyp_words: list[str],
    specialist: set[str],
    missed: dict[str, int] | None = None,
) -> dict[str, tuple[int, int]]:
    """WER هر کلاس از روی opcodes: حذف و جایگزینی به کلاسِ واژه‌ی
    مرجع (expected) حساب می‌شوند؛ درجِ اضافی نادیده گرفته می‌شود تا
    کلاس‌ها جمع‌پذیر باشند و جمعشان = WER کل."""
    exp_keys = [_cmp_key(w) for w in expected]
    hyp_keys = [_cmp_key(w) for w in hyp_words]
    errs = {"spec": 0, "common": 0}
    nref = {"spec": 0, "common": 0}
    for w in expected:
        nref["spec" if _cmp_key(w) in specialist else "common"] += 1
    m = SequenceMatcher(None, hyp_keys, exp_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            for k in range(j1, j2):
                cls = "spec" if exp_keys[k] in specialist else "common"
                errs[cls] += 1
                if missed is not None and cls == "spec":
                    missed[expected[k]] = missed.get(expected[k], 0) + 1
        elif tag == "delete":
            for k in range(j1, j2):
                cls = "spec" if exp_keys[k] in specialist else "common"
                errs[cls] += 1
                if missed is not None and cls == "spec":
                    missed[expected[k]] = missed.get(expected[k], 0) + 1
    return {"spec": (errs["spec"], nref["spec"]),
            "common": (errs["common"], nref["common"])}


def build_ref_timeline(wav, eng) -> tuple[list[str], list[float], list[str]]:
    """زمان مطلق هر واژه‌ی مرجع + متن واژه‌های decode آفلاین."""
    ref_text = REF.read_text(encoding="utf-8")
    ref_words = norm_words(ref_text)

    chunk = int(16.0 * 16000)
    dec_words: list[tuple[str, float]] = []  # (کلید نرمال، زمان شروع مطلق)
    dec_texts: list[str] = []
    for start in range(0, len(wav), chunk):
        seg = wav[start:start + chunk]
        hyp = eng.transcribe_with_details(seg)
        dec_texts.extend(hyp.text.split())
        t_base = start / 16000.0
        for w in hyp.words:
            dec_words.append((_cmp_key(w.text), t_base + (w.start or 0) * SEC_PER_FRAME))

    dec_keys = [d[0] for d in dec_words]
    ref_keys = [_cmp_key(w) for w in ref_words]
    m = SequenceMatcher(None, dec_keys, ref_keys, autojunk=False)
    times: list[float | None] = [None] * len(ref_words)
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                times[j1 + k] = dec_words[i1 + k][1]
    matched = sum(1 for t in times if t is not None)
    # درون‌یابی واژه‌های بی‌قرینه بین همسایه‌های زمان‌دار
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    for i in range(len(times)):
        if times[i] is not None:
            continue
        prev = max((j, t) for j, t in known if j < i) if any(j < i for j, _ in known) else None
        nxt = min((j, t) for j, t in known if j > i) if any(j > i for j, _ in known) else None
        if prev is None:
            times[i] = nxt[1] if nxt else 0.0
        elif nxt is None:
            times[i] = prev[1]
        else:
            span = nxt[0] - prev[0]
            times[i] = prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / max(1, span)
    print(f"مرجع: {len(ref_words)} واژه — {matched} واژه با زمان تطبیق‌شده "
          f"({100 * matched / max(1, len(ref_words)):.0f}%)")
    return ref_words, [float(t) for t in times], dec_texts


def simulate(wav, window: float, eng, ref_words, ref_times, specialist) -> dict:
    live = LiveTranscriber(eng, stable_live=True, window_sec=window)
    step = int(STEP_SEC * 16000)
    end = step
    display = ""
    prev_keys: list[str] = []
    rewrites = 0
    shrinks = 0
    latencies: list[float] = []
    cls_errs = {"spec": 0, "common": 0}
    cls_nref = {"spec": 0, "common": 0}
    missed: dict[str, int] = {}
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
            # مرجع همان بازه‌ی پنجره در این tick
            buf_end = end / 16000.0
            t_start = max(0.0, buf_end - window)
            expected = [w for w, t in zip(ref_words, ref_times)
                        if t_start <= t < buf_end]
            if expected:
                by_class = wer_by_class(expected, norm_words(text), specialist, missed)
                for cls in cls_errs:
                    cls_errs[cls] += by_class[cls][0]
                    cls_nref[cls] += by_class[cls][1]
        end += step
    return {
        "window": window,
        "display": display,
        "cls_errs": cls_errs,
        "cls_nref": cls_nref,
        "mean_tick": sum(latencies) / len(latencies),
        "max_tick": max(latencies),
        "over_budget": sum(1 for x in latencies if x > TICK_BUDGET_SEC),
        "rewrites": rewrites,
        "shrinks": shrinks,
        "missed": missed,
    }


def main():
    wav_path = extract_audio()
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000
    print(f"صدا: {wav_path.name} — {len(wav) / sr:.1f} ثانیه")

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    ref_words, ref_times, dec_texts = build_ref_timeline(wav, eng)
    specialist = load_specialist()
    n_spec = sum(1 for w in ref_words if _cmp_key(w) in specialist)
    print(f"واژه‌های تخصصی مرجع: {n_spec} از {len(ref_words)} "
          f"({100 * n_spec / max(1, len(ref_words)):.0f}%) — واژه‌نامه: {SPEC_LEXICON_PATH.name}")

    # خطای ذاتی مدل: دیکد کامل آفلاین بدون قیود زنده، با همان تفکیک
    off = wer_by_class(ref_words, dec_texts, specialist)
    for label, cls in (("تخصصی", "spec"), ("معمولی", "common")):
        e, n = off[cls]
        print(f"دیکد آفلاین — WER {label}: {e}/{n} ({100 * e / max(1, n):.1f}%)")

    print(f"\n{'پنجره':>6} | {'WER کل':>7} | {'WER تخصصی':>9} | {'WER معمولی':>10} | "
          f"{'میانگین tick':>11} | {'>۸۰۰ms':>6} | {'بازنویسی':>8}")
    print("-" * 84)
    for window in WINDOWS:
        r = simulate(wav, window, eng, ref_words, ref_times, specialist)
        total_e = r["cls_errs"]["spec"] + r["cls_errs"]["common"]
        total_n = r["cls_nref"]["spec"] + r["cls_nref"]["common"]
        se, sn = r["cls_errs"]["spec"], r["cls_nref"]["spec"]
        ce, cn = r["cls_errs"]["common"], r["cls_nref"]["common"]
        print(f"{r['window']:>5.0f}s | {100 * total_e / max(1, total_n):>6.1f}% | "
              f"{se}/{sn} ({100 * se / max(1, sn):.0f}%) | "
              f"{ce}/{cn} ({100 * ce / max(1, cn):.0f}%) | "
              f"{r['mean_tick'] * 1000:>9.0f}ms | "
              f"{r['over_budget']:>6d} | {r['rewrites']:>8d}")
        print(f"        نمایش آخر: {r['display'][:90]!r}" + ("…" if len(r["display"]) > 90 else ""))
        top_missed = sorted(r["missed"].items(), key=lambda kv: -kv[1])[:6]
        if top_missed:
            print(f"        بیشترین واژه‌های تخصصی ازدست‌رفته: "
                  + ", ".join(f"{w}×{c}" for w, c in top_missed))

    print("\nWER تخصصی/معمولی = حذف + جایگزینی در همان کلاس؛ درج اضافی حساب نمی‌شود.")
    print("«بازنویسی» بالای صفر یعنی متن نوشته‌شده وسط کار عوض شده (نشت پایداری).")


if __name__ == "__main__":
    main()
