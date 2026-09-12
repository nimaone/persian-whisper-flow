"""آزمایش ۱۸ — اثر واژه‌های ثبت‌شده (alias map) روی WER تخصصی در حالت زنده.

در main.py نگاشت واژه‌های ثبت‌شده فقط وقتی stable_live خاموش است اعمال
می‌شود (کامیت 69a7583). این آزمایش می‌سنجد که روشنش چقدر می‌ارزد.

روش:
1. واریانت‌های واقعی خطای مدل، از هم‌ترازسازی دیکد آفلاین با مرجع
   درمی‌آید: هر واژه‌ی تخصصیِ مرجع که مدل به شکل دیگری شنیده، یک
   (واژه‌ی درست ← واریانتِ شنیده‌شده) می‌سود. همین‌ها نقش «واژه‌های
   ثبت‌شده‌ی کاربر» را بازی می‌کنند.
2. شبیه‌سازی زنده با پنجره‌ی ۱۲ ثانیه (پیش‌فرض جدید) یک‌بار اجرا و
   متن هر tick دو بار سنجیده می‌شود: خام و با apply_aliases.
   WER تخصصی/معمولی هر دو مسیر گزارش می‌شود.

توجه: واریانت‌ها از همین ویدیو برداشت شده‌اند (در-sample) — عدد، سقف
تاثیر را نشان می‌دهد نه عملکرد روی واژه‌های ناشناخته.
"""
from __future__ import annotations

import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf

from app.config import model_dir
from app.asr import _cmp_key, DirectCtcAsrEngine, LiveTranscriber
from app import enroll
from experiment17_reference_window_sweep import (
    WAV, STEP_SEC, build_ref_timeline, extract_audio, load_specialist,
    norm_words, wer_by_class,
)

WINDOW = 12.0  # پیش‌فرض جدید حالت پایدار


def harvest_variants(ref_words, dec_texts, specialist):
    """(واژه‌ی درست، واریانت شنیده‌شده) از هم‌ترازسازی دیکد آفلاین."""
    ref_keys = [_cmp_key(w) for w in ref_words]
    dec_keys = [_cmp_key(w) for w in dec_texts]
    m = SequenceMatcher(None, dec_keys, ref_keys, autojunk=False)
    pairs: dict[tuple[str, str], int] = {}
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag != "replace":
            continue
        for k in range(min(i2 - i1, j2 - j1)):
            rk = ref_keys[j1 + k]
            if rk not in specialist:
                continue
            variant = dec_texts[i1 + k]
            if _cmp_key(variant) == rk:
                continue
            pairs[(ref_words[j1 + k], variant)] = \
                pairs.get((ref_words[j1 + k], variant), 0) + 1
    return sorted(pairs.items(), key=lambda kv: -kv[1])


def main():
    wav_path = extract_audio()
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000

    eng = DirectCtcAsrEngine(model_dir(), num_threads=4, beam_width=2)
    ref_words, ref_times, dec_texts = build_ref_timeline(wav, eng)
    specialist = load_specialist()

    # ۱) ساخت واژه‌های ثبت‌شده از خطاهای واقعی مدل
    found = harvest_variants(ref_words, dec_texts, specialist)
    entries = [
        {"word": word, "variants": [variant]} for (word, variant), _ in found
    ]
    alias_map = enroll.build_alias_map(entries)
    print(f"واژه‌های ثبت‌شده ساخته‌شده از خطاهای مدل ({len(alias_map)} نگاشت):")
    for (word, variant), n in found:
        key = enroll.norm_word(variant)
        mark = "✓" if alias_map.get(key) == word else "✗ (مبهم/حذف)"
        print(f"  {word!r} ← {variant!r} ×{n} {mark}")

    # ۲) شبیه‌سازی زنده — یک‌بار، دو سنجش (خام / با نگاشت)
    live = LiveTranscriber(eng, stable_live=True, window_sec=WINDOW)
    step = int(STEP_SEC * 16000)
    end = step
    latencies: list[float] = []
    acc = {"raw": {"spec": [0, 0], "common": [0, 0]},
           "alias": {"spec": [0, 0], "common": [0, 0]}}
    missed = {"raw": {}, "alias": {}}
    while end <= len(wav):
        buf = wav[max(0, end - int(WINDOW * 16000)):end]
        t_offset = max(0.0, (end - len(buf)) / 16000.0)
        t0 = time.perf_counter()
        text = live.partial(buf, t_offset=t_offset)
        latencies.append(time.perf_counter() - t0)
        if text.strip():
            buf_end = end / 16000.0
            t_start = max(0.0, buf_end - WINDOW)
            expected = [w for w, t in zip(ref_words, ref_times)
                        if t_start <= t < buf_end]
            if expected:
                for label, t_ in (("raw", text),
                                  ("alias", enroll.apply_aliases(text, alias_map))):
                    by = wer_by_class(expected, norm_words(t_), specialist,
                                      missed[label])
                    for cls in ("spec", "common"):
                        acc[label][cls][0] += by[cls][0]
                        acc[label][cls][1] += by[cls][1]
        end += step

    print(f"\n{'مسیر':>8} | {'WER تخصصی':>16} | {'WER معمولی':>16}")
    print("-" * 52)
    for label, title in (("raw", "بدون نگاشت"), ("alias", "با نگاشت")):
        se, sn = acc[label]["spec"]
        ce, cn = acc[label]["common"]
        print(f"{title:>8} | {se}/{sn} ({100 * se / max(1, sn):.1f}%) "
              f"| {ce}/{cn} ({100 * ce / max(1, cn):.1f}%)")

    print("\nبیشترین واژه‌های تخصصی ازدست‌رفته:")
    for label, title in (("raw", "بدون نگاشت"), ("alias", "با نگاشت")):
        top = sorted(missed[label].items(), key=lambda kv: -kv[1])[:5]
        print(f"  {title}: " + (", ".join(f"{w}×{c}" for w, c in top) or "—"))

    print(f"\nمیانگین tick: {sum(latencies) / len(latencies) * 1000:.0f}ms "
          f"(هزینه‌ی apply_aliases ناچیز است — پس‌پردازش متن)")


if __name__ == "__main__":
    main()
