"""آزمایش سرعت گفتار و اطمینان قفل پیشوند روی صدای واقعی.

چهار پرسش را روی صدای استخراج‌شده از sample/ می‌سنجد:
Q1) آیا اطمینانِ CTC با سرعت گفتار (واژه/دقیقه) همبستگی دارد؟
Q2) چقدر «واژه‌ی اشتباه با اطمینان بالا» رایج است؟ آستانه‌ها چطور خطا/صحیح را تفکیک می‌کنند؟
Q3) رفتار قفل پیشوند (LiveTranscriber واقعی) روی پنجره‌های متوالی چطور است؟
Q4) آیا تفسیر «سرعت گفتار» به‌عنوان سیگنال پایداری، بهتر از اطمینان خام است؟

خروجی: spikes/speech_rate/report.json + چاپ خلاصه.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.asr import DirectCtcAsrEngine, LiveTranscriber  # noqa: E402

WAV = ROOT / "spikes" / "speech_rate" / "sample_16k.wav"
SR = 16000
WIN_SEC = 10.0        # همان window_sec برنامه
STEP_SEC = 0.8        # همان PARTIAL_INTERVAL برنامه
MIN_WORD_CONF = 0.25   # همان مقدار پیش‌فرض LiveTranscriber


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    samples = load_wav(WAV)
    duration = samples.size / SR
    print(f"[info] audio: {duration:.1f}s")

    t0 = time.perf_counter()
    engine = DirectCtcAsrEngine(ROOT / "model", num_threads=4, beam_width=2)
    print(f"[info] engine loaded in {time.perf_counter() - t0:.2f}s")

    # ---- مرجع: decode مستقل هر chunk 16s (مثل transcribe نهایی) ----
    t0 = time.perf_counter()
    chunk_sec = 16.0
    chunk_sp = int(chunk_sec * SR)
    ref_chunks = []  # (t_start, t_end, [words])
    for cs in range(0, samples.size, chunk_sp):
        ce = min(cs + chunk_sp, samples.size)
        txt = engine.transcribe(samples[cs:ce])
        ref_chunks.append((cs / SR, ce / SR, txt.split()))
    ref_words_total = sum(len(w) for _, _, w in ref_chunks)
    ref_text = " ".join(" ".join(w) for _, _, w in ref_chunks)
    ref_rt = time.perf_counter() - t0
    print(f"[info] reference: {ref_words_total} words in "
          f"{ref_chunks.__len__()} chunks ({ref_rt:.1f}s)")
    print("  " + ref_text[:300])

    # ---- شبیه‌سازی حلقه‌ی زنده: پنجره‌ی 10s، گام 0.8s، یک decode در هر گام ----
    win_sp = int(WIN_SEC * SR)
    step_sp = int(STEP_SEC * SR)

    live = LiveTranscriber(
        engine, window_sec=WIN_SEC, stable_live=True,
        margin=0.02, hysteresis=2, min_word_confidence=MIN_WORD_CONF,
    )

    windows = []      # per-step raw decode (t_start, t_end, hyp, decode_s)
    lock_states = []  # (t, live_text, n_words)
    prev_live = None
    start = 0
    while start + 1600 <= samples.size:
        win = samples[start:min(start + win_sp, samples.size)]
        t_start, t_end = start / SR, (start + win.size) / SR
        t0 = time.perf_counter()
        hyp = engine.transcribe_with_details(win)
        dt = time.perf_counter() - t0
        windows.append((t_start, t_end, hyp, dt))

        # همان ورودی‌ای که برنامه به LiveTranscriber می‌دهد (پنجره‌ی انتهای بافر)
        shown = live.partial_result(win)
        lock_states.append((t_start, shown.text, len(shown.text.split())))
        prev_live = shown.text
        start += step_sp

    print(f"[info] simulated {len(windows)} live windows "
          f"(win={WIN_SEC}s, step={STEP_SEC}s)")

    # ============ Q1: سرعت گفتار vs اطمینان ============
    rows = []
    for t_start, t_end, hyp, dt in windows:
        if not hyp.text.strip():
            continue
        n = len(hyp.text.split())
        wpm = n / (WIN_SEC / 60.0)
        rows.append({
            "t": round(t_start, 2), "n_words": n, "wpm": round(wpm, 1),
            "confidence": round(hyp.confidence, 4), "margin": round(hyp.margin, 4),
            "decode_s": round(dt, 3),
        })
    wpm_a = np.array([r["wpm"] for r in rows])
    conf_a = np.array([r["confidence"] for r in rows])
    marg_a = np.array([r["margin"] for r in rows])
    nwords_a = np.array([r["n_words"] for r in rows])
    q1 = {
        "n_windows": len(rows),
        "wpm_mean": float(wpm_a.mean()) if rows else None,
        "wpm_min": float(wpm_a.min()) if rows else None,
        "wpm_max": float(wpm_a.max()) if rows else None,
        "conf_mean": float(conf_a.mean()) if rows else None,
        "conf_min": float(conf_a.min()) if rows else None,
        "conf_max": float(conf_a.max()) if rows else None,
        "pearson_wpm_conf": pearson(wpm_a, conf_a),
        "pearson_wpm_margin": pearson(wpm_a, marg_a),
        "pearson_nwords_conf": pearson(nwords_a, conf_a),
    }
    print(f"\n[Q1] speech rate vs confidence:")
    print(f"  wpm: {q1['wpm_min']:.0f}..{q1['wpm_max']:.0f} (mean {q1['wpm_mean']:.0f})")
    print(f"  conf: {q1['conf_min']:.3f}..{q1['conf_max']:.3f} (mean {q1['conf_mean']:.3f})")
    print(f"  pearson(wpm, conf)   = {q1['pearson_wpm_conf']:+.3f}")
    print(f"  pearson(wpm, margin) = {q1['pearson_wpm_margin']:+.3f}")

    # ============ Q2: واژه‌های اشتباه با اطمینان بالا ============
    # مرجع word-timing: توزیع یکنواخت واژه‌ها در chunk (تقریب)
    def ref_words_in(t_start, t_end) -> set:
        out = set()
        for cs, ce, ws in ref_chunks:
            if ce <= t_start or cs >= t_end or not ws:
                continue
            for i, w in enumerate(ws):
                wt = cs + (i + 0.5) / len(ws) * (ce - cs)
                if t_start <= wt < t_end:
                    out.add(w)
        return out

    word_obs = []  # (word, conf, margin, in_ref)
    for t_start, t_end, hyp, _ in windows:
        if not hyp.words:
            continue
        ref_span = ref_words_in(t_start, t_end)
        for w in hyp.words:
            word_obs.append((w.text, w.confidence, w.margin, w.text in ref_span))

    # سخت‌گیر: واژه باید عیناً در مرجع هم‌بازه باشد؛ آسان‌گیر: اگر در هیچ
    # chunk مرجعی نیامده باشد خطاست (واژه‌ای که مدل در هیچ decode مستقیمی
    # نگرفته احتمالاً ساخته‌ی نوسان پنجره است)
    all_ref_words = set()
    for _, _, ws in ref_chunks:
        all_ref_words.update(ws)

    strict_wrong = [x for x in word_obs if not x[3]]
    lenient_wrong = [x for x in word_obs if x[0] not in all_ref_words]
    correct = [x for x in word_obs if x[3]]

    def thr_table(wrong, right, label):
        print(f"  [{label}] wrong={len(wrong)} right={len(right)}")
        tbl = []
        for thr in (0.25, 0.5, 0.7, 0.9, 0.96):
            rej = sum(1 for x in wrong if x[1] < thr)
            drop = sum(1 for x in right if x[1] < thr)
            tbl.append({"threshold": thr,
                        "rejects_errors": rej, "total_errors": len(wrong),
                        "drops_correct": drop, "total_correct": len(right)})
            print(f"    thr={thr:.2f}: rejects {rej}/{len(wrong)} errors, "
                  f"drops {drop}/{len(right)} correct")
        return tbl

    print(f"\n[Q2] word error analysis ({len(word_obs)} observations):")
    q2 = {
        "word_observations": len(word_obs),
        "strict_wrong": len(strict_wrong),
        "lenient_wrong": len(lenient_wrong),
        "strict_wrong_conf_mean": float(np.mean([x[1] for x in strict_wrong])) if strict_wrong else None,
        "correct_conf_mean": float(np.mean([x[1] for x in correct])) if correct else None,
        "strict_wrong_margin_mean": float(np.mean([x[2] for x in strict_wrong])) if strict_wrong else None,
        "correct_margin_mean": float(np.mean([x[2] for x in correct])) if correct else None,
        "threshold_table_strict": thr_table(strict_wrong, correct, "strict"),
        "threshold_table_lenient": thr_table(
            lenient_wrong,
            [x for x in word_obs if x[0] in all_ref_words],
            "lenient"),
        "hi_conf_errors": [
            {"word": x[0], "conf": round(x[1], 3), "margin": round(x[2], 3)}
            for x in strict_wrong if x[1] >= 0.9
        ][:40],
        "sample_wrong_words": [x[0] for x in strict_wrong[:40]],
    }
    if strict_wrong:
        print(f"  conf(wrong)={q2['strict_wrong_conf_mean']:.3f} vs "
              f"conf(correct)={q2['correct_conf_mean']:.3f}")
        print(f"  margin(wrong)={q2['strict_wrong_margin_mean']:.3f} vs "
              f"margin(correct)={q2['correct_margin_mean']:.3f}")

    # ============ Q3: رفتار قفل پیشوند ============
    changes = []
    prev = None
    for t, txt, n in lock_states:
        if txt != prev:
            changes.append({"t": round(t, 2), "n_words": n, "text": txt[:120]})
            prev = txt
    kinds = Counter()
    prev_n = None
    for e in changes:
        if prev_n is not None:
            if e["n_words"] > prev_n:
                kinds["grow"] += 1
            elif e["n_words"] < prev_n:
                kinds["shrink"] += 1
            else:
                kinds["edit"] += 1
        prev_n = e["n_words"]
    q3 = {
        "n_steps": len(lock_states),
        "n_changes": len(changes),
        "kinds": dict(kinds),
        "changes": changes[:60],
    }
    print(f"\n[Q3] prefix-lock behaviour over {len(lock_states)} steps:")
    print(f"  text changed {len(changes)} times; kinds={dict(kinds)}")
    for e in changes[:10]:
        print(f"   t={e['t']:6.2f} n={e['n_words']:3d}  {e['text']}")

    # نوسان خام (بدون قفل) برای مقایسه: چند بار متنِ خام پنجره عوض شد؟
    raw_changes = 0
    prev_raw = None
    for _, _, hyp, _ in windows:
        if hyp.text != prev_raw:
            raw_changes += 1
            prev_raw = hyp.text
    print(f"  raw window text changed {raw_changes} times (without lock)")
    q3["raw_changes_without_lock"] = raw_changes

    # ============ Q4: سرعت گفتار به‌عنوان سیگنال پایداری ============
    # اگر WPM پنجره‌ی فعلی خیلی بالاتر از میانگین شخصی باشد، پنجره احتمالا
    # در حال «خوشه‌های خطا» است (دستِ عصبی مدل روی صدای سریع/نویزی).
    # سنجش: آیا پنجره‌هایی که WPM بالاتری دارند، واژه‌ی خطای بیشتری دارند؟
    per_win_err = []
    for t_start, t_end, hyp, _ in windows:
        if not hyp.words or not hyp.text.strip():
            continue
        ref_span = ref_words_in(t_start, t_end)
        n_err = sum(1 for w in hyp.words if w.text not in ref_span)
        per_win_err.append({
            "t": round(t_start, 2),
            "wpm": len(hyp.text.split()) / (WIN_SEC / 60.0),
            "n_err": n_err, "n_words": len(hyp.words),
            "err_rate": n_err / max(1, len(hyp.words)),
        })
    wpm_e = np.array([r["wpm"] for r in per_win_err])
    err_e = np.array([r["err_rate"] for r in per_win_err])
    q4 = {
        "pearson_wpm_errrate": pearson(wpm_e, err_e),
        "top_wpm_windows": sorted(per_win_err, key=lambda r: -r["wpm"])[:10],
        "low_wpm_windows": sorted(per_win_err, key=lambda r: r["wpm"])[:10],
    }
    print(f"\n[Q4] speech rate vs per-window error rate:")
    print(f"  pearson(wpm, err_rate) = {q4['pearson_wpm_errrate']:+.3f}")
    hi = [r for r in per_win_err if r["wpm"] >= np.median(wpm_e)]
    lo = [r for r in per_win_err if r["wpm"] < np.median(wpm_e)]
    if hi and lo:
        print(f"  err_rate: fast-half mean={np.mean([r['err_rate'] for r in hi]):.3f}, "
              f"slow-half mean={np.mean([r['err_rate'] for r in lo]):.3f}")

    # ---- خروجی ----
    out = {
        "audio_duration_s": round(duration, 2),
        "config": {"win_sec": WIN_SEC, "step_sec": STEP_SEC,
                   "min_word_confidence": MIN_WORD_CONF},
        "reference_words": ref_words_total,
        "reference_text": ref_text,
        "q1_rate_vs_conf": q1,
        "q2_error_analysis": q2,
        "q3_lock": q3,
        "q4_rate_vs_error": q4,
        "windows": rows,
    }
    out_path = Path(__file__).parent / "report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] report → {out_path}")


if __name__ == "__main__":
    main()
