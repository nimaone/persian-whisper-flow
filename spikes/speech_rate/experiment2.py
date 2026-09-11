"""آزمایش ۲ — سنجش راه‌حل‌های خلاقانه برای قفل پیشوند.

روی همان صدای نمونه، سه سیگنال جدید را در برابر اطمینان فعلی می‌سنجد:
E1) WPM فعالِ گفتاری (کلید ایده‌ی کاربر): واژه بر دقیقه‌ی *صدا*، نه پنجره.
    WPM خام با سکوت هم‌پوشان است (پنجره‌ی ساکت = WPM پایین + conf پایین).
E2) رأی‌گیری پایداری واژه: هر واژه در چند پنجره‌ی متوالی هم‌پوشان دیده می‌شود؛
    واژه‌ای که در پنجره‌های پیاپی یکسان تکرار شود «پایدار» است. آیا پایداری
    خطا را بهتر از confidence پیش‌بینی می‌کند؟
E3) طول واژه (تعداد فریم): واژه‌های توهمی CTC معمولاً span کوتاه دارند.

مقایسه با AUC (سطح زیر منحنی ROC) برای جدا کردن واژه‌های خطا از صحیح.
خروجی: spikes/speech_rate/report2.json + چاپ خلاصه.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.asr import _CtcHypothesisDecoder  # noqa: E402
from app.hotword_asr import kaldi_fbank, load_labels  # noqa: E402

WAV = ROOT / "spikes" / "speech_rate" / "sample_16k.wav"
CACHE = ROOT / "spikes" / "speech_rate" / "windows.pkl"
SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
FRAME_SEC = 0.01  # frame_shift=160 → 10ms
SILENCE_RMS = 0.003  # همان آستانه‌ی برنامه


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def decode_all(samples: np.ndarray) -> list[dict]:
    """همه‌ی پنجره‌ها را decode می‌کند؛ شامل span فریم هر واژه."""
    dec = _CtcHypothesisDecoder(ROOT / "model", num_threads=4, beam_width=2)
    import onnxruntime as ort  # noqa: F401
    from pyctcdecode import Alphabet, BeamSearchDecoderCTC

    labels = load_labels(ROOT / "model" / "tokens.txt")
    blank_id = labels.index("")
    beam_decoder = BeamSearchDecoderCTC(Alphabet(labels, is_bpe=True))

    win_sp = int(WIN_SEC * SR)
    step_sp = int(STEP_SEC * SR)
    out = []
    start = 0
    while start + 1600 <= samples.size:
        win = samples[start:min(start + win_sp, samples.size)]
        t_start = start / SR
        feat = kaldi_fbank(win, SR)
        with dec._lock:
            log_probs, lengths = dec._session.run(
                None,
                {
                    "audio_signal": feat.T[None].astype(np.float32),
                    "length": np.array([feat.shape[0]], dtype=np.int64),
                },
            )
        length = int(lengths[0])
        log_probs = log_probs[0][:length]
        beams = beam_decoder.decode_beams(
            log_probs, beam_width=2, beam_prune_logp=-5.0, token_min_logp=-5.0,
        )
        if not beams:
            out.append({"t_start": t_start, "words": []})
            start += step_sp
            continue
        top = beams[0]
        tokens = top[2]
        text = top[0].strip()
        # softmax یک‌بار؛ برای conf/margin واژه‌ها (همان منطق برنامه)
        from app.asr import _softmax, _word_confidences
        probs = _softmax(log_probs)
        words_meta = _word_confidences(
            log_probs, tokens, blank_id, probs=probs)
        # span فریم هر واژه → زمان مطلق
        words = []
        for (tok_text, (f0, f1)), w in zip(tokens, words_meta):
            words.append({
                "text": w.text, "conf": round(w.confidence, 4),
                "margin": round(w.margin, 4),
                "t0": round(t_start + max(f0, 0) * FRAME_SEC, 3),
                "t1": round(t_start + max(f1, f0 + 1) * FRAME_SEC, 3),
                "frames": max(f1 - f0, 1),
            })
        out.append({
            "t_start": t_start, "t_end": t_start + win.size / SR,
            "text": text, "words": words,
        })
        start += step_sp
    dec.release()
    return out


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(scores_pos, scores_neg) -> float:
    """AUC = P(score_pos > score_neg)؛ pos = کلاس «خطا»."""
    p, n = np.asarray(scores_pos, dtype=float), np.asarray(scores_neg, dtype=float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    order = np.concatenate([p, n])
    ranks = order.argsort().argsort().astype(float) + 1
    # rank ties را ساده می‌گیریم (خطای کوچک برای این آزمایش)
    rp = ranks[:p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2) / (p.size * n.size))


def speech_active_sec(win: np.ndarray) -> float:
    """ثانیه‌های فعال بر اساس RMS بلوک‌های 100ms (همان روح SILENCE_RMS)."""
    blk = int(0.1 * SR)
    n = win.size // blk
    if n == 0:
        return 0.0
    rms = np.sqrt((win[:n * blk].reshape(n, blk) ** 2).mean(axis=1))
    return float((rms > SILENCE_RMS).sum() * 0.1)


def main():
    samples = load_wav(WAV)
    duration = samples.size / SR
    print(f"[info] audio: {duration:.1f}s")

    # ---- decode پنجره‌ها (با cache) ----
    if CACHE.exists():
        wins = pickle.loads(CACHE.read_bytes())
        print(f"[info] loaded {len(wins)} windows from cache")
    else:
        t0 = time.perf_counter()
        wins = decode_all(samples)
        print(f"[info] decoded {len(wins)} windows in "
              f"{time.perf_counter() - t0:.1f}s")
        CACHE.write_bytes(pickle.dumps(wins))

    # ---- مرجع word-timing (uniform در chunkهای 16s) ----
    dec_ref = _CtcHypothesisDecoder(ROOT / "model", num_threads=4, beam_width=2)
    chunk_sp = int(16.0 * SR)
    ref_words = []  # (word, t_abs)
    for cs in range(0, samples.size, chunk_sp):
        ce = min(cs + chunk_sp, samples.size)
        txt = dec_ref.decode(samples[cs:ce]).text
        ws = txt.split()
        for i, w in enumerate(ws):
            ref_words.append(
                (w, cs / SR + (i + 0.5) / max(1, len(ws)) * (ce - cs) / SR))
    dec_ref.release()
    all_ref_vocab = {w for w, _ in ref_words}
    print(f"[info] reference: {len(ref_words)} timed words")

    def word_in_ref(text: str, t0: float, t1: float) -> bool:
        """واژه در مرجع، در بازه‌ی زمانی خودش؟"""
        return any(w == text and t0 - 0.6 <= t < t1 + 0.6
                   for w, t in ref_words)

    # ============ E2: رأی‌گیری پایداری واژه ============
    # هر واژه را در پنجره‌های بعدی دنبال می‌کنیم: چند پنجره‌ی پیاپی
    # همان واژه را در همان بازه‌ی زمانی تکرار می‌کنند؟
    instances = []  # (win_idx, text, conf, margin, t0, t1, frames)
    for wi, w in enumerate(wins):
        for word in w["words"]:
            instances.append((wi, word["text"], word["conf"], word["margin"],
                              word["t0"], word["t1"], word["frames"]))

    def streak(inst) -> tuple[int, int]:
        """(تعداد پنجره‌های پیاپی با همان متن، تعداد پنجره‌های رقیب)"""
        wi, text, _, _, t0, t1, _ = inst
        agree, contest = 1, 0
        for wj in range(wi + 1, min(wi + 10, len(wins))):
            for word in wins[wj]["words"]:
                ov = min(t1, word["t1"]) - max(t0, word["t0"])
                if ov > 0.5 * min(t1 - t0, word["t1"] - word["t0"]):
                    if word["text"] == text:
                        agree += 1
                    else:
                        contest += 1
        return agree, contest

    rows = []
    seen = set()
    for inst in instances:
        wi, text, conf, marg, t0, t1, frames = inst
        key = (text, round(t0 * 2) / 2)
        if key in seen:  # همان واژه از پنجره‌ی بعدی — یک‌بار حساب می‌شود
            continue
        # فقط استرک-استارت‌ها: پنجره‌ی قبلی این واژه را نداشته باشد
        prev_has = False
        if wi > 0:
            for word in wins[wi - 1]["words"]:
                ov = min(t1, word["t1"]) - max(t0, word["t0"])
                if ov > 0.5 * min(t1 - t0, word["t1"] - word["t0"]) \
                        and word["text"] == text:
                    prev_has = True
                    break
        if prev_has:
            seen.add(key)
            continue
        seen.add(key)
        agree, contest = streak(inst)
        rows.append({
            "text": text, "conf": conf, "margin": marg,
            "t0": t0, "t1": t1, "frames": frames,
            "streak": agree, "contest": contest,
            "dur": round(t1 - t0, 3),
            "wrong": not word_in_ref(text, t0, t1),
        })
    n_wrong = sum(1 for r in rows if r["wrong"])
    print(f"\n[E2] word events: {len(rows)}, wrong: {n_wrong}")

    wrong = [r for r in rows if r["wrong"]]
    right = [r for r in rows if not r["wrong"]]
    sig = {
        "conf": auc([r["conf"] for r in wrong], [r["conf"] for r in right]),
        "margin": auc([r["margin"] for r in wrong], [r["margin"] for r in right]),
        "streak": auc([r["streak"] for r in wrong], [r["streak"] for r in right]),
        "frames": auc([r["frames"] for r in wrong], [r["frames"] for r in right]),
        "conf*margin": auc([r["conf"] * r["margin"] for r in wrong],
                           [r["conf"] * r["margin"] for r in right]),
        "streak*conf": auc([r["streak"] * r["conf"] for r in wrong],
                           [r["streak"] * r["conf"] for r in right]),
    }
    print("  AUC برای پیش‌بینی خطا (۱=کاملاً تفکیک‌کننده، ۰.۵=بی‌سیگنال):")
    for k, v in sig.items():
        print(f"    {k:12s} AUC={v:.3f}")
    if wrong and right:
        print(f"  mean streak: wrong={np.mean([r['streak'] for r in wrong]):.2f}, "
              f"right={np.mean([r['streak'] for r in right]):.2f}")
        print(f"  mean frames: wrong={np.mean([r['frames'] for r in wrong]):.1f}, "
              f"right={np.mean([r['frames'] for r in right]):.1f}")

    # نمونه‌ی خروجی برای رأی‌گیری: واژه‌های پرتنازع
    contested = sorted(rows, key=lambda r: -r["contest"])[:15]
    print("  واژه‌های پرتنازع (متن‌های رقیب در پنجره‌های بعدی):")
    for r in contested:
        print(f"    {r['text']!r:30s} streak={r['streak']:2d} contest={r['contest']:2d} "
              f"conf={r['conf']:.2f} wrong={r['wrong']}")

    # ============ E1: WPM فعال vs خام ============
    win_sp = int(WIN_SEC * SR)
    e1_rows = []
    for w in wins:
        t0 = w["t_start"]
        win = samples[int(t0 * SR):int(t0 * SR) + win_sp]
        active = speech_active_sec(win)
        n = len(w["words"])
        if active < 0.5:
            continue
        # خطای پنجره: واژه‌هایی که در مرجع هم‌بازه نیستند
        n_err = sum(1 for word in w["words"]
                    if not word_in_ref(word["text"], word["t0"], word["t1"]))
        e1_rows.append({
            "t": round(t0, 2), "n": n, "active_s": round(active, 2),
            "wpm_raw": n / (WIN_SEC / 60),
            "wpm_active": n / (active / 60),
            "err_rate": n_err / max(1, n),
        })
    raw_a = np.array([r["wpm_raw"] for r in e1_rows])
    act_a = np.array([r["wpm_active"] for r in e1_rows])
    err_a = np.array([r["err_rate"] for r in e1_rows])
    conf_by_t = {round(w["t_start"], 2):
                 np.mean([x["conf"] for x in w["words"]]) if w["words"] else 0.0
                 for w in wins}
    conf_a = np.array([conf_by_t[r["t"]] for r in e1_rows])
    e1 = {
        "n_windows": len(e1_rows),
        "wpm_raw_mean": float(raw_a.mean()),
        "wpm_active_mean": float(act_a.mean()),
        "wpm_active_p10": float(np.percentile(act_a, 10)),
        "wpm_active_p90": float(np.percentile(act_a, 90)),
        "pearson_wpmraw_conf": pearson(raw_a, conf_a),
        "pearson_wpmactive_conf": pearson(act_a, conf_a),
        "pearson_wpmraw_err": pearson(raw_a, err_a),
        "pearson_wpmactive_err": pearson(act_a, err_a),
    }
    print(f"\n[E1] speech-active WPM over {len(e1_rows)} windows:")
    print(f"  WPM خام mean={e1['wpm_raw_mean']:.0f} | "
          f"WPM فعال mean={e1['wpm_active_mean']:.0f} "
          f"(p10={e1['wpm_active_p10']:.0f}, p90={e1['wpm_active_p90']:.0f})")
    print(f"  corr(WPM خام, conf)     = {e1['pearson_wpmraw_conf']:+.3f}  ← هم‌پوشانی با سکوت")
    print(f"  corr(WPM فعال, conf)    = {e1['pearson_wpmactive_conf']:+.3f}")
    print(f"  corr(WPM خام, err)      = {e1['pearson_wpmraw_err']:+.3f}")
    print(f"  corr(WPM فعال, err)     = {e1['pearson_wpmactive_err']:+.3f}")

    # ---- خروجی ----
    out = {
        "audio_duration_s": round(duration, 2),
        "e1_active_wpm": e1,
        "e2_stability": {
            "word_events": len(rows), "wrong": n_wrong,
            "auc": sig,
            "wrong_mean_streak": float(np.mean([r["streak"] for r in wrong])) if wrong else None,
            "right_mean_streak": float(np.mean([r["streak"] for r in right])) if right else None,
            "contested_examples": [
                {k: r[k] for k in ("text", "streak", "contest", "conf", "wrong")}
                for r in contested],
        },
        "word_rows": rows,
        "win_rows": e1_rows,
    }
    out_path = Path(__file__).parent / "report2.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] report → {out_path}")


if __name__ == "__main__":
    main()
