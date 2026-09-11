"""آزمایش ۵ — تکرار آزمایش‌های قبلی روی ۳۵۰ کلیپ از ۷ منبع (تن‌عامد).

روی هر کلیپ، همان حلقه‌ی زنده‌ی برنامه شبیه‌سازی می‌شود (پنجره‌ی 10s،
گام 0.8s) و برچسب «واژه‌ی خوب/بد» از متن مرجعِ manifest می‌آید —
نه از decode خود مدل. سپس همان سیگنال‌های آزمایش ۳/۴ سنجیده می‌شوند:
سرعت گفتار (WPM فعال/محلی)، conf، margin، رأی پنجره‌های قبلی (votes)،
contest، لبه‌ی پنجره — با AUC، به‌علاوه‌ی خطای به تفکیک منبع.

فاز ۱ (این اسکریپت): decode همه‌ی پنجره‌ها و کش در windows_multi.pkl
فاز ۲ (experiment6.py): تحلیل سیگنال‌ها روی کش
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
PER_SOURCE = 50          # حداقل ۵۰ نمونه از هر منبع
SEED = 42
CACHE = ROOT / "spikes" / "speech_rate" / "windows_multi.pkl"
MANIFEST = ROOT / "training" / "all" / "manifest.jsonl"
N_WORKERS = 4           # ۸ هسته — هر worker یک session ONNX با ۱ thread
MS_PER_MODEL_FRAME = 80.0 / 1000.0


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def decode_clip_windows(args):
    """decode همه‌ی پنجره‌های یک کلیپ — در worker process جدا.

    خروجی: dict با id، متن مرجع، و پنجره‌ها [(t_start, text, words)]
    هر word: text, conf, margin, f0, f1 (فریم مدل، 80ms/فریم)
    """
    clip_id, audio_path, ref_text, dur = args
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT))
    from app.asr import _CtcHypothesisDecoder

    samples = load_wav(Path(audio_path))
    if samples.size == 0:
        return {"id": clip_id, "ref": ref_text, "duration": dur,
                "windows": []}

    # هر worker یک بار session می‌سازد و برای همه‌ی کلیپ‌هایش نگه می‌دارد
    global _DECODER
    try:
        _DECODER
    except NameError:
        _DECODER = _CtcHypothesisDecoder(
            ROOT / "model", num_threads=1, beam_width=2)

    win_sp = int(WIN_SEC * SR)
    step_sp = int(STEP_SEC * SR)
    windows = []
    start = 0
    # پنجره‌ها همیشه از انتهای بافر بریده می‌شوند؛ برای کلیپ‌های کوتاه‌تر
    # از 10s، پنجره = کل کلیپ (رفتار واقعی برنامه با بافر کوتاه).
    while start + 1600 <= samples.size:
        win = samples[start:min(start + win_sp, samples.size)]
        t_start = start / SR
        hyp = _DECODER.decode(win, previous_text=None)
        words = []
        # بازخوانی span از tokens برای زمان‌بندی
        # (decode خودش tokens را مصرف می‌کند؛ دوباره decode نمی‌کنیم بلکه
        # از همان log_probs مسیر می‌رویم؟ — نه: در این spike، decode دوباره
        # با run مستقیم انجام می‌شود تا span ها در دسترس باشند.)
        words = _decode_with_spans(_DECODER, win, t_start)
        windows.append({
            "t_start": t_start, "text": hyp.text,
            "conf": hyp.confidence, "margin": hyp.margin,
            "words": words,
        })
        start += step_sp
    return {"id": clip_id, "ref": ref_text, "duration": dur,
            "windows": windows}


def _decode_with_spans(dec, samples, t_start):
    """مثل _CtcHypothesisDecoder.decode اما با بازگرداندن span فریم‌ها."""
    import numpy as np
    from app.asr import _softmax, _word_confidences

    feat = dec._kaldi_fbank(samples, SR)
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
    beams = dec._decoder.decode_beams(
        log_probs, beam_width=dec._beam_width,
        beam_prune_logp=-5.0, token_min_logp=-5.0,
    )
    if not beams:
        return []
    tokens = beams[0][2]
    probs = _softmax(log_probs)
    wc = _word_confidences(log_probs, tokens, dec._blank_id, probs=probs)
    out = []
    for (tok_text, (f0, f1)), w in zip(tokens, wc):
        out.append({
            "text": w.text, "conf": round(w.confidence, 4),
            "margin": round(w.margin, 4),
            "f0": int(f0), "f1": int(max(f1, f0 + 1)),
        })
    return out


def main():
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    # 50 از هر منبع (mixed train/dev — هدف سنجش رفتار زنده است نه ارزیابی مدل)
    rng = random.Random(SEED)
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    chosen = []
    for src, lst in sorted(by_src.items()):
        pool = [r for r in lst if 2.5 <= r["duration"] <= 13.5]
        take = rng.sample(pool, min(PER_SOURCE, len(pool)))
        chosen.extend(take)
        print(f"  {src}: {len(take)} clips")
    print(f"[info] total {len(chosen)} clips "
          f"({sum(r['duration'] for r in chosen)/60:.1f} min audio)")

    jobs = []
    for r in chosen:
        audio = (MANIFEST.parent / r["audio"]).resolve()
        jobs.append((r["id"], str(audio), r["text"], r["duration"]))

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(decode_clip_windows, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(jobs)} clips "
                      f"({el:.0f}s, ~{el/(i+1):.1f}s/clip)", flush=True)
    print(f"[done] decoded in {time.perf_counter() - t0:.0f}s")

    CACHE.write_bytes(pickle.dumps({
        "win_sec": WIN_SEC, "step_sec": STEP_SEC,
        "ms_per_frame": MS_PER_MODEL_FRAME, "clips": results,
    }))
    n_wins = sum(len(c["windows"]) for c in results)
    print(f"[done] {len(results)} clips, {n_wins} windows → {CACHE}")


if __name__ == "__main__":
    main()
