"""آزمایش ۸ — رگرسیون صادقانه: LiveTranscriber روی بافر پیوسته‌ی واقعی.

از هر WAV خام training/raw یک بازه‌ی پیوسته بریده می‌شود و حلقه‌ی
زنده‌ی برنامه واقعاً روی آن اجرا می‌شود:
  - پنجره‌ی 10s، گام 0.8s (PARTIAL_INTERVAL) — decode با همان دیکدر
  - LiveTranscriber.partial_result با بافر رشدی (مثل rec.get_tail_16k)
مرجع دقت: decode مستقل chunkهای 16s همان بازه (همان مسیر final برنامه).

مقایسه: قوانین خاموش (رفتار پایه) در برابر روشن (contest/edge/votes).
خروجی: spikes/speech_rate/report_live_regression.json + کش windows_real.pkl
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "training" / "raw"
SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
SEGMENT_SEC = 360.0     # 6 دقیقه از هر منبع
SEGMENT_START = 120.0    # از دقیقه‌ی ۲ (رد کردن مقدمه/سکوت ابتدایی)
N_WORKERS = 4
CACHE = ROOT / "spikes" / "speech_rate" / "windows_real.pkl"


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def load_wav_span(path: Path, t0: float, dur: float) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        assert sr == SR
        w.setpos(int(t0 * sr))
        raw = np.frombuffer(w.readframes(int(dur * sr)), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def process_source(args):
    """یک منبع: decode پنجره‌های زنده + مرجع chunk — در worker جدا."""
    name, wav_path = args
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT))
    from app.asr import DirectCtcAsrEngine

    global _ENG
    try:
        _ENG
    except NameError:
        _ENG = DirectCtcAsrEngine(ROOT / "model", num_threads=1, beam_width=2)

    samples = load_wav_span(Path(wav_path), SEGMENT_START, SEGMENT_SEC)
    win_sp = int(WIN_SEC * SR)
    step_sp = int(STEP_SEC * SR)

    # پنجره‌های زنده — پنجره‌ی متحرکِ واقعی: بافر رشدی و پنجره = انتهایش
    # previous_text=None: میان‌بر «متن تکراری بدون words» دیکدر را
    # غیرفعال می‌کند؛ در بازپخش، هر پنجره باید words کامل داشته باشد.
    windows = []
    for buf_end in range(int(1.0 * SR), samples.size + 1, step_sp):
        win = samples[:buf_end][-win_sp:]
        t_start = (buf_end - win.size) / SR
        hyp = _ENG.transcribe_with_details(win, previous_text=None)
        windows.append({
            "t_start": t_start, "text": hyp.text,
            "conf": hyp.confidence, "margin": hyp.margin,
            "words": [
                {"text": w.text, "conf": w.confidence, "margin": w.margin,
                 "f0": w.start, "f1": w.end}
                for w in hyp.words
            ],
        })

    # مرجع: chunkهای 16s (مسیر final)
    chunk_sp = int(16.0 * SR)
    ref_chunks = []
    for cs in range(0, samples.size, chunk_sp):
        ce = min(cs + chunk_sp, samples.size)
        txt = _ENG.transcribe(samples[cs:ce])
        ref_chunks.append((cs / SR, ce / SR, txt))
    return {"source": name, "windows": windows, "ref_chunks": ref_chunks,
            "duration": SEGMENT_SEC}


def main():
    jobs = []
    for wav in sorted(RAW.glob("*_16k.wav")):
        name = wav.name.replace("_16k.wav", "")
        jobs.append((name, str(wav)))
    print(f"[info] {len(jobs)} sources × {SEGMENT_SEC/60:.0f}min")

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for res in ex.map(process_source, jobs):
            results.append(res)
            print(f"  {res['source']}: {len(res['windows'])} windows "
                  f"(+{time.perf_counter()-t0:.0f}s)", flush=True)
    CACHE.write_bytes(pickle.dumps(results))
    print(f"[done] decode in {time.perf_counter()-t0:.0f}s → {CACHE}")

    # ---- بازپخش با LiveTranscriber واقعی ----
    from app.asr import AsrHypothesis, AsrWord, LiveTranscriber

    class ReplayEngine:
        def __init__(self, windows):
            self.windows = windows
            self.calls = 0

        def transcribe_with_details(self, samples, sample_rate=16000,
                                    previous_text=None):
            w = self.windows[min(self.calls, len(self.windows) - 1)]
            self.calls += 1
            words = tuple(
                AsrWord(x["text"], x["conf"], x["margin"],
                        start=x["f0"], end=x["f1"])
                for x in w["words"]
            )
            confs = [x["conf"] for x in w["words"]] or [0.0]
            return AsrHypothesis(w["text"], float(np.mean(confs)), 0.0, words)

    def ref_norm_in(t0, t1, ref_chunks):
        out = []
        for cs, ce, txt in ref_chunks:
            if ce <= t0 or cs >= t1:
                continue
            ws = [norm_word(x) for x in txt.split()]
            if not ws:
                continue
            for i, w in enumerate(ws):
                wt = cs + (i + 0.5) / len(ws) * (ce - cs)
                if t0 - 0.4 <= wt <= t1 + 0.4:
                    out.append(w)
        return out

    out_all = {}
    for rules_on in (False, True):
        label = "قوانین روشن" if rules_on else "قوانین خاموش (پایه)"
        precs, recs, changes, empties = [], [], 0, 0
        for res in results:
            live = LiveTranscriber(
                ReplayEngine(res["windows"]),
                window_sec=WIN_SEC, stable_live=True,
            )
            if not rules_on:
                live.max_contest = 1.01
                live.lock_edge_sec = 0.0
                live.vote_high_conf = 0.0
            prev = None
            for wi, w in enumerate(res["windows"]):
                buf_end = w["t_start"] + WIN_SEC
                n = int(min(buf_end, SEGMENT_SEC) * SR)
                txt = live.partial_result(np.zeros(n, dtype=np.float32)).text
                if txt != prev:
                    changes += 1
                    prev = txt
                if not txt.strip():
                    empties += 1
                    continue
                dw = [norm_word(x) for x in txt.split()]
                rw = ref_norm_in(w["t_start"], w["t_start"] + WIN_SEC,
                                 res["ref_chunks"])
                if not rw:
                    continue
                m = SequenceMatcher(None, dw, rw, autojunk=False)
                matched = sum(b.size for b in m.get_matching_blocks())
                precs.append(matched / max(1, len(dw)))
                recs.append(matched / max(1, len(rw)))
        out_all[label] = {
            "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4),
            "changes": changes,
            "empty_steps": empties,
        }
        print(f"[{label}] precision={np.mean(precs):.3f} "
              f"recall={np.mean(recs):.3f} changes={changes} empty={empties}")

    out_path = ROOT / "spikes" / "speech_rate" / "report_live_regression.json"
    out_path.write_text(json.dumps(out_all, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[done] → {out_path}")


if __name__ == "__main__":
    main()
