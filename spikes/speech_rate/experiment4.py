"""آزمایش ۴ — شبیه‌سازی سرتاسری: قاعده‌ی فعلی قفل در برابر رأی‌گیری واژه.

بازپخش تصمیم‌های LiveTranscriber روی ۱۷۹ پنجره‌ی کش‌شده (بدون inference):

A) وضع فعلی: پذیرش با _accepts + سوپاپ hysteresis (تکرار ۲ بار)
B) رأی‌گیری واژه: هر واژه‌ی *جدید* در نمایش قفل‌شده باید در ≥1 پنجره‌ی
   قبلی هم‌زمان دیده شده باشد، یا conf≥0.90. سوپاپ hysteresis کور
   خاموش است (جایش همین رأی است).
C) سخت‌گیرتر: ≥2 پنجره‌ی قبلی یا conf≥0.95.

سنجه‌ها در هر گام: دقت/یادآوری/F1 نمایش در برابر مرجعِ همان بازه‌ی 10s
(نرمال‌شده)، تعداد و نوع تغییرهای نمایش، و منحنی رشد متن زنده.
خروجی: spikes/speech_rate/report4.json
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.asr import AsrHypothesis, AsrWord, DirectCtcAsrEngine, LiveTranscriber  # noqa: E402

CACHE = ROOT / "spikes" / "speech_rate" / "windows.pkl"
WAV = ROOT / "spikes" / "speech_rate" / "sample_16k.wav"
SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
MS_PER_MODEL_FRAME = 80.0 / 1000.0
CLUSTER_TOL = 0.45


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def replay(wins, rule, pv_of):
    """بازپخط partial_result با فرضیه‌های کش‌شده.

    rule:
      'A' وضع فعلی — _accepts + سوپاپ hysteresis کور
      'B' رأی‌گیری — دروازه‌ی دنباله با رأی واژه *جایگزین* conf≥0.25:
          هر واژه‌ی تازه: votes≥1 (دیده‌شده در پنجره‌ی قبلی) یا conf≥0.90
          سوپاپ hysteresis هم مشروط به عبور رأی همان واژه‌هاست
      'C' سخت‌گیر — votes≥2 یا conf≥0.95
    """
    live = LiveTranscriber(
        engine=None, window_sec=WIN_SEC, stable_live=True,
        margin=0.02, hysteresis=2, min_word_confidence=0.25,
    )
    states = []

    def tail_ok(wi, result, j1, j2, need, hi):
        for j in range(j1, j2):
            word = result.words[j] if j < len(result.words) else None
            if word is None:
                continue
            if pv_of(wi, word.text, j) < need and word.confidence < hi:
                return False
        return True

    for wi, w in enumerate(wins):
        result = AsrHypothesis(
            text=w["text"],
            confidence=float(np.mean([x["conf"] for x in w["words"]]))
            if w["words"] else 0.0,
            margin=float(np.mean([x["margin"] for x in w["words"]]))
            if w["words"] else 0.0,
            words=tuple(AsrWord(x["text"], x["conf"], x["margin"])
                        for x in w["words"]),
        )
        current = live._displayed
        if current is not None and result.text == current.text:
            live._candidate = None
            live._candidate_count = 0
            states.append((w["t_start"], current.text))
            continue

        if live._candidate is not None and live._candidate.text == result.text:
            live._candidate_count += 1
        else:
            live._candidate = result
            live._candidate_count = 1

        if rule == "A" or current is None:
            accepted = current is None or live._accepts(result, current)
            if rule == "A" and not accepted and (
                    live._candidate_count >= live.hysteresis
                    and result.confidence > 0.0):
                accepted = True
        else:
            need = 1 if rule == "B" else 2
            hi = 0.90 if rule == "B" else 0.95
            cur_n = [norm_word(x) for x in current.text.split()]
            new_n = [norm_word(x) for x in result.text.split()]
            accepted = False
            # مسیر ادامه: پیشوند یکسان، دنباله با رأی
            if len(new_n) >= len(cur_n) and new_n[:len(cur_n)] == cur_n:
                accepted = (len(new_n) == len(cur_n)
                            or tail_ok(wi, result, len(cur_n), len(new_n),
                                       need, hi))
            else:
                # مسیر جابه‌جایی پنجره (_shift_prefix_len) با رأی روی دنباله
                shift = live._shift_prefix_len(
                    live._words(current), live._words(result))
                if shift is not None:
                    overlap = len(cur_n) - shift
                    accepted = tail_ok(wi, result, overlap, len(new_n),
                                       need, hi)
                else:
                    accepted = live._accepts(result, current)
            # سوپاپ تکرار: فقط اگر واژه‌های تازه رأی دارند
            if not accepted and live._candidate_count >= live.hysteresis:
                m = SequenceMatcher(None, cur_n, new_n, autojunk=False)
                new_idx = set()
                for tag, _, _, j1, j2 in m.get_opcodes():
                    if tag in ("insert", "replace"):
                        new_idx.update(range(j1, j2))
                if not new_idx or tail_ok(wi, result, min(new_idx),
                                          max(new_idx) + 1, need, hi):
                    accepted = True

        if accepted:
            live._displayed = result
            live._candidate_count = 0
            live._candidate = None
            states.append((w["t_start"], result.text))
        else:
            states.append((w["t_start"], current.text if current else ""))
    return states


def kinds_of(states):
    prev, prev_n = None, None
    kinds = defaultdict(int)
    for _, txt in states:
        if txt != prev:
            n = len(txt.split())
            if prev is not None:
                if n > prev_n:
                    kinds["grow"] += 1
                elif n < prev_n:
                    kinds["shrink"] += 1
                else:
                    kinds["edit"] += 1
            prev, prev_n = txt, n
    return dict(kinds)


def main():
    samples = load_wav(WAV)
    wins = pickle.loads(CACHE.read_bytes())
    print(f"[info] {len(wins)} windows")

    # ---- زمان‌های مطلق واژه‌ها ----
    for w in wins:
        ts = w["t_start"]
        for x in w["words"]:
            f0 = (x["t0"] - ts) / 0.01
            x["f1"] = f0 + x["frames"]
            x["at0"] = ts + f0 * MS_PER_MODEL_FRAME
            x["at1"] = ts + max(x["f1"], f0 + 1) * MS_PER_MODEL_FRAME

    # ---- مرجع chunkهای 16s ----
    engine = DirectCtcAsrEngine(ROOT / "model", num_threads=4, beam_width=2)
    chunk_sp = int(16.0 * SR)
    ref_chunks = []
    for cs in range(0, samples.size, chunk_sp):
        ce = min(cs + chunk_sp, samples.size)
        ref_chunks.append((cs / SR, ce / SR, engine.transcribe(samples[cs:ce]).split()))
    print(f"[info] reference: {len(ref_chunks)} chunks")

    def ref_norm_in(t0, t1):
        out = []
        for cs, ce, ws in ref_chunks:
            if ce <= t0 or cs >= t1 or not ws:
                continue
            for i, wd in enumerate(ws):
                wt = cs + (i + 0.5) / len(ws) * (ce - cs)
                if t0 - 0.4 <= wt <= t1 + 0.4:
                    out.append(norm_word(wd))
        return out

    # ---- pv_of: رأی پنجره‌های قبلی برای (win, text, j) ----
    # خوشه‌های هم‌متنِ هم‌زمان → لیست پنجره‌ها
    by_text = defaultdict(list)
    for wi, w in enumerate(wins):
        for x in w["words"]:
            by_text[x["text"]].append((x["at0"], x["at1"], wi))
    clusters = defaultdict(list)  # text → [(t0, t1, [wis])]
    for text, evs in by_text.items():
        evs.sort()
        for t0, t1, wi in evs:
            if clusters[text] and t0 - clusters[text][-1][1] <= CLUSTER_TOL:
                c = clusters[text][-1]
                c[1] = max(c[1], t1)
                c[2].append(wi)
            else:
                clusters[text].append([t0, t1, [wi]])

    def pv_of(wi, text, j):
        """پنجره‌های قبلی که همین واژه را در همین جای متن دیده‌اند.

        j = جایگاه واژه در متن پنجره؛ برای هم‌ارزی، بازه‌ی زمانی واژه‌ی
        j از داده‌ی پنجره خوانده می‌شود.
        """
        w = wins[wi]
        # پیدا کردن بازه‌ی زمانی این occurrence (واژه‌ی j از متن)
        # متن پنجره ممکن است با words یکی باشد؛ جایگاه j → occurrence
        occ = 0
        target = None
        for x in w["words"]:
            if x["text"] == text:
                if occ == count_before(w, text, j):
                    target = x
                    break
                occ += 1
        if target is None:
            # جایگاه دقیق نبود؛ اولین occurrence
            for x in w["words"]:
                if x["text"] == text:
                    target = x
                    break
        if target is None:
            return 0
        for t0, t1, wis in clusters.get(text, ()):
            ov = min(target["at1"], t1) - max(target["at0"], t0)
            if ov > 0.3 * min(target["at1"] - target["at0"], t1 - t0):
                return sum(1 for x in wis if x < wi)
        return 0

    def count_before(w, text, j):
        """تعداد واژه‌های text قبل از جایگاه j در متن پنجره."""
        words = w["text"].split()
        n = 0
        for i, t in enumerate(words[:j]):
            if t == text:
                n += 1
        return n

    # ---- بازپخط سه واریانت ----
    # مرجع انباشتی: همه‌ی واژه‌های مرجع با زمان ≤ انتهای پنجره — چون نمایشِ
    # قفل‌شده واژه‌های قدیمی‌تر از پنجره را هم نگه می‌دارد (رفتار دیکته).
    def ref_norm_cumulative(t_end):
        out = []
        for cs, ce, ws in ref_chunks:
            if cs >= t_end or not ws:
                continue
            for i, wd in enumerate(ws):
                wt = cs + (i + 0.5) / len(ws) * (ce - cs)
                if wt <= t_end + 0.4:
                    out.append(norm_word(wd))
        return out

    results = {}
    for rule in ("A", "B", "C"):
        states = replay(wins, rule, pv_of)
        precs, recs, f1s = [], [], []
        n_words_curve = []
        for t, txt in states:
            n_words_curve.append(len(txt.split()))
            if t < 12:
                continue
            dw = [norm_word(x) for x in txt.split()]
            rw = ref_norm_cumulative(t + WIN_SEC)
            if not dw and not rw:
                continue
            m = SequenceMatcher(None, dw, rw, autojunk=False)
            matched = sum(b.size for b in m.get_matching_blocks())
            precs.append(matched / max(1, len(dw)))
            recs.append(matched / max(1, len(rw)))
            f1s.append(2 * matched / max(1, len(dw) + len(rw)))
        k = kinds_of(states)
        results[rule] = {
            "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4),
            "f1": round(float(np.mean(f1s)), 4),
            "kinds": k,
            "n_changes": sum(k.values()),
            "mean_words_shown": round(float(np.mean(n_words_curve)), 2),
        }
        print(f"[rule {rule}] P={results[rule]['precision']:.3f} "
              f"R={results[rule]['recall']:.3f} F1={results[rule]['f1']:.3f} "
              f"changes={results[rule]['n_changes']} {k} "
              f"mean_words={results[rule]['mean_words_shown']:.1f}")

    # نوسان خام برای مقایسه
    raw_changes = 0
    prev = None
    for w in wins:
        if w["text"] != prev:
            raw_changes += 1
            prev = w["text"]
    print(f"[raw] بدون قفل: {raw_changes} تغییر")

    out = {"variants": results, "raw_changes": raw_changes}
    out_path = Path(__file__).parent / "report4.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[done] report → {out_path}")


if __name__ == "__main__":
    main()
