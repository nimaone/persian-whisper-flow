"""آزمایش ۳ (نسخه‌ی نهایی) — کدام سیگنال‌ها «واژه‌ی زنده‌ی بد» را پیش‌بینی می‌کنند؟

برچسب مرجع: هر واژه‌ی پنجره‌ی زنده با ترنسکرایپ نهاییِ همان بازه
(chunkهای 16s، همان مسیر final برنامه) هم‌تراز می‌شود؛ واژه‌ای که در
ترنسکرایپ نهایی نماند «بد» است — دقیقاً همان چیزی که قفل پیشوند باید
از نمایشِ پایدارش منع کند.

سیگنال‌ها (همه با جهت «بالا = بدتر»):
  conf_inv        = 1 - conf        (اطمینان فعلی برنامه)
  margin_inv      = 1 - margin
  votes           = چند پنجره همان واژه را در همان زمان تکرار کرده (پایداری)
  contest         = رأی رقیب‌های هم‌زمان / کل
  edge_s          = فاصله تا لبه‌ی پنجره (نزدیک لبه = برش واژه)
  dur_inv         = 1 - dur_ms/1000 (واژه‌های خیلی کوتاه)
  wpm_local       = سرعت گفتار فعال تا این واژه (ایده‌ی کاربر)
  no_consensus    = واژه در greedy decode همان پنجره نیست (اجماع دو دیکدر)

مقایسه با AUC + جدول قواعد قفل عملی.
خروجی: spikes/speech_rate/report3.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.asr import _CtcHypothesisDecoder, _softmax  # noqa: E402
from app.hotword_asr import kaldi_fbank, load_labels  # noqa: E402

WAV = ROOT / "spikes" / "speech_rate" / "sample_16k.wav"
CACHE = ROOT / "spikes" / "speech_rate" / "windows.pkl"
SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
MS_PER_MODEL_FRAME = 80.0 / 1000.0  # تأیید شده: 198 فریم fbank → 26 فریم مدل
SILENCE_RMS = 0.003
CLUSTER_TOL = 0.45  # تلورانس جابه‌جایی زمانی برای هم‌ارزی دو event


def load_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def norm_word(w: str) -> str:
    """نرمال‌سازی برای مقایسه: نیم‌فاصله، هم‌آوای عربی، نشانه‌گذاری."""
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    w = "".join(ch for ch in w if ch.isalnum())
    return w


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(pos, neg) -> float:
    """AUC = P(score_pos > score_neg)؛ pos = کلاس «بد»."""
    p, n = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    order = np.concatenate([p, n])
    sorter = np.argsort(order, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[sorter] = np.arange(1, order.size + 1)
    s = order[sorter]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[sorter[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[:p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2) / (p.size * n.size))


def greedy_words(log_probs: np.ndarray, labels: list[str], blank_id: int) -> list[str]:
    """دیکد greedy روی همان logits (تقریباً رایگان)."""
    ids = log_probs.argmax(axis=-1)
    out = []
    prev = blank_id
    for i in ids:
        if i != prev and i != blank_id:
            out.append(labels[i])
        prev = i
    text = "".join(out).replace("▁", " ")
    return [w for w in text.split() if w]


def main():
    samples = load_wav(WAV)
    duration = samples.size / SR
    wins = pickle.loads(CACHE.read_bytes())
    print(f"[info] audio: {duration:.1f}s, {len(wins)} cached windows")

    # ---- بازسازی فریم شروع/پایان از pickle (t0 = t_start + f0*10ms) ----
    for w in wins:
        ts = w["t_start"]
        for x in w["words"]:
            f0 = (x["t0"] - ts) / 0.01
            x["f0"], x["f1"] = f0, f0 + x["frames"]

    # ---- مرجع: chunkهای 16s با زمان‌بندی یکنواخت واژه‌ها ----
    dec = _CtcHypothesisDecoder(ROOT / "model", num_threads=4, beam_width=2)
    chunk_sp = int(16.0 * SR)
    ref_chunks = []  # (cs, ce, [words_norm])
    for cs in range(0, samples.size, chunk_sp):
        ce = min(cs + chunk_sp, samples.size)
        txt = dec.decode(samples[cs:ce]).text
        ref_chunks.append((cs / SR, ce / SR,
                           [norm_word(w) for w in txt.split()]))
    dec.release()
    print(f"[info] reference chunks: {len(ref_chunks)}")

    def ref_words_in(t0: float, t1: float) -> list[str]:
        out = []
        for cs, ce, ws in ref_chunks:
            if ce <= t0 or cs >= t1 or not ws:
                continue
            for i, w in enumerate(ws):
                wt = cs + (i + 0.5) / len(ws) * (ce - cs)
                if t0 - 0.4 <= wt <= t1 + 0.4:
                    out.append(w)
        return out

    # ---- instances: هم‌ارزی واژه‌ها بین پنجره‌ها (هم‌متن + هم‌زمان) ----
    events = []  # (win_idx, text, t0, t1)
    for wi, w in enumerate(wins):
        ts = w["t_start"]
        for x in w["words"]:
            t0 = ts + x["f0"] * MS_PER_MODEL_FRAME
            t1 = ts + max(x["f1"], x["f0"] + 1) * MS_PER_MODEL_FRAME
            events.append((wi, x["text"], t0, t1))

    by_text = defaultdict(list)
    for wi, text, t0, t1 in events:
        by_text[text].append((t0, t1, wi))
    inst_id = {}   # (wi, text, t0) → instance id
    inst_words = []  # per instance: text, t0, t1, votes
    for text, evs in by_text.items():
        evs.sort()
        cur = None
        for t0, t1, wi in evs:
            if cur is None or t0 - cur["t1"] > CLUSTER_TOL:
                cur = {"text": text, "t0": t0, "t1": t1, "wins": [wi]}
                inst_words.append(cur)
            else:
                cur["t1"] = max(cur["t1"], t1)
                cur["wins"].append(wi)
            inst_id[(wi, text, round(t0, 3))] = len(inst_words) - 1
    print(f"[info] word instances: {len(inst_words)} "
          f"(از {len(events)} مشاهده‌ی پنجره)")

    # رقبای هم‌زمان: برای هر instance، رأی instanceهای هم‌پوشان با متن متفاوت
    inst_by_time = sorted(
        range(len(inst_words)),
        key=lambda i: inst_words[i]["t0"])
    contest_votes = [0] * len(inst_words)
    for i, inst in enumerate(inst_words):
        own = len(inst["wins"])
        rival = 0
        for j in inst_by_time:
            other = inst_words[j]
            if j == i or other["text"] == inst["text"]:
                continue
            ov = min(inst["t1"], other["t1"]) - max(inst["t0"], other["t0"])
            if ov > 0.3 * min(inst["t1"] - inst["t0"],
                              other["t1"] - other["t0"]):
                rival += len(other["wins"])
        contest_votes[i] = rival / max(1, own + rival)

    # ---- سیگنال اجماع دو دیکدر: greedy روی همان logits هر پنجره ----
    print("[info] computing greedy consensus per window …")
    labels = load_labels(ROOT / "model" / "tokens.txt")
    blank_id = labels.index("")
    dec2 = _CtcHypothesisDecoder(ROOT / "model", num_threads=4, beam_width=2)
    win_sp = int(WIN_SEC * SR)
    t0 = time.perf_counter()
    greedy_sets = []  # per window: set(norm words)
    active_secs = []
    for w in wins:
        ts = w["t_start"]
        win = samples[int(ts * SR):int(ts * SR) + win_sp]
        feat = kaldi_fbank(win, SR)
        with dec2._lock:
            lp, lengths = dec2._session.run(
                None, {"audio_signal": feat.T[None].astype(np.float32),
                       "length": np.array([feat.shape[0]], dtype=np.int64)})
        lp = lp[0][: int(lengths[0])]
        gw = set(norm_word(x) for x in greedy_words(lp, labels, blank_id))
        greedy_sets.append(gw)
        blk = int(0.1 * SR)
        nb = win.size // blk
        if nb:
            rms = np.sqrt((win[: nb * blk].reshape(nb, blk) ** 2).mean(axis=1))
            active_secs.append(float((rms > SILENCE_RMS).sum() * 0.1))
        else:
            active_secs.append(0.0)
    dec2.release()
    print(f"[info] greedy pass done in {time.perf_counter() - t0:.1f}s")

    # ---- برچسب‌گذاری: واژه‌ی پنجره در ترنسکرایپ نهاییِ همان بازه؟ ----
    rows = []
    for wi, w in enumerate(wins):
        ts = w["t_start"]
        te = ts + WIN_SEC
        active_s = active_secs[wi]
        greedy = greedy_sets[wi]
        # واژه‌های نرمال مرجع در بازه
        refw = ref_words_in(ts, te)
        beam_words = [(x["text"],
                       ts + x["f0"] * MS_PER_MODEL_FRAME,
                       ts + max(x["f1"], x["f0"] + 1) * MS_PER_MODEL_FRAME)
                      for x in w["words"]]
        beam_norm = [norm_word(t) for t, _, _ in beam_words]
        matcher = SequenceMatcher(None, beam_norm, refw, autojunk=False)
        matched = set()
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == "equal":
                matched.update(range(i1, i2))
        cum = 0.0
        for idx, (text, wt0, wt1) in enumerate(beam_words):
            nw = beam_norm[idx]
            ii = inst_id.get((wi, text, round(wt0, 3)))
            if ii is None:  # برچسب‌گذاری اولین eventِ instance
                for cand, inst in enumerate(inst_words):
                    if (inst["text"] == text and wt0 - CLUSTER_TOL
                            <= inst["t1"] and wt1 + CLUSTER_TOL >= inst["t0"]):
                        ii = cand
                        break
            votes = len(inst_words[ii]["wins"]) if ii is not None else 1
            contest = contest_votes[ii] if ii is not None else 0.0
            edge = min(wt0 - ts, te - wt1)
            active_until = max(0.1, wt0 - ts)
            wpm_local = (cum + 1) / (active_until / 60.0) if active_until > 0.3 else 0.0
            rows.append({
                "win": wi, "text": text, "norm": nw,
                "conf": float(w["words"][idx]["conf"]),
                "margin": float(w["words"][idx]["margin"]),
                "dur_ms": round((wt1 - wt0) * 1000, 1),
                "edge_s": round(max(0.0, edge), 3),
                "votes": votes, "contest": round(contest, 3),
                "wpm_local": round(wpm_local, 1),
                "greedy_ok": nw in greedy,
                "good": idx in matched,
                "active_s": round(active_s, 2),
            })
            cum += 1
    n_good = sum(1 for r in rows if r["good"])
    n_bad = len(rows) - n_good
    print(f"[info] labeled {len(rows)} word observations: "
          f"good={n_good}, bad={n_bad} ({100*n_bad/max(1,len(rows)):.1f}%)")

    bad = [r for r in rows if not r["good"]]
    good = [r for r in rows if r["good"]]

    # ---- AUC سیگنال‌ها (جهت: بالا = بدتر) ----
    def sig(key, transform=None):
        f = transform or (lambda r: r[key])
        return auc([f(r) for r in bad], [f(r) for r in good])

    signals = {
        "conf_inv (1-conf)": sig("conf", lambda r: 1.0 - r["conf"]),
        "margin_inv": sig("margin", lambda r: 1.0 - r["margin"]),
        "votes_inv (ناپایداری)": sig("votes", lambda r: -float(r["votes"])),
        "contest (رقیب هم‌زمان)": sig("contest"),
        "edge_inv (نزدیک لبه)": sig("edge_s", lambda r: -r["edge_s"]),
        "dur_inv (واژه‌ی کوتاه)": sig("dur_ms", lambda r: -r["dur_ms"] / 1000),
        "wpm_local (سرعت)": sig("wpm_local"),
        "no_consensus (greedy)": sig("greedy_ok", lambda r: 0.0 if r["greedy_ok"] else 1.0),
    }
    print("\n[AUC] پیش‌بینی واژه‌ی بد (1 = کامل، 0.5 = بی‌سیگنال):")
    for k, v in sorted(signals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:.3f}")

    # ---- ترکیب دوم Signal-ها ----
    combos = {
        "conf + votes": lambda r: (1 - r["conf"]) - 0.05 * r["votes"],
        "conf + contest": lambda r: (1 - r["conf"]) + r["contest"],
        "conf + consensus": lambda r: (1 - r["conf"]) + (0.0 if r["greedy_ok"] else 0.5),
        "conf + votes + contest": lambda r: (1 - r["conf"]) - 0.05 * r["votes"] + r["contest"],
        "همه (بدون wpm)": lambda r: (1 - r["conf"]) - 0.05 * r["votes"] + r["contest"] + (0.0 if r["greedy_ok"] else 0.5) - r["dur_ms"] / 1000,
        "همه + wpm": lambda r: (1 - r["conf"]) - 0.05 * r["votes"] + r["contest"] + (0.0 if r["greedy_ok"] else 0.5) - r["dur_ms"] / 1000 + r["wpm_local"] / 1000,
    }
    print("\n[AUC] ترکیب‌ها:")
    combo_auc = {}
    for k, f in combos.items():
        v = auc([f(r) for r in bad], [f(r) for r in good])
        combo_auc[k] = v
        print(f"  {k:24s} {v:.3f}")

    # ---- جدول قواعد قفل ----
    print("\n[قواعد قفل] خطای میان واژه‌هایی که قفل می‌شوند (پایه: "
          f"{100*n_bad/max(1,len(rows)):.1f}%):")
    rules = {
        "فعلی: conf≥0.25": lambda r: r["conf"] >= 0.25,
        "conf≥0.5": lambda r: r["conf"] >= 0.5,
        "conf≥0.25 و consensus": lambda r: r["conf"] >= 0.25 and r["greedy_ok"],
        "votes≥2 (دو پنجره)": lambda r: r["votes"] >= 2,
        "votes≥3": lambda r: r["votes"] >= 3,
        "votes≥2 و conf≥0.25": lambda r: r["votes"] >= 2 and r["conf"] >= 0.25,
        "votes≥3 و conf≥0.25 و consensus": lambda r: r["votes"] >= 3 and r["conf"] >= 0.25 and r["greedy_ok"],
        "contest<0.3 و conf≥0.25": lambda r: r["contest"] < 0.3 and r["conf"] >= 0.25,
    }
    rule_table = []
    for name, f in rules.items():
        locked = [r for r in rows if f(r)]
        n_l_bad = sum(1 for r in locked if not r["good"])
        pct = 100 * n_l_bad / max(1, len(locked))
        cover = 100 * len(locked) / max(1, len(rows))
        rule_table.append({"rule": name, "locked": len(locked),
                           "bad_locked": n_l_bad, "bad_pct": round(pct, 1),
                           "coverage_pct": round(cover, 1)})
        print(f"  {name:38s} خطا {n_l_bad:>4}/{len(locked):<4} ({pct:5.1f}%)  "
              f"پوشش {cover:5.1f}%")

    # ---- سنجش ایده‌ی کاربر: سرعت گفتار ----
    wl = np.array([r["wpm_local"] for r in rows if r["wpm_local"] > 0])
    gd = np.array([1 if r["good"] else 0 for r in rows if r["wpm_local"] > 0])
    cf = np.array([r["conf"] for r in rows if r["wpm_local"] > 0])
    med = np.median(wl)
    fast = [r for r in rows if r["wpm_local"] > med]
    slow = [r for r in rows if r["wpm_local"] <= med]
    e1 = {
        "corr_wpm_err": pearson(wl, 1 - gd),
        "corr_wpm_conf": pearson(wl, cf),
        "wpm_median": float(med),
        "err_fast_pct": round(100 * sum(1 for r in fast if not r["good"]) / max(1, len(fast)), 1),
        "err_slow_pct": round(100 * sum(1 for r in slow if not r["good"]) / max(1, len(slow)), 1),
    }
    print(f"\n[E1 ایده‌ی سرعت گفتار]")
    print(f"  corr(wpm_local, خطا)  = {e1['corr_wpm_err']:+.3f}")
    print(f"  corr(wpm_local, conf) = {e1['corr_wpm_conf']:+.3f}")
    print(f"  خطا: سریع‌تر از میانه={e1['err_fast_pct']}%  کند‌تر={e1['err_slow_pct']}%")

    out = {
        "audio_duration_s": round(duration, 2),
        "ms_per_model_frame": MS_PER_MODEL_FRAME,
        "word_observations": len(rows), "n_good": n_good, "n_bad": n_bad,
        "auc_signals": signals, "auc_combos": combo_auc,
        "rule_table": rule_table, "e1_speech_rate": e1,
        "rows_sample": rows[:400],
    }
    out_path = Path(__file__).parent / "report3.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] report → {out_path}")


if __name__ == "__main__":
    main()
