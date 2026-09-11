"""آزمایش ۶ — تحلیل سیگنال‌ها روی ۳۵۰ کلیپ (۷ منبع) با مرجع انسانی.

برچسب: هر واژه‌ی هر پنجره با متن manifestِ همان کلیپ هم‌تراز می‌شود
(SequenceMatcher با نرمال‌سازی نیم‌فاصله/هم‌آوا/نشانه)؛ واژه‌ای که در
مرجع نماند «بد» است.

سیگنال‌ها (همان آزمایش ۳/۴):
  conf, margin, votes (رأی پنجره‌های قبلی هم‌زمان), contest (رقیب),
  edge (لبه‌ی قدیمی/زنده), dur_ms, wpm_active محلی, wpm_raw
به‌علاوه: تفکیک خطا per-source و per-جایگاه، و شبیه‌سازی قواعد قفل.
خروجی: spikes/speech_rate/report_multi.json
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "spikes" / "speech_rate" / "windows_multi.pkl"
MS_PER_MODEL_FRAME = 80.0 / 1000.0
WIN_SEC = 10.0
STEP_SEC = 0.8
SILENCE_RMS = 0.003
CLUSTER_TOL = 0.45


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(pos, neg) -> float:
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


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]
    print(f"[info] {len(clips)} clips, "
          f"{sum(len(c['windows']) for c in clips)} windows")

    # ---- زمان‌های مطلق + ref نرمال ----
    for c in clips:
        c["ref_norm"] = [norm_word(w) for w in c["ref"].split()]
        for w in c["windows"]:
            ts = w["t_start"]
            for x in w["words"]:
                x["at0"] = ts + x["f0"] * MS_PER_MODEL_FRAME
                x["at1"] = ts + x["f1"] * MS_PER_MODEL_FRAME

    # ---- خوشه‌های هم‌متن هم‌زمان در هر کلیپ (برای votes/contest) ----
    for c in clips:
        by_text = defaultdict(list)
        for wi, w in enumerate(c["windows"]):
            for x in w["words"]:
                by_text[x["text"]].append((x["at0"], x["at1"], wi))
        clusters = defaultdict(list)
        for text, evs in by_text.items():
            evs.sort()
            for t0, t1, wi in evs:
                if clusters[text] and t0 - clusters[text][-1][1] <= CLUSTER_TOL:
                    cc = clusters[text][-1]
                    cc[1] = max(cc[1], t1)
                    cc[2].append(wi)
                else:
                    clusters[text].append([t0, t1, [wi]])
        c["clusters"] = clusters

    def pv_of(c, wi, x):
        """تعداد پنجره‌های قبلی با همین واژه در همین خوشه."""
        for t0, t1, wis in c["clusters"].get(x["text"], ()):
            ov = min(x["at1"], t1) - max(x["at0"], t0)
            if ov > 0.3 * min(x["at1"] - x["at0"], t1 - t0):
                return sum(1 for wj in wis if wj < wi)
        return 0

    def contest_of(c, wi, x):
        """سهم رأی رقیب‌های هم‌زمان با متن متفاوت."""
        own = 0
        rival = 0
        for t0, t1, wis in c["clusters"].get(x["text"], ()):
            ov = min(x["at1"], t1) - max(x["at0"], t0)
            if ov > 0.3 * min(x["at1"] - x["at0"], t1 - t0):
                own += len(wis)
                break
        for text, cls in c["clusters"].items():
            if text == x["text"]:
                continue
            for t0, t1, wis in cls:
                ov = min(x["at1"], t1) - max(x["at0"], t0)
                if ov > 0.3 * min(x["at1"] - x["at0"], t1 - t0):
                    rival += len(wis)
        return rival / max(1, own + rival)

    # ---- برچسب‌گذاری + سیگنال‌ها ----
    rows = []
    per_src = defaultdict(lambda: [0, 0])       # source → [bad, good]
    per_pos = {"leading_edge": [0, 0], "trailing_edge": [0, 0], "middle": [0, 0]}
    for c in clips:
        src = c["id"].rsplit("-", 1)[0] if "-" in c["id"] else c["id"]
        # مرجع word-timing: توزیع یکنواخت در بازه‌ی کلیپ (ref فقط یک متن است)
        dur = c["duration"]
        ref_w = c["ref_norm"]
        n_ref = max(1, len(ref_w))
        for wi, w in enumerate(c["windows"]):
            ts = w["t_start"]
            win_words = w["words"]
            beam_norm = [norm_word(x["text"]) for x in win_words]
            # واژه‌های مرجعِ داخل بازه‌ی پنجره
            ref_in_win = []
            for i, rw in enumerate(ref_w):
                wt = (i + 0.5) / n_ref * dur
                if ts - 0.4 <= wt <= ts + WIN_SEC + 0.4:
                    ref_in_win.append(rw)
            m = SequenceMatcher(None, beam_norm, ref_in_win, autojunk=False)
            matched = set()
            for tag, i1, i2, _, _ in m.get_opcodes():
                if tag == "equal":
                    matched.update(range(i1, i2))
            # گفتار فعال تا این لحظه (برای wpm محلی) — از RMS نیست؛
            # تقریب: پیش‌رفت زمانی×ضریب فعالیت پنجره
            active_ratio = 1.0  # ساده: wpm محلی از زمان سپری‌شده
            cum = 0
            for idx, x in enumerate(win_words):
                good = idx in matched
                pv = pv_of(c, wi, x)
                contest = contest_of(c, wi, x)
                d_start = x["at0"] - ts          # فاصله از لبه‌ی قدیمی
                d_end = (ts + min(WIN_SEC, c["duration"])) - x["at1"]
                # جایگاه: لبه‌ی قدیمی < 1s / لبه‌ی زنده < 1.5s / میانه
                pos = ("leading_edge" if d_start < 1.0
                       else "trailing_edge" if d_end < 1.5 else "middle")
                # wpm محلی: واژه‌های قبلی + این واژه بر دقیقه‌ی گفتار فعال
                active_until = max(0.1, x["at0"] - ts) * active_ratio
                wpm_local = (cum + 1) / (active_until / 60.0) if active_until > 0.3 else 0.0
                rows.append({
                    "clip": c["id"], "src": src, "win": wi,
                    "text": x["text"], "conf": x["conf"], "margin": x["margin"],
                    "dur_ms": round((x["at1"] - x["at0"]) * 1000, 1),
                    "d_start": round(d_start, 3), "d_end": round(d_end, 3),
                    "pos": pos, "votes": pv, "contest": round(contest, 3),
                    "wpm_local": round(wpm_local, 1),
                    "good": good,
                })
                per_src[src][0 if not good else 1] += 1
                per_pos[pos][0 if not good else 1] += 1
                cum += 1

    bad = [r for r in rows if not r["good"]]
    good = [r for r in rows if r["good"]]
    print(f"[info] {len(rows)} word-obs: bad={len(bad)} "
          f"({100*len(bad)/max(1,len(rows)):.1f}%), good={len(good)}")

    # ---- AUC ----
    def sig(name, f):
        return auc([f(r) for r in bad], [f(r) for r in good])

    signals = {
        "conf_inv": sig("c", lambda r: 1.0 - r["conf"]),
        "margin_inv": sig("m", lambda r: 1.0 - r["margin"]),
        "votes_inv (ناپایداری)": sig("v", lambda r: -float(r["votes"])),
        "contest": sig("x", lambda r: r["contest"]),
        "d_start_inv (لبه‌ی قدیمی)": sig("d", lambda r: -r["d_start"]),
        "d_end_inv (لبه‌ی زنده)": sig("e", lambda r: -r["d_end"]),
        "dur_inv": sig("u", lambda r: -r["dur_ms"] / 1000),
        "wpm_local": sig("w", lambda r: r["wpm_local"]),
    }
    combos = {
        "conf+votes": lambda r: (1 - r["conf"]) - 0.05 * r["votes"],
        "conf+contest": lambda r: (1 - r["conf"]) + r["contest"],
        "conf+votes+contest": lambda r: (1 - r["conf"]) - 0.05 * r["votes"] + r["contest"],
    }
    print("\n[AUC] پیش‌بینی واژه‌ی بد (1=کامل، 0.5=بی‌سیگنال):")
    for k, v in sorted(signals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:.3f}")
    print("[AUC] ترکیب‌ها:")
    combo_auc = {}
    for k, f in combos.items():
        combo_auc[k] = auc([f(r) for r in bad], [f(r) for r in good])
        print(f"  {k:28s} {combo_auc[k]:.3f}")

    # ---- تفکیک per-source و per-position ----
    print("\n[خطا به تفکیک منبع]:")
    src_err = {}
    for src, (b, g) in sorted(per_src.items()):
        src_err[src] = round(100 * b / max(1, b + g), 1)
        print(f"  {src:24s} {src_err[src]:5.1f}%  (n={b+g})")
    print("[خطا به تفکیک جایگاه]:")
    pos_err = {}
    for pos, (b, g) in per_pos.items():
        pos_err[pos] = round(100 * b / max(1, b + g), 1)
        print(f"  {pos:16s} {pos_err[pos]:5.1f}%  (n={b+g})")

    # ---- ایده‌ی سرعت گفتار: سطح کلیپ و سطح واژه ----
    # WPM کلیپ از ref: تعداد واژه / دقیقه‌ی گفتار فعال — تقریبی بدون VAD،
    # از میانگین سکوت پنجره‌ها محاسبه نمی‌شود؛ همین همبستگی سطح واژه را سنجیدیم
    wl = np.array([r["wpm_local"] for r in rows])
    gd = np.array([0 if r["good"] else 1 for r in rows])
    cf = np.array([r["conf"] for r in rows])
    e1 = {
        "corr_wpm_err": pearson(wl, gd),
        "corr_wpm_conf": pearson(wl, cf),
        "auc_wpm": signals["wpm_local"],
    }
    print(f"\n[E1 سرعت گفتار] corr(wpm, err)={e1['corr_wpm_err']:+.3f}  "
          f"corr(wpm, conf)={e1['corr_wpm_conf']:+.3f}  AUC={e1['auc_wpm']:.3f}")

    # ---- جدول قواعد قفل ----
    print("\n[قواعد قفل] نرخ خطا در واژه‌های قفل‌شده:")
    rules = {
        "فعلی: conf≥0.25": lambda r: r["conf"] >= 0.25,
        "votes≥2": lambda r: r["votes"] >= 1,  # votes=رأی پنجره‌های قبلی
        "votes≥3": lambda r: r["votes"] >= 2,
        "votes≥2 و conf≥0.25": lambda r: r["votes"] >= 1 and r["conf"] >= 0.25,
        "contest<0.3 و conf≥0.25": lambda r: r["contest"] < 0.3 and r["conf"] >= 0.25,
        "conf≥0.25 و d_start≥1s": lambda r: r["conf"] >= 0.25 and r["d_start"] >= 1.0,
    }
    rule_table = []
    for name, f in rules.items():
        locked = [r for r in rows if f(r)]
        n_bad = sum(1 for r in locked if not r["good"])
        pct = 100 * n_bad / max(1, len(locked))
        cov = 100 * len(locked) / max(1, len(rows))
        rule_table.append({"rule": name, "locked": len(locked),
                           "bad": n_bad, "bad_pct": round(pct, 1),
                           "coverage": round(cov, 1)})
        print(f"  {name:30s} خطا {n_bad:>5}/{len(locked):<5} ({pct:5.1f}%)  پوشش {cov:5.1f}%")
    base = 100 * len(bad) / max(1, len(rows))
    print(f"  (پایه: {base:.1f}%)")

    out = {
        "n_clips": len(clips),
        "word_observations": len(rows),
        "bad_pct": round(base, 1),
        "auc_signals": signals, "auc_combos": combo_auc,
        "per_source_err": src_err, "per_position_err": pos_err,
        "e1_speech_rate": e1, "rule_table": rule_table,
    }
    out_path = Path(__file__).parent / "report_multi.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
