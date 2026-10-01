"""آزمایش ۷ — رگرسیون رفتاری: LiveTranscriber جدید (سه قانون) روی ۳۵۰ کلیپ.

بازپخش فرضیه‌های کش‌شده از windows_multi.pkl از طریق همان partial_result
واقعی (بدون شبیه‌سازی دستی) و مقایسه با نسخه‌ی بدون قوانین (قوانین خاموش).
سنجه: دقتِ واژه‌های نمایش‌داده‌شده در برابر متن مرجع manifest + پایداری.
خروجی: spikes/speech_rate/report_rules_regression.json
"""
from __future__ import annotations

import json
import pickle
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.asr import AsrHypothesis, AsrWord, LiveTranscriber  # noqa: E402

CACHE = ROOT / "spikes" / "speech_rate" / "windows_multi.pkl"
SR = 16000
WIN_SEC = 10.0
STEP_SEC = 0.8
MS_FRAME = 80.0 / 1000.0


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


class ReplayEngine:
    """موتوری که فرضیه‌های کش‌شده را به ترتیب می‌دهد.

    مثل دیکدر واقعی: واژه‌ها با span فریم نسبت به شروع پنجره.
    پنجره‌ی k از بافرِ دیکته‌ی پیوسته بریده می‌شود؛ فرضیه‌ی آن از کش
    می‌آید. windows کش به ترتیب زمانی کلیپ‌های چسبیده‌اند (offset اعمال
    شده در متن قبلی).
    """

    def __init__(self, windows, offset=0.0):
        self.windows = windows
        self.calls = 0
        self.offset = offset

    def transcribe_with_details(self, samples, sample_rate=16000, previous_text=None):
        w = self.windows[min(self.calls, len(self.windows) - 1)]
        self.calls += 1
        words = tuple(
            AsrWord(x["text"], x["conf"], x["margin"], start=x["f0"], end=x["f1"])
            for x in w["words"]
        )
        confs = [x["conf"] for x in w["words"]] or [0.0]
        return AsrHypothesis(
            text=w["text"],
            confidence=float(np.mean(confs)),
            margin=0.0,
            words=words,
        )

    def transcribe(self, samples, sample_rate=16000):
        return self.transcribe_with_details(samples).text


def replay_session(clip_list, rules_on: bool):
    """پخش کلیپ‌های یک منبع به‌صورت دیکته‌ی پیوسته — مثل برنامه‌ی واقعی.

    بافر صوتی مفهومی است (رشد کلیپ‌ها)؛ هر پنجره‌ی کش‌شده معادل
    rec.get_tail_16k در لحظه‌ی خودش. واژه‌های هر پنجره با span نسبت به
    شروع همان پنجره‌اند و پنجره‌ها در خط زمانی بافر جابه‌جا می‌شوند —
    دقیقاً همان چیزی که LiveTranscriber انتظار دارد.
    """
    windows = []
    base = 0.0
    ref_with_times = []  # (t0, t1, [norm_words])
    for c in clip_list:
        for w in c["windows"]:
            windows.append({**w, "t_start": w["t_start"] + base})
        ref_with_times.append((
            base, base + c["duration"],
            [norm_word(x) for x in c["ref"].split()],
        ))
        base += c["duration"]
    # پنجره‌ها اکنون روی خط زمانی پیوسته‌اند؛ هر 0.8s یکی — مثل PARTIAL_INTERVAL
    live = LiveTranscriber(
        ReplayEngine(windows), window_sec=WIN_SEC, stable_live=True,
    )
    if not rules_on:
        live.max_contest = 1.01
        live.lock_edge_sec = 0.0
        live.vote_high_conf = 0.0
    states = []
    for wi in range(len(windows)):
        buf_end = windows[wi]["t_start"] + WIN_SEC
        n = int(buf_end * SR)
        states.append((windows[wi]["t_start"],
                       live.partial_result(np.zeros(n, dtype=np.float32)).text))
    return states, ref_with_times


def replay_clip(clip, rules_on: bool):
    """بازپخش یک کلیپ جدا — پنجره‌ها همان کش‌شده‌ها با بافر رشدی.

    هر کلیپ یک «دیکته‌ی کوتاه» است؛ سنجه: نمایش در برابر مرجعِ همان
    پنجره (کل کلیپ داخل پنجره‌ی 10s جای می‌گیرد).
    """
    live = LiveTranscriber(
        ReplayEngine(clip["windows"]),
        window_sec=WIN_SEC, stable_live=True,
    )
    if not rules_on:
        # قوانین خاموش: مثل نسخه‌ی قبل از این تغییر
        live.max_contest = 1.01        # هر رقیبی مجاز
        live.lock_edge_sec = 0.0       # قانون لبه خاموش
        live.vote_high_conf = 0.0       # قانون رأی همیشه پاس (conf>=0)
    states = []
    for wi in range(len(clip["windows"])):
        # بافر رشدی — پنجره = کل بافر تا این لحظه (کلیپ < 10s)
        buf_end = clip["windows"][wi]["t_start"] + WIN_SEC
        n = int(min(buf_end, clip["duration"]) * SR)
        states.append((clip["windows"][wi]["t_start"],
                       live.partial_result(np.zeros(n, dtype=np.float32)).text))
    return states


def words_shown_vs_ref(states, ref_with_times):
    """دقت/یادآوری در برابر مرجعِ هم‌بازه‌ی زمانی هر پنجره."""
    prec_rows, rec_rows = [], []
    for t_start, txt in states:
        t_end = t_start + WIN_SEC
        ref = []
        for r0, r1, ws in ref_with_times:
            if r1 <= t_start or r0 >= t_end or not ws:
                continue
            for i, w in enumerate(ws):
                wt = r0 + (i + 0.5) / len(ws) * (r1 - r0)
                if t_start <= wt < t_end:
                    ref.append(w)
        if not txt.strip():
            if ref:
                rec_rows.append(0.0)
            continue
        dw = [norm_word(w) for w in txt.split()]
        m = SequenceMatcher(None, dw, ref, autojunk=False)
        matched = sum(b.size for b in m.get_matching_blocks())
        prec_rows.append(matched / max(1, len(dw)))
        rec_rows.append(matched / max(1, len(ref)))
    return prec_rows, rec_rows


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]
    # گروه‌بندی بر منبع → هر منبع یک «دیکته‌ی پیوسته»
    by_src = {}
    for c in clips:
        by_src.setdefault(c["id"].rsplit("-", 1)[0], []).append(c)
    print(f"[info] {len(clips)} clips in {len(by_src)} sessions")

    for rules_on in (False, True):
        label = "قوانین روشن" if rules_on else "قوانین خاموش (پایه)"
        precs, recs = [], []
        n_changes = 0
        empty_shown = 0
        for src, clip_list in sorted(by_src.items()):
            states, ref_with_times = replay_session(clip_list, rules_on)
            p, r = words_shown_vs_ref(states, ref_with_times)
            precs.extend(p)
            recs.extend(r)
            prev = None
            for _, t in states:
                if t != prev:
                    n_changes += 1
                    prev = t
                if not t.strip():
                    empty_shown += 1
        print(f"[{label}] precision={np.mean(precs):.3f} recall={np.mean(recs):.3f} "
              f"changes={n_changes} empty_steps={empty_shown}")
        if rules_on:
            out_on = {
                "rules_on": rules_on,
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "changes": n_changes,
                "empty_steps": empty_shown,
            }
        else:
            out_off = {
                "rules_on": rules_on,
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "changes": n_changes,
                "empty_steps": empty_shown,
            }

    out = {"baseline_rules_off": out_off, "new_rules_on": out_on}
    out_path = ROOT / "spikes" / "speech_rate" / "report_rules_regression.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[done] → {out_path}")


if __name__ == "__main__":
    main()
