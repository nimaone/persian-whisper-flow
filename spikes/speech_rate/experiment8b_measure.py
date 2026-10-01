"""آزمایش ۸ب — سنجه‌ی نهایی رگرسیون روی بافر پیوسته‌ی واقعی.

بازپخش windows_real.pkl (۴۴۹ پنجره × ۷ منبع، بافر پیوسته‌ی واقعی) از
طریق LiveTranscriber واقعی؛ سنجه:
  precision = دقت واژه‌های «متن نمایش» در برابر مرجع انباشتی تا همان لحظه
  recall    = پوشش مرجع انباشتی
  changes   = تغییرهای متن نمایش (نوسان)
  empty     = گام‌های بدون متن
مقایسه: قوانین خاموش (پایه) در برابر روشن (contest/edge/votes + trim).
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

CACHE = ROOT / "spikes" / "speech_rate" / "windows_real.pkl"
SR = 16000
WIN_SEC = 10.0


def norm(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch if ch.isalnum() else "" for ch in w)


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


def ref_cum(ref_chunks, t):
    """مرجع انباشتی تا لحظه‌ی t — با برش جزئی آخرین chunk در جریان."""
    out = []
    for cs, ce, txt in ref_chunks:
        if cs >= t:
            break
        frac = min(1.0, (t - cs) / (ce - cs))
        ws = [norm(x) for x in txt.split()]
        k = int(len(ws) * frac)
        out.extend(ws[:k])
    return out


def main():
    res = pickle.loads(CACHE.read_bytes())
    print(f"[info] {len(res)} sources × {len(res[0]['windows'])} windows")

    out_all = {}
    for rules_on in (False, True):
        label = "قوانین روشن" if rules_on else "قوانین خاموش (پایه)"
        precs, recs, changes, empties = [], [], 0, 0
        for r in res:
            live = LiveTranscriber(
                ReplayEngine(r["windows"]),
                window_sec=WIN_SEC, stable_live=True,
            )
            if not rules_on:
                live.max_contest = 1.01
                live.lock_edge_sec = 0.0
                live.vote_high_conf = 0.0
            prev = None
            for wi, w in enumerate(r["windows"]):
                buf_end = w["t_start"] + WIN_SEC
                n = int(buf_end * SR)
                txt = live.partial_result(
                    np.zeros(n, dtype=np.float32)).text
                if txt != prev:
                    changes += 1
                    prev = txt
                if not txt.strip():
                    empties += 1
                    continue
                dw = [norm(x) for x in txt.split()]
                rw = ref_cum(r["ref_chunks"], buf_end)
                if not rw:
                    continue
                m = SequenceMatcher(None, dw, rw, autojunk=False)
                matched = sum(b.size for b in m.get_matching_blocks())
                precs.append(matched / max(1, len(dw)))
                recs.append(matched / max(1, len(rw)))
        out_all[label] = {
            "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4),
            "f1": round(float(np.mean(
                2 / (1 / max(1e-9, np.mean(precs)) + 1 / max(1e-9, np.mean(recs)))
            )), 4),
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
