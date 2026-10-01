"""آزمایش ۱۲ — سنجش نهایی: بازپیوند «می/نمی» + ثبت صوتی (alias) با هم.

لایه‌ها:
  A) بازپیوند پیشوند: «می/نمی» جداشده از فعل دوباره به هم می‌چسبند
     («می کنم»→«میکنم») — قطعی، بدون نیاز به ثبت
  B) ثبت صوتی (alias): برای هر واژه‌ی ثبت‌شده، ۳ نمونه‌ی تصادفی (شبیه‌سازی
     ۳ ضبط کاربر) → شکل‌های شنیده‌شده → نگاشت. بوت‌استرپ ۵ بار، میانه گزارش
     می‌شود. واژه‌های چندشکلی مثل «ایآی» با تأیید کاربر تا ۳ واریانت می‌گیرند.

سنجه: خطای سمت مرجع (واژه‌های مرجع که در خروجی نیامده‌اند) روی bucket 1.
bucketing با crc32 (پایدار بین اجراها). خروجی: report_final_sim.json
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "spikes" / "speech_rate" / "windows_multi.pkl"

MIN_WORD_LEN = 4
MIN_ERRORS = 5
MIN_TOP_SHARE = 0.4
MAX_VARIANT_COMMON = 2
N_BOOTSTRAP = 5
FORCE_WORDS = ["ایآی"]          # واژه‌های درخواستی — بدون سقف share
PREFIXES = {"می", "نمی"}


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def rejoin(tokens: list[str]) -> list[str]:
    """بازپیوند پیشوند جداشده: «می کنم» → «میکنم»."""
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] in PREFIXES and i + 1 < len(tokens):
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def matched_count(beam: list[str], ref: list[str]) -> int:
    m = SequenceMatcher(None, beam, ref, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks())


def align_pairs(beam: list[str], ref: list[str]):
    """جفت‌های (شکل غلط، واژه‌ی مرجع) از بلوک‌های replace."""
    m = SequenceMatcher(None, beam, ref, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                wrong, right = beam[i1 + k], ref[j1 + k]
                if wrong != right and len(wrong) >= 2 and len(right) >= 2:
                    pairs.append((wrong, right))
    return pairs


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]

    wins = []  # (clip, bucket, beam_tokens, ref_tokens)
    for c in clips:
        ref = [norm_word(w) for w in c["ref"].split()]
        bucket = zlib.crc32(c["id"].encode()) % 2
        for w in c["windows"]:
            beam = [norm_word(x["text"]) for x in w["words"]]
            wins.append({"clip": c["id"], "bucket": bucket,
                         "beam": beam, "ref": ref})
    train = [w for w in wins if w["bucket"] == 0]
    test = [w for w in wins if w["bucket"] == 1]
    print(f"[info] {len(wins)} windows → train {len(train)} / test {len(test)}")

    def score(windows, beam_key):
        total_ref = sum(len(w["ref"]) for w in windows)
        matched = sum(matched_count(w[beam_key], w["ref"]) for w in windows)
        return total_ref - matched, total_ref  # خطای سمت مرجع

    # ---- آماده‌سازی: state0 و state1 ----
    for w in wins:
        w["b0"] = w["beam"]
        w["b1"] = rejoin(w["beam"])

    err0, nref = score(test, "b0")
    err1, _ = score(test, "b1")
    print(f"\n[لایه A — بازپیوند می/نمی روی test]")
    print(f"  خطا قبل: {err0} | بعد از بازپیوند: {err1} "
          f"| کاهش: {100*(err0-err1)/err0:.1f}% ({err0-err1} واژه)")

    # نمونه‌ی joinها
    shown = 0
    for w in test:
        for a, b in zip(w["beam"], w["b1"]):
            if a != b and shown < 8:
                print(f"    «{a}» + بعدی → «{b}»")
                shown += 1

    # ---- لایه B: ثبت صوتی روی text پس از بازپیوند ----
    # خطاها و جفت‌ها روی bucket 0 با b1
    errs = defaultdict(Counter)
    correct_freq = Counter()
    train_pairs = defaultdict(list)   # right → [wrong_forms به ترتیب زمانی]
    for w in train:
        pairs = align_pairs(w["b1"], w["ref"])
        m = SequenceMatcher(None, w["b1"], w["ref"], autojunk=False)
        matched_keys = set()
        for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    matched_keys.add(w["b1"][i1 + k])
        for tk in w["b1"]:
            if tk in matched_keys:
                correct_freq[tk] += 1
        for wrong, right in pairs:
            errs[right][wrong] += 1
            train_pairs[right].append(wrong)

    rng = random.Random(42)

    def make_aliases(run_seed: int):
        r = random.Random(run_seed)
        alias = {}
        log = {}
        cand = []
        for right, cnt in errs.items():
            n = sum(cnt.values())
            if n < MIN_ERRORS or len(right) < MIN_WORD_LEN:
                continue
            top_wrong, top_n = cnt.most_common(1)[0]
            if top_n / n < MIN_TOP_SHARE and right not in FORCE_WORDS:
                continue
            if top_wrong in PREFIXES or len(top_wrong) < 3:
                continue
            if correct_freq.get(top_wrong, 0) > MAX_VARIANT_COMMON:
                continue
            cand.append((right, n, cnt))
        cand.sort(key=lambda t: -t[1])
        for right, n, cnt in cand[:20]:
            # شبیه‌سازی ۳ ضبط: ۳ نمونه‌ی تصادفی از خطاهای این واژه
            sample = r.sample(train_pairs[right], min(3, len(train_pairs[right])))
            variants = [v for v in dict.fromkeys(sample)
                        if v not in PREFIXES and len(v) >= 3
                        and correct_freq.get(v, 0) <= MAX_VARIANT_COMMON][:3]
            for v in variants:
                if v in alias and alias[v] != right:
                    continue
                alias[v] = right
            log[right] = {"heard": sample, "variants": variants,
                          "n_err_train": n}
        return alias, log

    def apply_aliases(windows, alias):
        """بازگرداند متن با جایگزینی واژه‌ای alias (روی b1)."""
        res = []
        for w in windows:
            res.append([alias.get(t, t) for t in w["b1"]])
        return res

    runs = []
    for run in range(N_BOOTSTRAP):
        alias, log = make_aliases(1000 + run)
        beams = apply_aliases(test, alias)
        # امتیازدهی بعد از alias
        total_ref = sum(len(w["ref"]) for w in test)
        matched = sum(matched_count(b, w["ref"])
                      for b, w in zip(beams, test))
        err2 = total_ref - matched
        # شمارش fixed/corrupt دقیق: بر اساس تغییر matched
        runs.append({"run": run, "n_alias": len(alias), "err_after": err2,
                     "log": log})
    err2s = sorted(r["err_after"] for r in runs)
    median_err2 = err2s[len(err2s) // 2]
    print(f"\n[لایه B — ثبت صوتی (alias) روی test، {N_BOOTSTRAP} بار بوت‌استرپ]")
    print(f"  خطاها در اجراها: {err2s}")
    print(f"  میانه: {median_err2} | نسبت به پس از بازپیوند ({err1}): "
          f"{100*(err1-median_err2)/max(1,err1):.1f}% کاهش بیشتر")
    print(f"  کاهش کل نسبت به شروع: {100*(err0-median_err2)/err0:.1f}% "
          f"({err0} → {median_err2})")

    # جزئیات یک اجرا: واژه‌های ثبت‌شده و واریانت‌ها
    best = min(runs, key=lambda r: r["err_after"])
    print(f"\n[نمونه‌ی نگاشت یک اجرا] {best['n_alias']} مدخل:")
    for right, info in list(best["log"].items())[:15]:
        print(f"  «{right}» (خطای train={info['n_err_train']}) ← "
              f"شنیدها: {info['heard']} → واریانت‌ها: {info['variants']}")

    # وضعیت واژه‌های FORCE (مثل ایآی) در ارزیابی
    print("\n[واژه‌های FORCE]:")
    alias_best, _ = make_aliases(1000 + best["run"])
    for fw in FORCE_WORDS:
        variants = [v for v, t in alias_best.items() if t == fw]
        fixed_fw = 0
        bad_fw = 0
        for w in test:
            ref = w["ref"]
            for t in alias_best and w["b1"]:
                pass
            for t in w["b1"]:
                if alias_best.get(t) == fw:
                    # آیا مرجع در جایگاهش fw است؟ تقریب: شمارش ساده
                    fixed_fw += 1 if fw in ref else 0
            bad_fw += sum(1 for t in w["b1"] if t == fw)
        print(f"  «{fw}»: واریانت‌های ثبت‌شده={variants} | "
              f"تکرار واریانت در test: {bad_fw}")

    out = {
        "err_before": err0, "err_after_rejoin": err1,
        "err_after_alias_runs": err2s, "err_median": median_err2,
        "reduction_rejoin_pct": round(100*(err0-err1)/err0, 1),
        "reduction_total_pct": round(100*(err0-median_err2)/err0, 1),
        "runs": [{k: r[k] for k in ("run", "n_alias", "err_after")}
                 for r in runs],
    }
    out_path = Path(__file__).parent / "report_final_sim.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
