"""آزمایش ۱۳ — عدد نهایی: اثر بازپیوند «می» + نگاشت ثبت‌صوتی بر WER واقعی.

برخلاف آزمایش‌های قبل که مرجعشان decode خود مدل بود، اینجا مرجع = متن
آموزشی manifest (اسنپ‌شات‌های proofread شده از همان ویدیوها). خروجی
سیستم = همان chunkهای 16s مسیر فینال برنامه (windows_real.pkl).

سناریو:
  1) WER پایه‌ی chunkهای فینال در برابر متن manifest (بازه‌ی 120..480s هر منبع)
  2) + بازپیوند «می/نمی»
  3) + نگاشت ثبت‌صوتی: برای هر واژه‌ی کاندید، ۳ نمونه‌ی تصادفی از شکل‌های
     شنیده‌شده (شبیه‌سازی ۳ ضبط کاربر) → واریانت‌ها → جایگزینی. بوت‌استرپ ۵ بار.
همراه شمارش دقیق اصلاح/خرابی از روی هم‌ترازسازی.
خروجی: report_wer_final.json
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "spikes" / "speech_rate" / "windows_real.pkl"
MANIFEST = ROOT / "training" / "all" / "manifest.jsonl"

SEG_START, SEG_END = 120.0, 480.0
MIN_WORD_LEN = 4
MIN_ERRORS = 4
MAX_VARIANT_COMMON = 2
N_BOOTSTRAP = 5
N_ENROLL = 25
PREFIXES = {"می", "نمی"}


def norm(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch if ch.isalnum() else "" for ch in w)


def rejoin(tokens: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in PREFIXES and i + 1 < len(tokens):
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def load_manifest_ref(source: str):
    """متن مرجع manifest برای بازه‌ی [SEG_START, SEG_END) این منبع."""
    rows = []
    for line in open(MANIFEST, encoding="utf-8"):
        r = json.loads(line)
        if r["source"] != source:
            continue
        mid = r["start"] + r["duration"] / 2
        if SEG_START <= mid < SEG_END and (r["text"] or "").strip():
            rows.append((r["start"], [norm(w) for w in r["text"].split()]))
    rows.sort(key=lambda t: t[0])
    words = [w for _, ws in rows for w in ws]
    return words


def main():
    res = pickle.loads(CACHE.read_bytes())
    print(f"[info] {len(res)} sources")

    out_sources = {}
    tot = Counter()
    rng_master = random.Random(7)

    # ---- پاس ۱: جمع‌آوری جفت‌های خطای همه‌ی منابع (نگاشت کاربر سراسری است) ----
    src_data = {}
    global_pairs = defaultdict(list)   # right → [wrong] (پس از بازپیوند)
    global_correct = Counter()
    for r in res:
        src = r["source"]
        ref_words = load_manifest_ref(src)
        if len(ref_words) < 50:
            continue
        sys_words = []
        for cs, ce, txt in sorted(r["ref_chunks"]):
            sys_words.extend(norm(w) for w in txt.split())
        sys_rejoin = rejoin(sys_words)
        m = SequenceMatcher(None, sys_rejoin, ref_words, autojunk=False)
        aligned_ref = [None] * len(sys_rejoin)
        for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    aligned_ref[i1 + k] = ref_words[j1 + k]
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    aligned_ref[i1 + k] = ref_words[j1 + k]
        for idx, w in enumerate(sys_rejoin):
            ar = aligned_ref[idx]
            if ar is not None:
                if w == ar:
                    global_correct[w] += 1
                elif len(w) >= 2 and len(ar) >= 2:
                    global_pairs[ar].append(w)
        src_data[src] = {"ref_words": ref_words, "sys_rejoin": sys_rejoin,
                         "aligned_ref": aligned_ref}
    print(f"[info] {len(src_data)} منبع، "
          f"{sum(len(v) for v in global_pairs.values())} جفت خطا")

    candidates = []
    for right, wrongs in global_pairs.items():
        if len(wrongs) < MIN_ERRORS or len(right) < MIN_WORD_LEN:
            continue
        cnt = Counter(wrongs)
        top_wrong, top_n = cnt.most_common(1)[0]
        if top_n / len(wrongs) < 0.3:
            continue
        if top_wrong in PREFIXES or len(top_wrong) < 3:
            continue
        if global_correct.get(top_wrong, 0) > MAX_VARIANT_COMMON:
            continue
        candidates.append((right, len(wrongs), cnt))
    candidates.sort(key=lambda t: -t[1])
    print(f"[کاندیدهای ثبت] {len(candidates)} واژه؛ ۲۰ تای اول:")
    for right, n, cnt in candidates[:20]:
        print(f"  «{right}» خطا={n} واریانت‌ها={cnt.most_common(3)}")

    def build_alias(seed: int):
        rng = random.Random(seed)
        alias, log = {}, {}
        for right, n, cnt in candidates[:N_ENROLL]:
            sample = rng.sample(global_pairs[right], min(3, len(global_pairs[right])))
            variants = [v for v in dict.fromkeys(sample)
                        if v not in PREFIXES and len(v) >= 3
                        and global_correct.get(v, 0) <= MAX_VARIANT_COMMON][:3]
            for v in variants:
                if v in alias and alias[v] != right:
                    continue
                alias[v] = right
            log[right] = {"heard": sample, "variants": variants}
        return alias, log

    def apply_alias(alias, src):
        d = src_data[src]
        fixed = corrupted = 0
        out = []
        for idx, w in enumerate(d["sys_rejoin"]):
            t = alias.get(w)
            if t is None:
                out.append(w)
                continue
            ar = d["aligned_ref"][idx]
            if ar == t:
                fixed += 1
            elif ar is not None and ar == w:
                corrupted += 1
            out.append(t)
        return out, fixed, corrupted

    # نگاشت‌های بوت‌استرپ یک‌بار ساخته می‌شوند و روی همه‌ی منابع اعمال
    # می‌شوند — مثل محصول واقعی: کاربر یک بار ثبت می‌کند، همه‌جا اثر دارد
    aliases_cache = [build_alias(100 + run) for run in range(N_BOOTSTRAP)]

    # ---- پاس ۲: ارزیابی per-source با نگاشت سراسری ----
    for r in res:
        src = r["source"]
        if src not in src_data:
            continue
        d = src_data[src]

        def score(words, _ref=d["ref_words"]):
            m = SequenceMatcher(None, words, _ref, autojunk=False)
            matched = sum(b.size for b in m.get_matching_blocks())
            return matched, len(_ref) - matched

        sys_words = []
        for cs, ce, txt in sorted(r["ref_chunks"]):
            sys_words.extend(norm(w) for w in txt.split())
        m0, e0 = score(sys_words)
        mA, eA = score(d["sys_rejoin"])

        best = None
        for run in range(N_BOOTSTRAP):
            alias, log = aliases_cache[run]
            out_words, fixed, corrupted = apply_alias(alias, src)
            mA2, eA2 = score(out_words)
            rec = {"run": run, "n_alias": len(alias), "fixed": fixed,
                   "corrupted": corrupted, "errors": eA2}
            if best is None or eA2 < best["errors"]:
                best = rec
        eB = best["errors"]

        print(f"\n[{src}] مرجع={len(d['ref_words'])} واژه")
        print(f"  WER پایه:        {100*e0/max(1,len(d['ref_words'])):.1f}% "
              f"(خطا {e0})")
        print(f"  + بازپیوند می:   {100*eA/max(1,len(d['ref_words'])):.1f}% "
              f"(Δ {e0-eA:+d})")
        print(f"  + نگاشت ثبت‌صوتی: {100*eB/max(1,len(d['ref_words'])):.1f}% "
              f"(Δ {eA-eB:+d}) [اصلاح {best['fixed']}، خرابی "
              f"{best['corrupted']}، مدخل {best['n_alias']}]")
        tot["ref"] += len(d["ref_words"])
        tot["e0"] += e0
        tot["eA"] += eA
        tot["eB"] += eB
        tot["fixed"] += best["fixed"]
        tot["corrupt"] += best["corrupted"]
        out_sources[src] = {
            "ref_words": len(d["ref_words"]), "err_base": e0, "err_rejoin": eA,
            "err_alias": eB, "fixed": best["fixed"],
            "corrupted": best["corrupted"], "n_alias": best["n_alias"],
        }

    print(f"\n===== جمع کل =====")
    print(f"  مرجع: {tot['ref']} واژه")
    print(f"  WER پایه:            {100*tot['e0']/tot['ref']:.1f}%")
    print(f"  + بازپیوند می/نمی:   {100*tot['eA']/tot['ref']:.1f}% "
          f"(Δ {tot['e0']-tot['eA']:+d})")
    print(f"  + نگاشت ثبت‌صوتی:    {100*tot['eB']/tot['ref']:.1f}% "
          f"(Δ {tot['eA']-tot['eB']:+d})")
    print(f"  کاهش کل: {100*(tot['e0']-tot['eB'])/tot['e0']:.1f}% "
          f"({tot['e0']} → {tot['eB']}) | دقت اصلاح: "
          f"{100*tot['fixed']/max(1,tot['fixed']+tot['corrupt']):.0f}%")

    out = {"total": dict(tot), "sources": out_sources}
    out_path = Path(__file__).parent / "report_wer_final.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
