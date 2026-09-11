"""آزمایش ۹ — سنجش عملی ایده‌ی «ثبت صوتی واژه» (enrollment).

پرسش: اگر کاربر واژه را صوتی ثبت کند و مدل بگوید «چه شنید»، آیا یک
نگاشت نام‌مستعار (شکل غلط → شکل درست) می‌تواند خطاها را اصلاح کند؟

روش: از کش ۳۵۰ کلیپ (windows_multi.pkl) جفت‌های (شکل غلط مدل → واژه‌ی
مرجع) از روی opcodes هم‌ترازسازی استخراج می‌شود. سپس:
  - ثبات: برای هر واژه‌ی مرجع، سهم رایج‌ترین شکل غلط (determinism)
  - تعمیم: نگاشت روی نیمی از کلیپ‌ها ساخته، روی نیم دیگر سنجیده می‌شود
    (leave-split) — چند درصد خطاها واقعا اصلاح می‌شوند؟
خروجی: spikes/speech_rate/report_alias_map.json
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "spikes" / "speech_rate" / "windows_multi.pkl"


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]
    print(f"[info] {len(clips)} clips")

    # ---- استخراج جفت‌های (شکل غلط → واژه‌ی مرجع) ----
    pairs = []  # (clip_id, wrong_form, correct_form)
    for c in clips:
        ref = [norm_word(w) for w in c["ref"].split()]
        for w in c["windows"]:
            beam = [norm_word(x["text"]) for x in w["words"]]
            m = SequenceMatcher(None, beam, ref, autojunk=False)
            for tag, i1, i2, j1, j2 in m.get_opcodes():
                if tag != "replace":
                    continue
                for k in range(min(i2 - i1, j2 - j1)):
                    wrong, right = beam[i1 + k], ref[j1 + k]
                    if wrong and right and wrong != right \
                            and len(wrong) >= 2 and len(right) >= 2:
                        pairs.append((c["id"], wrong, right))

    print(f"[info] {len(pairs)} جفت غلط→درست استخراج شد")

    # ---- ثبات: هر واژه‌ی مرجع چند شکل غلط دارد؟ ----
    by_right = defaultdict(Counter)
    for _, wrong, right in pairs:
        by_right[right][wrong] += 1
    words_with_data = {r: cnt for r, cnt in by_right.items()
                       if sum(cnt.values()) >= 3}
    print(f"[info] واژه‌های مرجع با ≥۳ مشاهده‌ی خطا: {len(words_with_data)}")

    det_buckets = Counter()
    top_examples = []
    for right, cnt in sorted(words_with_data.items(),
                             key=lambda kv: -sum(kv[1].values()))[:400]:
        n = sum(cnt.values())
        top_wrong, top_n = cnt.most_common(1)[0]
        share = top_n / n
        n_variants = len(cnt)
        if share >= 0.8:
            det_buckets["≥80% یک شکل"] += 1
        elif share >= 0.5:
            det_buckets["۵۰-۸۰%"] += 1
        else:
            det_buckets["<50% (بی‌ثبات)"] += 1
        top_examples.append({
            "correct": right, "n": n, "variants": n_variants,
            "top_share": round(share, 2),
            "top_variants": [f"{w}({c})" for w, c in cnt.most_common(3)],
        })

    total = sum(det_buckets.values())
    print("\n[ثبات شکل غلط] (سهم رایج‌ترین شکل غلط برای هر واژه):")
    for k, v in det_buckets.items():
        print(f"  {k:22s} {v:3d} واژه ({100*v/max(1,total):.0f}%)")

    print("\n[نمونه‌ها] پرتکرارترین جفت‌ها:")
    for e in top_examples[:15]:
        print(f"  «{e['correct']}» n={e['n']:3d} واریانت={e['variants']:2d} "
              f"سهم_بالا={e['top_share']:.0%} ← {e['top_variants']}")

    # ---- تعمیم: نگاشت از نیمه‌ی اول، ارزیابی روی نیمه‌ی دوم ----
    train_map = defaultdict(Counter)
    eval_pairs = []
    for cid, wrong, right in pairs:
        h = hash(cid) % 2
        if h == 0:
            train_map[right][wrong] += 1
        else:
            eval_pairs.append((cid, wrong, right))

    alias_map = {}
    for right, cnt in train_map.items():
        for wrong, c in cnt.most_common(2):  # دو شکل رایج هر واژه
            if c >= 2 and len(wrong) >= 3:
                alias_map[wrong] = right

    fixed = ambiguous = unfixable = 0
    fixed_words = Counter()
    for _, wrong, right in eval_pairs:
        if wrong in alias_map:
            if alias_map[wrong] == right:
                fixed += 1
                fixed_words[right] += 1
            else:
                ambiguous += 1  # نگاشت به واژه‌ی دیگری می‌رود — تداخل
        else:
            unfixable += 1
    n_eval = len(eval_pairs)
    print(f"\n[ارزیابی leave-split] از {n_eval} خطای نیمه‌ی ارزیابی:")
    print(f"  اصلاح‌شده با نگاشت:      {fixed:4d} ({100*fixed/n_eval:.0f}%)")
    print(f"  تداخل نگاشت (ناامن):     {ambiguous:4d} ({100*ambiguous/n_eval:.0f}%)")
    print(f"  بدون نگاشت (شکل تازه):   {unfixable:4d} ({100*unfixable/n_eval:.0f}%)")
    print(f"  اندازه‌ی نگاشت: {len(alias_map)} مدخل")
    print("  بیشترین اصلاح‌ها:", fixed_words.most_common(10))

    out = {
        "pairs_total": len(pairs),
        "n_ref_words_with_data": len(words_with_data),
        "determinism_buckets": dict(det_buckets),
        "top_examples": top_examples[:40],
        "leave_split": {
            "n_eval": n_eval, "fixed": fixed, "ambiguous": ambiguous,
            "unfixable": unfixable, "map_size": len(alias_map),
        },
    }
    out_path = Path(__file__).parent / "report_alias_map.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
