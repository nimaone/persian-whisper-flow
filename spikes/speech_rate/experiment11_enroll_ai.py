"""آزمایش ۱۱ — شبیه‌سازی ثبت صوتی با تمرکز واژه‌های پرتکرار فنی/AI.

تفاوت با آزمایش ۱۰ (که کاندیدهای بد انتخاب کرد و ۹۱ خرابی ساخت):
  - فیلتر امنیتی: واریانت نباید خودش واژه‌ی رایج مرجع باشد (وگرنه
    جایگزینی، واژه‌های درستِ هم‌شکل را خراب می‌کند — «کرده»→«کردیم»)
  - تمرکز روی واژه‌های پرتکرار با خطای بالا: «ایآی»، «ریسرچ»، «کلاد»…
  - تا ۲ واریانت برای هر واژه (کاربر در UI هر دو را تأیید می‌کند)

خروجی: report_enroll_ai.json
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

N_ENROLL_SAMPLES = 3
MIN_WORD_LEN = 4
TOP_N_ENROLL = 20
MIN_ERRORS = 5          # حداقل خطا در bucket 0 تا کاندید شود
MIN_TOP_SHARE = 0.4     # حداقل سهم واریانت برتر
MAX_VARIANT_COMMON = 2  # واریانت حداکثر چند بار مجاز است به‌عنوان واژه‌ی درست دیده شود
ALSO_REPORT = ["ایآی", "ریسرچ", "کلاد", "دیتا", "مدلا", "هوش", "مصنوعی",
               "الگوریتم", "چت", "جیپیتی", "دیتاسنتر", "اپلیکیشن"]


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]

    instances = []
    for c in clips:
        ref = [norm_word(w) for w in c["ref"].split()]
        bucket = hash(c["id"]) % 2
        for w in c["windows"]:
            beam = [norm_word(x["text"]) for x in w["words"]]
            m = SequenceMatcher(None, beam, ref, autojunk=False)
            aligned = [None] * len(beam)
            for tag, i1, i2, j1, j2 in m.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        aligned[i1 + k] = ref[j1 + k]
                elif tag == "replace":
                    for k in range(min(i2 - i1, j2 - j1)):
                        aligned[i1 + k] = ref[j1 + k]
            for widx, x in enumerate(w["words"]):
                instances.append({
                    "clip": c["id"], "bucket": bucket,
                    "text": norm_word(x["text"]), "ref": aligned[widx],
                })
    print(f"[info] {len(instances)} word instances")

    # ---- آمار bucket 0: خطا و رایجی هر واژه‌ی مرجع ----
    errs = defaultdict(Counter)      # right → Counter(wrong)
    correct_freq = Counter()         # چند بار یک کلید به‌عنوان واژه‌ی درست آمده
    for it in instances:
        if it["bucket"] != 0:
            continue
        if it["ref"] is not None and it["text"] == it["ref"]:
            correct_freq[it["ref"]] += 1
        elif it["ref"] is not None and it["text"] != it["ref"] \
                and len(it["text"]) >= 2 and len(it["ref"]) >= 2:
            errs[it["ref"]][it["text"]] += 1

    # ---- کاندیدهای ثبت ----
    candidates = []
    for right, cnt in errs.items():
        n = sum(cnt.values())
        if n < MIN_ERRORS or len(right) < MIN_WORD_LEN:
            continue
        top_wrong, top_n = cnt.most_common(1)[0]
        share = top_n / n
        if share < MIN_TOP_SHARE:
            continue
        # فیلتر امنیتی: واریانت نباید خودش واژه‌ی رایجِ درست باشد
        if correct_freq.get(top_wrong, 0) > MAX_VARIANT_COMMON:
            continue
        candidates.append({
            "right": right, "n_err": n, "variant": top_wrong,
            "share": round(share, 2),
            "variant_common_as_correct": correct_freq.get(top_wrong, 0),
        })
    candidates.sort(key=lambda c: -c["n_err"])
    print(f"\n[کاندیدهای امن] {len(candidates)} واژه "
          f"(≥{MIN_ERRORS} خطا، واریانت غیررایج):")
    for c in candidates[:25]:
        print(f"  «{c['right']}» خطا={c['n_err']:3d} واریانت=«{c['variant']}» "
              f"سهم={c['share']:.0%} رایجی‌وارینت={c['variant_common_as_correct']}")

    # واژه‌های خاص درخواستی — حتی اگر در لیست بالا نبودند گزارش شوند
    print("\n[واژه‌های خاص]:")
    for w in ALSO_REPORT:
        st = next((c for c in candidates if c["right"] == w), None)
        if st:
            print(f"  «{w}»: کاندید ✓ خطا={st['n_err']} واریانت=«{st['variant']}»")
        elif w in errs:
            cnt = errs[w]
            n = sum(cnt.values())
            print(f"  «{w}»: {n} خطا ولی کاندید نشد — واریانت‌ها: "
                  f"{cnt.most_common(3)}")
        else:
            print(f"  «{w}»: در خطاهای bucket0 نیست")

    # ---- ثبت: ۳ نمونه‌ی اول، تا ۲ واریانت متمایز ----
    alias_map = {}
    enroll_log = {}
    for c in candidates[:TOP_N_ENROLL]:
        right = c["right"]
        wrongs = [t["text"] for t in instances
                  if t["bucket"] == 0 and t["ref"] == right
                  and t["text"] != right][:N_ENROLL_SAMPLES]
        variants = [v for v in dict.fromkeys(wrongs)
                    if correct_freq.get(v, 0) <= MAX_VARIANT_COMMON
                    and len(v) >= 3][:2]
        if not variants:
            continue
        forms = []
        for v in variants:
            if v in alias_map and alias_map[v] != right:
                continue  # تداخل
            alias_map[v] = right
            forms.append(v)
        if forms:
            enroll_log[right] = forms
    print(f"\n[نگاشت نهایی] {len(alias_map)} مدخل از {len(enroll_log)} واژه:")
    for r, vs in enroll_log.items():
        print(f"  «{r}» ← {vs}")

    # ---- ارزیابی روی bucket 1 ----
    before_bad = after_bad = fixed = corrupted = 0
    per_word = defaultdict(lambda: {"bad": 0, "fixed": 0, "corrupt": 0})
    for it in instances:
        if it["bucket"] != 1:
            continue
        key, ref = it["text"], it["ref"]
        is_bad = (ref is None) or (key != ref)
        if is_bad:
            before_bad += 1
        if key in alias_map:
            target = alias_map[key]
            if ref == target:
                fixed += 1
                per_word[target]["fixed"] += 1
            elif ref is not None and key == ref:
                corrupted += 1
                per_word[target]["corrupt"] += 1
        if ref is None or key != ref:
            if not (key in alias_map and alias_map[key] == ref):
                after_bad += 1
            if key in alias_map:
                per_word[alias_map[key]]["bad"] += 1

    print(f"\n[ارزیابی bucket 1]")
    print(f"  خطای واژه‌ای قبل: {before_bad} | بعد: {after_bad} "
          f"| کاهش: {100*(before_bad-after_bad)/max(1,before_bad):.1f}%")
    print(f"  اصلاح‌شده: {fixed} | خراب‌شده: {corrupted} "
          f"| دقت اصلاح: {100*fixed/max(1, fixed+corrupted):.0f}%")
    print("[به تفکیک واژه]:")
    for r, s in sorted(per_word.items(), key=lambda kv: -kv[1]["fixed"]):
        if s["fixed"] or s["corrupt"]:
            print(f"  «{r}»: {s['bad']} خطا → {s['fixed']} اصلاح، "
                  f"{s['corrupt']} خرابی")

    out = {
        "candidates": candidates[:30],
        "alias_map": alias_map,
        "before_bad": before_bad, "after_bad": after_bad,
        "fixed": fixed, "corrupted": corrupted,
        "reduction_pct": round(100 * (before_bad - after_bad) / max(1, before_bad), 1),
        "per_word": {r: dict(s) for r, s in per_word.items()},
    }
    out_path = Path(__file__).parent / "report_enroll_ai.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
