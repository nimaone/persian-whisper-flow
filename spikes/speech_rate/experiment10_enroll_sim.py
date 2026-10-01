"""آزمایش ۱۰ — شبیه‌سازی کامل ثبت صوتی واژه (enrollment) و سنجش بهبود دقت.

سناریو (مطابق پیشنهاد محصول):
  1. کاربر واژه‌های مشکل‌دار را «ثبت» می‌کند → شبیه‌سازی: واژه‌هایی با بیشترین
     خطای پایدار در نیمه‌ی اول کلیپ‌ها (bucket 0)
  2. ثبت = ۳ نمونه‌ی صوتی → شبیه‌سازی: ۳ مشاهده‌ی اول از کلیپ‌های bucket 0؛
     واریانت = شکل غلط اکثریت (همان که UI به کاربر نشان می‌دهد)
  3. نگاشت واریانت→واژه‌ی درست ساخته می‌شود (با چک تداخل)
  4. ارزیابی روی bucket 1: چند خطا اصلاح می‌شود؟ چند واژه‌ی درست خراب می‌شود؟
     اثر خالص بر نرخ خطای واژه؟

این سنجه فقط لایه‌ی متن (نگاشت) را می‌سنجد؛ لایه‌ی beam (تقویت واریانت)
روی این، اضافه می‌کند. خروجی: report_enroll_sim.json
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
MIN_WORD_LEN = 4        # واژه‌های کوتاه/رایج کاندید ثبت نیستند
TOP_N_ENROLL = 20       # کاربر ۲۰ واژه‌ی مشکل‌دار را ثبت می‌کند
MIN_DETERMINISM = 0.5   # حداقل سهم واریانت برتر در نمونه‌های ثبت


def norm_word(w: str) -> str:
    w = w.replace("\u200c", "")
    w = w.translate(str.maketrans("يكٱىى", "یکییی"))
    return "".join(ch for ch in w if ch.isalnum())


def main():
    data = pickle.loads(CACHE.read_bytes())
    clips = data["clips"]
    print(f"[info] {len(clips)} clips")

    # ---- نگاشت هر واژه‌ی پنجره به واژه‌ی مرجع هم‌تراز (یا None) ----
    # instances: (clip_id, bucket, win_idx, widx, text_key, ref_key|None)
    instances = []
    for c in clips:
        ref = [norm_word(w) for w in c["ref"].split()]
        bucket = hash(c["id"]) % 2
        for wi, w in enumerate(c["windows"]):
            beam = [norm_word(x["text"]) for x in w["words"]]
            m = SequenceMatcher(None, beam, ref, autojunk=False)
            aligned = [None] * len(beam)
            for tag, i1, i2, j1, j2 in m.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        aligned[i1 + k] = ref[j1 + k]
                elif tag == "replace":
                    for k in range(min(i2 - i1, j2 - j1)):
                        aligned[i1 + k] = ref[j1 + k]  # جفت‌سازی جایگزینی
            for widx, x in enumerate(w["words"]):
                instances.append({
                    "clip": c["id"], "bucket": bucket, "win": wi,
                    "text": norm_word(x["text"]),
                    "ref": aligned[widx],
                    "conf": x["conf"],
                })
    print(f"[info] {len(instances)} word instances")

    # ---- ۱) انتخاب واژه‌های کاندید ثبت (از bucket 0) ----
    err_by_right = defaultdict(list)  # right → [wrong_key]
    for it in instances:
        if it["bucket"] == 0 and it["ref"] and it["text"] != it["ref"] \
                and len(it["ref"]) >= MIN_WORD_LEN:
            err_by_right[it["ref"]].append(it["text"])

    enrolled = {}   # right → variant
    skipped = []
    cand_sorted = sorted(err_by_right.items(), key=lambda kv: -len(kv[1]))
    for right, wrongs in cand_sorted:
        if len(enrolled) >= TOP_N_ENROLL:
            break
        # ثبت = ۳ نمونه‌ی اول؛ واریانت = اکثریت بین همان ۳
        sample_wrongs = wrongs[:N_ENROLL_SAMPLES]
        variant, n = Counter(sample_wrongs).most_common(1)[0]
        determinism = n / len(wrongs)
        if determinism < MIN_DETERMINISM or len(variant) < MIN_WORD_LEN:
            skipped.append((right, "بی‌ثبات یا کوتاه", determinism))
            continue
        if variant in enrolled and enrolled[variant] != right:
            skipped.append((right, f"تداخل با {enrolled[variant]}", determinism))
            continue
        enrolled[variant] = right
    print(f"\n[ثبت‌شده] {len(enrolled)} واژه (از {TOP_N_ENROLL} کاندید):")
    for v, r in enrolled.items():
        heard = [w for w in err_by_right[r][:N_ENROLL_SAMPLES]]
        print(f"  «{r}» ← مدل می‌شنود: {heard} → نگاشت از «{v}»")
    print(f"  ردشده: {len(skipped)} ({Counter(s[1] for s in skipped)})")

    # ---- ۲) ارزیابی روی bucket 1 ----
    before_bad = after_bad = 0
    fixed = corrupted = 0
    per_word = defaultdict(lambda: {"bad": 0, "fixed": 0, "corrupt": 0})
    for it in instances:
        if it["bucket"] != 1:
            continue
        key = it["text"]
        ref = it["ref"]
        is_bad = (ref is None) or (key != ref)
        if is_bad:
            before_bad += 1
        if key in enrolled:
            target = enrolled[key]
            if ref == target:
                # خطای واژه‌ی ثبت‌شده که واریانتش آمده → اصلاح
                fixed += 1
                per_word[target]["fixed"] += 1
            elif ref is not None and key == ref:
                # واژه درستِ هم‌شکل با واریانت → خراب می‌شود
                corrupted += 1
                per_word[target]["corrupt"] += 1
            # ref None یا ref دیگر: همچنان خطا (بدون تغییر شمارش)
        if ref is None or key != ref:
            # شمارش بعد از اصلاح: اگر این instance اصلاح شده، دیگر خطا نیست
            if not (key in enrolled and enrolled[key] == ref):
                after_bad += 1
            if key in enrolled:
                per_word[enrolled[key]]["bad"] += 1

    print(f"\n[ارزیابی روی bucket 1]")
    print(f"  خطای واژه‌ای قبل:  {before_bad}")
    print(f"  خطای واژه‌ای بعد:  {after_bad}")
    print(f"  اصلاح‌شده:         {fixed}")
    print(f"  خراب‌شده (FP):     {corrupted}")
    if before_bad:
        print(f"  کاهش خالص خطا:    {100*(before_bad-after_bad)/before_bad:.1f}%")

    print("\n[به تفکیک واژه‌ی ثبت‌شده] bad → fixed (corrupt):")
    for r, s in sorted(per_word.items(), key=lambda kv: -kv[1]["fixed"]):
        if s["fixed"] or s["corrupt"]:
            print(f"  «{r}»: {s['bad']} خطا → {s['fixed']} اصلاح، "
                  f"{s['corrupt']} خرابی")

    out = {
        "n_enrolled": len(enrolled),
        "enrolled_map": enrolled,
        "before_bad": before_bad, "after_bad": after_bad,
        "fixed": fixed, "corrupted": corrupted,
        "reduction_pct": round(100 * (before_bad - after_bad) / max(1, before_bad), 1),
        "per_word": {r: dict(s) for r, s in per_word.items()},
    }
    out_path = Path(__file__).parent / "report_enroll_sim.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
