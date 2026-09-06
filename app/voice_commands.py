"""فرمان‌های صوتی فارسی — تبدیل کلمات کلیدی گفتاری به نشانه/کلید.

خروجی parse یک لیست از سگمنت‌هاست: متن قابل‌درج یا کلید قابل‌ارسال.
"""
from __future__ import annotations

from dataclasses import dataclass

# کلمه(ها)ی گوینده → اکشن
# نکته: کلمات تک‌واژه‌ای فقط آن‌هایی باشند که در گفتار عادی به‌عنوان واژه
# به‌کار نمی‌روند («سوال» تنها حذف شد چون در جمله عادی false-positive می‌داد)
COMMANDS = {
    "نقطه": ".",
    "ویرگول": "،",
    "علامت سوال": "؟",
    "نقطه ویرگول": "؛",
    "گیومه": "«»",
    "گیومه باز": "«",
    "گیومه بسته": "»",
    "خط جدید": "\n",
    "خط بعد": "\n",
}


@dataclass
class Segment:
    kind: str  # "text" | "key" | "punct"
    value: str  # متن، یا نام کلید برای kind="key"


def parse(text: str) -> list[Segment]:
    """متن نهایی را به سگمنت‌های درج/کلید تبدیل می‌کند.

    «حذف آخرین کلمه» به‌صورت پاک‌کردن آخرین کلمه‌ی درج‌شده قبل از خودش
    پیاده‌سازی نمی‌شود اینجا — در سطح main مدیریت می‌شود (نوع key).
    """
    segments: list[Segment] = []
    # اول فرمان‌های چندکلمه‌ای، بعد تک‌کلمه‌ای — ترتیب مهم است
    multi = sorted(COMMANDS.keys(), key=len, reverse=True)
    i = 0
    words = text.split()
    n = len(words)
    while i < n:
        matched = False
        # بررسی ترکیب ۳، ۲ و ۱ کلمه‌ای از اینجا به بعد
        for span in (3, 2, 1):
            if i + span > n:
                continue
            phrase = " ".join(words[i:i + span])
            if phrase in COMMANDS:
                val = COMMANDS[phrase]
                if val == "\n":
                    segments.append(Segment("key", "enter"))
                else:
                    segments.append(Segment("punct", val))
                i += span
                matched = True
                break
        if matched:
            continue
        # عبارت «حذف آخرین کلمه» / «پاک کن»
        if i + 2 < n and words[i] == "حذف" and words[i + 1] in ("آخرین", "اخرین") and words[i + 2] == "کلمه":
            segments.append(Segment("key", "delete_word"))
            i += 3
            continue
        if i + 1 < n and words[i] == "حذف" and words[i + 1] == "کن":
            segments.append(Segment("key", "delete_word"))
            i += 2
            continue
        # کلمه عادی — به سگمنت متن جاری اضافه/ایجاد
        if segments and segments[-1].kind == "text":
            segments[-1].value += " " + words[i]
        else:
            segments.append(Segment("text", words[i]))
        i += 1
    return segments


def execute(segments: list[Segment], insert_fn, key_fn):
    """اجرای سگمنت‌ها به ترتیب: درج متن/نشانه، ارسال کلید.

    insert_fn(text) — درج متن در محل کرسر.
    key_fn(name)  — ارسال کلید (enter / delete_word).
    """
    for seg in segments:
        if seg.kind == "key" and seg.value == "delete_word":
            key_fn("delete_word")
            continue
        if seg.kind == "key":
            key_fn(seg.value)
            continue
        if seg.kind == "punct":
            if seg.value == "\n":
                key_fn("enter")
            else:
                insert_fn(seg.value)
            continue
        # text
        insert_fn(seg.value)
