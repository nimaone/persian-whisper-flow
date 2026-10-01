"""ثبت صوتی واژه (enrollment) — نگاشت شکل‌های شنیده‌شده به واژه‌ی درست.

مدل برای بعضی واژه‌ها (اسم‌ها، برندها، «ای‌آی» و…) مدام شکل غلطی می‌شنود.
کاربر واژه را در تنظیمات چند بار صوتی ثبت می‌کند؛ شکل‌هایی که مدل از
ضبط‌ها می‌شنود به‌عنوان «واریانت» ذخیره می‌شوند و پس از تأیید کاربر، در
خروجی زنده و نهایی جای واژه‌ی درست می‌نشینند.

سنجش (آزمایش ۱۲/۱۳ روی ۳۵۰ کلیپ): ۲۰ مدخل با ۳ ضبط ≈ ۳۴ خطا کمتر،
دقت اصلاح ۹۲٪ — تأیید کاربر روی واریانت‌ها همان سد امنیتی است که
خرابی‌های ۵ موردی فیلتر خودکار را حذف می‌کند.

ساختار فایل enrollments.json:
    {"entries": [{"word": "...", "variants": ["..."], "enabled": true}]}
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from app.config import app_data_dir

MI_PREFIXES = ("می", "نمی")
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_AR2FA = str.maketrans("يكٱىى", "یکییی")


def norm_word(w: str) -> str:
    """کلید مقایسه: بدون اعراب/ZWNJ، ی/ک عربی → فارسی، فقط حروف و رقم."""
    w = _DIACRITICS_RE.sub("", w.replace("\u200c", "")).translate(_AR2FA)
    return "".join(ch for ch in w if ch.isalnum())


def store_path() -> Path:
    return app_data_dir() / "enrollments.json"


class EnrollStore:
    """فهرست واژه‌های ثبت‌شده — ماندگار در %APPDATA%."""

    def __init__(self, entries: list[dict] | None = None):
        self.entries = entries or []

    @classmethod
    def load(cls) -> "EnrollStore":
        try:
            data = json.loads(store_path().read_text(encoding="utf-8"))
            entries = data.get("entries")
            if isinstance(entries, list):
                return cls([e for e in entries
                            if isinstance(e, dict) and e.get("word")])
        except Exception:
            pass
        return cls()

    def save(self):
        store_path().write_text(
            json.dumps({"entries": self.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_entry(self, word: str, variants: list[str]):
        """مدخل جدید (یا به‌روزرسانی واریانت‌های مدخل موجود)."""
        word = word.strip()
        self.entries = [e for e in self.entries
                        if norm_word(e.get("word", "")) != norm_word(word)]
        self.entries.append({
            "word": word,
            "variants": [v for v in variants if v and v.strip()],
            "enabled": True,
        })

    def remove_entry(self, word: str):
        self.entries = [e for e in self.entries if e.get("word") != word]

    def active(self) -> list[dict]:
        return [e for e in self.entries if e.get("enabled", True)]


def harvest_variants(engine, samples, correct_word: str,
                     text: str | None = None) -> list[str]:
    """دیکد ضبط واژه → «شکل شنیده‌شده» به‌صورت عبارت کامل (نه تکه‌تکه).

    هر توکنِ جدا به‌عنوان واریانت خطرناک است: تکه‌ای مثل «فلو» در جمله‌های
    عادی هم می‌آید و به اشتباه به واژه‌ی کامل تبدیل می‌شد. واریانت = کل
    عبارت شنیده‌شده که تطبیق چندتوکنی روی آن انجام می‌شود. توکن‌های پیشوند
    می/نمی و تک‌حرفی (نویز) از عبارت حذف می‌شوند. اگر مدل دقیقاً خود واژه
    را شنیده باشد، واریانتی لازم نیست.
    """
    if samples is None or getattr(samples, "size", 0) == 0:
        return []
    if text is None:
        try:
            text = engine.transcribe(samples, 16000)
        except Exception:
            return []
    toks = [t for t in text.split()
            if t not in MI_PREFIXES and len(norm_word(t)) >= 2]
    if not toks:
        return []
    if len(toks) == 1 and norm_word(toks[0]) == norm_word(correct_word):
        return []  # مدل درست شنیده — چیزی برای اصلاح نیست
    return [" ".join(toks)]


def build_alias_map(entries: list[dict]) -> dict[str, str]:
    """واریانت‌های تأییدشده → واژه‌ی درست.

    کلید = کلید نرمال واریانت (بدون فاصله/نیم‌فاصله/اعراب). اگر یک کلید
    به دو واژه‌ی مختلف برسد، مبهم است و کنار گذاشته می‌شود. واریانت
    تک‌واژه‌ای که کلیدش با خود واژه یکی است no-op است و نمی‌رود؛ واریانت
    چندواژه‌ای با کلید چسبیده‌ی یکسان (مثل «می کنم» و «می‌کنم») هر دو
    همان کلید را می‌سازند و بدون تداخل جایگزین می‌شوند — چون تطبیق در
    apply_aliases هر دنباله‌ی مجاور توکن‌ها با کلید یکسان را می‌گیرد.
    """
    alias: dict[str, str] = {}
    for e in entries:
        word = str(e.get("word", "")).strip()
        right = norm_word(word)
        if not right:
            continue
        for v in e.get("variants", []):
            v = str(v).strip()
            key = norm_word(v)
            if not key:
                continue
            if len(v.split()) == 1 and key == right:
                continue  # no-op
            if key in alias and alias[key] != word:
                alias[key] = ""  # مبهم — دائمی حذف
            elif key not in alias:
                alias[key] = word
    return {k: w for k, w in alias.items() if w}


MAX_SPAN = 4  # سقف توکن‌های یک واریانت — شکل شنیده‌ی یک واژه بیشتر نمی‌شود


FUZZY_MIN_TOKEN = 5   # توکن کوتاه‌تر فازی تطبیق نمی‌شود — ریسک خرابی واژه عادی
FUZZY_MIN_KEY = 6     # کلید کوتاه فازی هدف قرار نمی‌گیرد
FUZZY_RATIO = 0.8     # آستانه شباهت (difflib)


def apply_aliases(text: str, alias_map: dict[str, str]) -> str:
    """جایگزینی واریانت‌ها در متن خروجی — تطبیق حریصانه از طولانی‌ترین.

    هر دنباله‌ی مجاور توکن‌ها که کلید نرمال چسبیده‌اش در alias_map باشد
    جایگزین می‌شود؛ پس شکل‌های جدا («ای ای») و چسبیده («ای‌ای») هر دو
    گرفته می‌شوند. اگر تطبیق دقیق نشد، برای توکن‌های بلند تطبیق فازی با
    کلیدهای بلند انجام می‌شود — چون decode پنجره‌ی زنده و decode نهایی
    (چانک ۱۶ ثانیه‌ای) همان واژه را به شکل‌های نزدیک اما متفاوتی می‌شنوند
    (~۳۰٪ توکن در مرزها فرق دارد) و تطبیق دقیقِ تنهایی کافی نیست.
    جایگزین، واژه‌ی درستِ واردشده‌ی کاربر است.
    """
    if not alias_map or not text or not text.strip():
        return text
    tokens = text.split()
    keys = [norm_word(t) for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        hit, hit_span = None, 1
        for span in range(min(MAX_SPAN, len(tokens) - i), 0, -1):
            w = alias_map.get("".join(keys[i:i + span]))
            if w:
                hit, hit_span = w, span
                break
        if hit is None and len(keys[i]) >= FUZZY_MIN_TOKEN:
            w = _fuzzy_hit(keys[i], alias_map)
            if w:
                hit = w  # فازی فقط تک‌توکنی است
        if hit is not None:
            out.append(hit)
            i += hit_span
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _fuzzy_hit(key: str, alias_map: dict[str, str]) -> str | None:
    """نزدیک‌ترین کلید بلند با شباهت ≥ FUZZY_RATIO — وگرنه None."""
    if len(key) < FUZZY_MIN_TOKEN:
        return None
    best, best_r = None, 0.0
    for k, w in alias_map.items():
        if len(k) < FUZZY_MIN_KEY:
            continue
        r = SequenceMatcher(None, key, k).ratio()
        if r > best_r:
            best, best_r = w, r
    return best if best_r >= FUZZY_RATIO else None
