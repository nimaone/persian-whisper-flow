"""ITN فارسی — تبدیل اعداد حروفی گفتاری به رقم (Inverse Text Normalization).

خروجی خام مدل‌های ASR فارسی اعداد را حروفی می‌نویسد («بیست و سه»).
این ماژول آن‌ها را به رقم تبدیل می‌کند («۲۳») تا محدودیت شناخته‌شدهٔ
README (بخش محدودیت‌ها) رفع شود.

واژگان پایه (و شکل‌های محاوره‌ای شیش/پونصد/شیشصد/هیفده) برگرفته از
python-api-examples/persian_itn.py در مخزن k2-fsa/sherpa-onnx (Apache-2.0)؛
پارسر بازنویسی شده تا «و» میان عدد و واژهٔ بعدی حذف نشود، اعشار
(«ممیز»)، نیم‌فاصله، حروف عربی و جداکنندهٔ هزارگان پشتیبانی شود.

قواعد:
  • اعداد صحیح تا تریلیون با مقیاس‌های «هزار/میلیون/میلیارد/تریلیون»
  • اعشاری با «ممیز» («سه ممیز چهارده» → ۳٫۱۴)
  • پیوند «و» بین جزءها؛ نیم‌فاصله («سی‌وپنج») هم تطبیق می‌شود
  • حروف عربی ي/ك به فارسی تبدیل می‌شوند قبل از تطبیق
  • نیم‌فاصلهٔ بقیهٔ متن حفظ می‌شود («می‌روم» دست‌نخورده می‌ماند)

استفاده:
    from app import persian_itn
    persian_itn.normalize_text("من بیست و سه سال دارم")   # → من ۲۳ سال دارم
    persian_itn.normalize_text("من بیست و سه سال دارم", min_tokens=2)
    persian_itn.convert("یک میلیون و دویست هزار")          # → 1200000
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ————— واژگان —————

ONES = {
    "صفر": 0, "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
    "شش": 6, "شیش": 6, "هفت": 7, "هشت": 8, "نه": 9,
    "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13, "چهارده": 14,
    "پانزده": 15, "پونزده": 15, "شانزده": 16, "شونزده": 16,
    "هفده": 17, "هیفده": 17, "هجده": 18, "هیجده": 18, "نوزده": 19,
}

TENS = {
    "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50,
    "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
}

HUNDREDS = {
    "صد": 100, "یکصد": 100, "دویست": 200, "سیصد": 300,
    "چهارصد": 400, "پانصد": 500, "پونصد": 500,
    "ششصد": 600, "شیشصد": 600, "هفتصد": 700,
    "هشتصد": 800, "نهصد": 900,
}

SCALES = {
    "هزار": 1_000,
    "میلیون": 1_000_000,
    "میلیارد": 1_000_000_000,
    "بیلیون": 1_000_000_000_000,
    "تریلیون": 1_000_000_000_000,
}

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
FA_THOUSANDS_SEP = "٬"   # U+066C
FA_DECIMAL_SEP = "٫"     # U+066B


def _to_fa_digits(s: str) -> str:
    return s.translate(str.maketrans("0123456789", FA_DIGITS))


def _normalize(text: str) -> str:
    """نرمال‌سازی متن ورودی: ی/ک عربی → فارسی، حذف اعراب، فاصله‌ی یکدست."""
    text = text.translate(str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه"}))
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    # نیم‌فاصله و علائم جهت‌دهی → فاصله‌ی ساده برای توکن‌سازی
    text = text.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ————— پارسر عدد حروفی —————

@dataclass
class ParseResult:
    value: float | int
    tokens_consumed: int
    had_decimal: bool = False


def _parse_group(tokens: list[str], i: int) -> tuple[int, int] | None:
    """یک گروه صد/ده/یک را از tokens[i] می‌خواند.

    برمی‌گرداند: (مقدار، تعداد توکن مصرف‌شده) یا None اگر توکن اول عضو
    گروه نباشد. ترتیب رایج فارسی: صد → ده → یک، با «و» اختیاری بین جزءها.
    """
    value = 0
    used = 0
    saw_part = False

    def peek(k: int = 0):
        j = i + used + k
        return tokens[j] if j < len(tokens) else None

    def consume_and() -> int:
        """«و» بعدی را می‌بلعد اگر بلافاصله جزء عددی دیگری بیاید."""
        if peek() == "و" and (
            peek(1) in ONES or peek(1) in TENS or peek(1) in HUNDREDS
        ):
            return 1
        return 0

    t = peek()
    if t in HUNDREDS:
        value += HUNDREDS[t]
        used += 1
        used += consume_and()
        saw_part = True

    t = peek()
    if t in TENS:
        value += TENS[t]
        used += 1
        used += consume_and()
        saw_part = True

    # یکان (شامل ۱۰ تا ۱۹) — فقط وقتی جای یکان خالی است
    t = peek()
    if t is not None and t in ONES and value % 10 == 0:
        value += ONES[t]
        used += 1
        saw_part = True

    return (value, used) if saw_part else None


def _parse_number(tokens: list[str], i: int) -> ParseResult | None:
    """عدد حروفی کامل را از tokens[i] می‌خواند (گروه‌های مقیاس‌دار + اعشار)."""
    total = 0
    used = 0
    saw_any = False

    while True:
        g = _parse_group(tokens, i + used)
        gval, gu = g if g is not None else (0, 0)
        j = i + used + gu

        if j < len(tokens) and tokens[j] in SCALES:
            if g is None and not saw_any:
                gval = 1  # «هزار» به‌تنهایی = ۱×هزار
            elif g is None:
                break
            total += gval * SCALES[tokens[j]]
            used += gu + 1
            saw_any = True
        elif g is not None:
            total += gval
            used += gu
            saw_any = True
        else:
            break

        # ادامه با «و» تا جزء عددی بعدی
        j = i + used
        if j < len(tokens) and tokens[j] == "و":
            nxt = tokens[j + 1] if j + 1 < len(tokens) else None
            if nxt is not None and (
                nxt in ONES or nxt in TENS or nxt in HUNDREDS or nxt in SCALES
            ):
                used += 1
                continue
        break

    if not saw_any:
        return None

    # اعشار: «ممیز» + رقم‌به‌رقم
    had_decimal = False
    if i + used < len(tokens) and tokens[i + used] == "ممیز":
        frac: list[str] = []
        j = i + used + 1
        while j < len(tokens) and tokens[j] in ONES and len(frac) < 6:
            frac.append(tokens[j])
            j += 1
        if frac:
            total = float(f"{int(total)}.{ ''.join(str(ONES[t]) for t in frac) }")
            used += 1 + len(frac)
            had_decimal = True

    value = total
    if isinstance(value, float) and value.is_integer() and not had_decimal:
        value = int(value)
    return ParseResult(value, used, had_decimal)


def words_to_number(text: str) -> float | int | None:
    """اولین عدد حروفی داخل text را به عدد برمی‌گرداند؛ None اگر هیچ‌نباشد."""
    tokens = _normalize(text).split()
    for i in range(len(tokens)):
        r = _parse_number(tokens, i)
        if r is not None:
            return r.value
    return None


def convert(text: str) -> float | int | None:
    """عدد حروفی (کل رشته یا شروعِ آن) → عدد؛ None اگر قابل تبدیل نباشد."""
    return words_to_number(text)


# ————— نرمال‌سازی متن کامل (ITN روی کل جمله) —————

_ZWNJ = "\u200c"
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def _probe_token(tok: str) -> list[str]:
    """زیرتوکن‌های قابل تطبیق یک توکن خام (اعراب حذف، نیم‌فاصله → فاصله).

    فقط برای تشخیص عدد به‌کار می‌رود؛ توکن خام هرگز با نسخهٔ پروب
    جایگزین نمی‌شود مگر اینکه کامل به عدد تبدیل شود.
    پیوند «و» چسبیده («سی‌وپنج») به «سی / و / پنج» می‌شکند.
    """
    subs: list[str] = []
    for part in _DIACRITICS_RE.sub("", tok).replace(_ZWNJ, " ").split():
        if part.startswith("و") and len(part) > 1:
            subs.append("و")
            subs.append(part[1:])
        else:
            subs.append(part)
    return subs


def _format_number(num: float | int, had_decimal: bool, fa_digits: bool) -> str:
    if isinstance(num, int) and abs(num) >= 1_000:
        s = f"{num:,}"
    else:
        s = str(num)
    if had_decimal:
        s = s.replace(".", FA_DECIMAL_SEP if fa_digits else ".")
    if fa_digits:
        s = s.replace(",", FA_THOUSANDS_SEP)
        s = _to_fa_digits(s)
    return s


def normalize_text(text: str, fa_digits: bool = True, min_tokens: int = 1) -> str:
    """عددهای حروفی داخل جمله را به رقم تبدیل می‌کند.

    توکن‌های غیرعددی عیناً حفظ می‌شوند — نیم‌فاصله و اعراب متن دست
    نمی‌خورد («می‌روم» می‌ماند «می‌روم»).

    با min_tokens=2 فقط اعداد چندجزئی («بیست و سه»، «سی‌وپنج»، «دو
    میلیون») و مقیاس‌های تکیِ بزرگ («هزار»، «میلیون») تبدیل می‌شوند؛
    اعداد تکی کوچک («یک»، «هشت»، «صد») حروفی می‌مانند.

    «من بیست و سه سال دارم» → «من ۲۳ سال دارم»
    «یک میلیون و دویست هزار تومان» → «۱٬۲۰۰٬۰۰۰ تومان»
    """
    text = text.translate(str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه"}))
    raw = text.split()
    if not raw:
        return text.strip()

    # جریان پروب: هر توکن خام به زیرتوکن‌های تطبیق‌پذیر می‌شکند
    probe: list[str] = []
    owner: list[int] = []          # هر زیرتوکن متعلق به کدام توکن خام است
    token_span: dict[int, list[int]] = {}  # توکن خام → [اولین، آخرین] اندیس پروب
    for idx, tok in enumerate(raw):
        for sub in _probe_token(tok):
            token_span.setdefault(idx, [len(probe), len(probe)])[1] = len(probe)
            probe.append(sub)
            owner.append(idx)

    replacement: dict[int, str] = {}  # توکن خام → متن جایگزین ("" = ادغام‌شده)
    i = 0
    while i < len(probe):
        r = _parse_number(probe, i)
        if r is None or (r.tokens_consumed < min_tokens and r.value < 1000):
            i += 1
            continue
        start_tok = owner[i]
        end_tok = owner[i + r.tokens_consumed - 1]
        # تطبیق باید «کل» توکن خام ابتدایی و انتهایی را بپوشاند
        # («سه‌تایی» → سه + تایی پوشش ناقص → دست‌نخورده)
        covers_whole = i == token_span[start_tok][0] and (
            i + r.tokens_consumed - 1 == token_span[end_tok][1]
        )
        if not covers_whole:
            i += 1
            continue
        replacement[start_tok] = _format_number(r.value, r.had_decimal, fa_digits)
        for t in range(start_tok + 1, end_tok + 1):
            replacement[t] = ""
        i += r.tokens_consumed

    out: list[str] = []
    for idx, tok in enumerate(raw):
        if idx in replacement:
            if replacement[idx]:
                out.append(replacement[idx])
        else:
            out.append(tok)
    return " ".join(out)
