"""تست‌های ITN فارسی — تبدیل عدد حروفی به رقم.

اجرا:
    .venv/Scripts/python.exe -m pytest tests/test_persian_itn.py -v
یا بدون pytest:
    .venv/Scripts/python.exe -m tests.test_persian_itn
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import persian_itn
from app.persian_itn import convert, normalize_text, words_to_number


class TestSingleWords(unittest.TestCase):
    def test_units(self):
        for word, val in [
            ("صفر", 0), ("یک", 1), ("دو", 2), ("سه", 3), ("چهار", 4),
            ("پنج", 5), ("شش", 6), ("هفت", 7), ("هشت", 8), ("نه", 9),
        ]:
            self.assertEqual(convert(word), val, word)

    def test_teens(self):
        self.assertEqual(convert("ده"), 10)
        self.assertEqual(convert("یازده"), 11)
        self.assertEqual(convert("دوازده"), 12)
        self.assertEqual(convert("پانزده"), 15)
        self.assertEqual(convert("پونزده"), 15)  # تلفظ محاوره‌ای
        self.assertEqual(convert("شانزده"), 16)
        self.assertEqual(convert("شونزده"), 16)
        self.assertEqual(convert("نوزده"), 19)

    def test_tens(self):
        self.assertEqual(convert("بیست"), 20)
        self.assertEqual(convert("سی"), 30)
        self.assertEqual(convert("چهل"), 40)
        self.assertEqual(convert("پنجاه"), 50)
        self.assertEqual(convert("شصت"), 60)
        self.assertEqual(convert("نود"), 90)

    def test_hundreds(self):
        self.assertEqual(convert("صد"), 100)
        self.assertEqual(convert("یکصد"), 100)
        self.assertEqual(convert("دویست"), 200)
        self.assertEqual(convert("سیصد"), 300)
        self.assertEqual(convert("پانصد"), 500)
        self.assertEqual(convert("نهصد"), 900)


class TestCompound(unittest.TestCase):
    def test_ten_units(self):
        self.assertEqual(convert("بیست و سه"), 23)
        self.assertEqual(convert("سی و پنج"), 35)
        self.assertEqual(convert("چهل و هفت"), 47)
        self.assertEqual(convert("نود و نه"), 99)

    def test_hundred_tens(self):
        self.assertEqual(convert("صد و بیست و سه"), 123)
        self.assertEqual(convert("دویست و سی و چهار"), 234)
        self.assertEqual(convert("نهصد و نود و نه"), 999)

    def test_colloquial_variants(self):
        # شکل‌های محاوره‌ای که ASR واقعاً می‌نویسد (برگرفته از واژگان sherpa-onnx)
        self.assertEqual(convert("شیشصد"), 600)
        self.assertEqual(convert("پونصد"), 500)
        self.assertEqual(convert("شیش"), 6)
        self.assertEqual(convert("هیفده"), 17)

    def test_thousands(self):
        self.assertEqual(convert("هزار"), 1000)
        self.assertEqual(convert("دو هزار"), 2000)
        self.assertEqual(convert("هزار و دویست"), 1200)
        self.assertEqual(convert("هزار و سیصد و بیست و سه"), 1323)
        self.assertEqual(convert("سه هزار و چهارصد و پنجاه و شش"), 3456)
        self.assertEqual(convert("نه هزار و نهصد و نود و نه"), 9999)

    def test_millions(self):
        self.assertEqual(convert("یک میلیون"), 1_000_000)
        self.assertEqual(convert("یک میلیون و دویست هزار"), 1_200_000)
        self.assertEqual(convert("دو میلیون و سیصد و چهل و پنج هزار و ششصد و هفتاد و هشت"), 2_345_678)

    def test_billions(self):
        self.assertEqual(convert("یک میلیارد و دویست میلیون"), 1_200_000_000)
        self.assertEqual(convert("دو میلیارد و پانصد میلیون و سیصد هزار"), 2_500_300_000)

    def test_trillion(self):
        self.assertEqual(convert("یک تریلیون"), 1_000_000_000_000)


class TestDecimal(unittest.TestCase):
    def test_momayez(self):
        self.assertEqual(convert("سه ممیز چهارده"), 3.14)
        self.assertEqual(convert("یک ممیز پنج"), 1.5)
        self.assertEqual(convert("صفر ممیز پنج"), 0.5)

    def test_decimal_digits(self):
        self.assertAlmostEqual(convert("دو ممیز صفر پنج"), 2.05)


class TestSentenceLevel(unittest.TestCase):
    """سطح جمله — همان چیزی که در خروجی ASR استفاده می‌شود."""

    def test_simple_sentence(self):
        self.assertEqual(
            normalize_text("من بیست و سه سال دارم"),
            "من ۲۳ سال دارم",
        )

    def test_money(self):
        self.assertEqual(
            normalize_text("یک میلیون و دویست هزار تومان"),
            "۱٬۲۰۰٬۰۰۰ تومان",
        )

    def test_decimal_sentence(self):
        self.assertEqual(
            normalize_text("نرخ رشد سه ممیز چهارده درصد بود"),
            "نرخ رشد ۳٫۱۴ درصد بود",
        )

    def test_multiple_numbers(self):
        self.assertEqual(
            normalize_text("بیست و سه نفر آمدند و سی نفر رفتند"),
            "۲۳ نفر آمدند و ۳۰ نفر رفتند",
        )

    def test_non_numbers_untouched(self):
        self.assertEqual(normalize_text("سلام دنیا"), "سلام دنیا")

    def test_ordinal_not_converted(self):
        # «چهارم» رتبه است؛ نباید به ۴ تبدیل شود
        out = normalize_text("بار چهارم بود")
        self.assertNotIn("۴", out)

    def test_no_false_positive_on_plain_words(self):
        # کلمهٔ عادی که توکن عددی نیست دست‌نخورده می‌ماند
        self.assertEqual(normalize_text("کتاب را خواندم"), "کتاب را خواندم")

    def test_wavand_between_number_and_word_preserved(self):
        # رگرسیون باگ نسخهٔ sherpa-onnx: «و» حذف نمی‌شود
        # (نسخهٔ اصلی: «هشت و نیم» → «۸ نیم»)
        self.assertEqual(
            normalize_text("صبح ساعت هشت و نیم رسید"),
            "صبح ساعت ۸ و نیم رسید",
        )


class TestNormalization(unittest.TestCase):
    def test_arabic_ye_kaf(self):
        # ي/ك عربی در خروجی بعضی مدل‌ها؛ باید نرمال و تبدیل شوند
        self.assertEqual(convert("يكصد و بيست"), 120)
        self.assertEqual(normalize_text("يك دو سه"), "۱ ۲ ۳")

    def test_partial_parse_returns_leading_number(self):
        # convert اولین عدد را می‌گیرد؛ دنبالهٔ نامشخص بی‌توجه می‌ماند
        self.assertEqual(convert("بیست و aproximadamente"), 20)

    def test_half_space_variants(self):
        self.assertEqual(convert("سی\u200cو\u200cپنج"), 35)

    def test_none_on_garbage(self):
        self.assertIsNone(convert("سلام"))
        self.assertIsNone(convert(""))
        self.assertIsNone(words_to_number("تست بدون عدد"))


class TestCompoundOnly(unittest.TestCase):
    """حالت min_tokens=2 — همان چیزی که سوییچ تنظیمات اپ استفاده می‌کند."""

    def test_compound_converted(self):
        self.assertEqual(
            persian_itn.normalize_text("من بیست و سه سال دارم", min_tokens=2),
            "من ۲۳ سال دارم",
        )

    def test_single_small_number_untouched(self):
        self.assertEqual(
            persian_itn.normalize_text("یک متن بنویس", min_tokens=2),
            "یک متن بنویس",
        )
        self.assertEqual(
            persian_itn.normalize_text("ساعت هشت رسید", min_tokens=2),
            "ساعت هشت رسید",
        )

    def test_scale_words_alone_converted(self):
        # «هزار/میلیون» تکی مبهم نیستند و تبدیل می‌شوند
        self.assertEqual(
            persian_itn.normalize_text("هزار تومان دادم", min_tokens=2),
            "۱٬۰۰۰ تومان دادم",
        )
        self.assertEqual(
            persian_itn.normalize_text("دو میلیون نسخه فروخت", min_tokens=2),
            "۲٬۰۰۰٬۰۰۰ نسخه فروخت",
        )

    def test_zwnj_preserved_in_surrounding_text(self):
        # رگرسیون: نیم‌فاصلهٔ متن اطراف نباید حذف شود
        self.assertEqual(
            persian_itn.normalize_text("من می‌روم و سی‌وپنج سال دارم", min_tokens=2),
            "من می‌روم و ۳۵ سال دارم",
        )

    def test_zwnj_attached_word_not_converted(self):
        # «سه‌تایی» عدد کامل نیست؛ نباید به «۳‌تایی» تبدیل شود
        self.assertEqual(
            persian_itn.normalize_text("سه‌تایی آمدند", min_tokens=2),
            "سه‌تایی آمدند",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
