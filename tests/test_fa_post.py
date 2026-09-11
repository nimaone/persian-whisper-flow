"""تست‌های پس‌پردازش متن فارسی — قانون بازپیوند می/نمی."""
from __future__ import annotations

from app.fa_post import rejoin_prefixes


def test_basic_mi_rejoin():
    assert rejoin_prefixes("من می کنم") == "من میکنم"


def test_nemi_rejoin():
    assert rejoin_prefixes("او نمی رود") == "او نمیرود"


def test_multiple_prefixes_in_one_sentence():
    assert rejoin_prefixes("نمی دانم می خواهد برود") == "نمیدانم میخواهد برود"


def test_mi_at_end_of_text_stays_alone():
    # «می» انتهای جمله — واژه‌ی بعدی ندارد، دست نمی‌خورد
    assert rejoin_prefixes("شاید یک می") == "شاید یک می"


def test_no_join_on_punctuation_token():
    assert rejoin_prefixes("من می ، او رفت") == "من می ، او رفت"


def test_joined_form_passes_through_unchanged():
    assert rejoin_prefixes("من میکنم") == "من میکنم"
    assert rejoin_prefixes("من می‌کنم") == "من می‌کنم"


def test_normal_text_untouched():
    assert rejoin_prefixes("این متن عادی است") == "این متن عادی است"


def test_empty_and_whitespace():
    assert rejoin_prefixes("") == ""
    assert rejoin_prefixes("   ") == ""


def test_mi_only():
    assert rejoin_prefixes("می") == "می"
