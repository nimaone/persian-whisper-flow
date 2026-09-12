"""تست‌های ثبت صوتی واژه — برداشت واریانت، نگاشت alias و اعمال روی خروجی."""
from __future__ import annotations

import numpy as np

from app import enroll


# ---------- norm_word ----------

def test_norm_word_strips_zwnj_diacritics_and_arabic_letters():
    assert enroll.norm_word("می\u200cکنم") == enroll.norm_word("میکنم")
    assert enroll.norm_word("اٰن") == "ان"
    assert enroll.norm_word("يك") == "یک"
    assert enroll.norm_word("سلام،") == "سلام"


# ---------- EnrollStore ----------

def test_store_roundtrip_and_add_update(tmp_path, monkeypatch):
    path = tmp_path / "enrollments.json"
    monkeypatch.setattr(enroll, "store_path", lambda: path)
    store = enroll.EnrollStore()
    store.add_entry("ویسپرفلو", ["ویسپر فلو", "ویس پر فلو"])
    store.add_entry("ای‌آی", ["ای ای"])
    store.save()

    loaded = enroll.EnrollStore.load()
    assert [e["word"] for e in loaded.entries] == ["ویسپرفلو", "ای‌آی"]
    assert loaded.entries[0]["variants"] == ["ویسپر فلو", "ویس پر فلو"]
    assert all(e["enabled"] for e in loaded.entries)

    # افزودن همان واژه (کلید نرمال یکی) = به‌روزرسانی، نه تکرار
    store.add_entry("ویسپرفلو", ["ویسپر فلومو"])
    assert len(store.entries) == 2
    store.remove_entry("ای‌آی")
    assert [e["word"] for e in store.entries] == ["ویسپرفلو"]


def test_store_corrupt_file_is_tolerated(tmp_path, monkeypatch):
    path = tmp_path / "enrollments.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(enroll, "store_path", lambda: path)
    assert enroll.EnrollStore.load().entries == []


# ---------- harvest_variants ----------

class FakeEngine:
    def __init__(self, text: str):
        self.text = text

    def transcribe(self, samples, sample_rate: int = 16000) -> str:
        return self.text


def test_harvest_heard_correct_returns_empty():
    eng = FakeEngine("ویسپرفلو")
    out = enroll.harvest_variants(eng, np.zeros(32000, dtype=np.float32), "ویسپرفلو")
    assert out == []


def test_harvest_split_heard_form_is_whole_phrase():
    # مدل «ویسپرفلو» را دو‌تکه شنید — واریانت باید کل عبارت باشد، نه تکه‌ها
    eng = FakeEngine("ویسپر فلو")
    out = enroll.harvest_variants(eng, np.zeros(32000, dtype=np.float32), "ویسپرفلو")
    assert out == ["ویسپر فلو"]


def test_harvest_drops_prefixes_and_short_noise_tokens():
    eng = FakeEngine("می ویسپر فلو و")
    out = enroll.harvest_variants(eng, np.zeros(32000, dtype=np.float32), "ویسپرفلو")
    assert out == ["ویسپر فلو"]


def test_harvest_wrong_single_word_is_variant():
    eng = FakeEngine("کلاود")
    out = enroll.harvest_variants(eng, np.zeros(32000, dtype=np.float32), "کلود")
    assert out == ["کلاود"]


def test_harvest_empty_samples_returns_empty():
    assert enroll.harvest_variants(FakeEngine("هرچی"), np.zeros(0, dtype=np.float32), "x") == []


# ---------- build_alias_map ----------

def test_alias_map_variants():
    entries = [
        {"word": "ویسپرفلو", "variants": ["ویسپر فلو", "ویس پر فلو"], "enabled": True},
        {"word": "ای‌آی", "variants": ["ای ای", "آی"], "enabled": True},
    ]
    amap = enroll.build_alias_map(entries)
    # هر دو شکل چندواژه‌ای به یک کلید چسبیده می‌رسند
    assert amap[enroll.norm_word("ویسپر فلو")] == "ویسپرفلو"
    assert amap[enroll.norm_word("ویس پر فلو")] == "ویسپرفلو"
    assert amap[enroll.norm_word("ای ای")] == "ای‌آی"
    assert amap[enroll.norm_word("آی")] == "ای‌آی"


def test_alias_map_noop_single_variant_skipped():
    # واریانت تک‌واژه‌ای که کلیدش با خود واژه یکی است هیچ تغییری نمی‌دهد
    amap = enroll.build_alias_map([
        {"word": "سلام", "variants": ["سلام"], "enabled": True},
    ])
    assert amap == {}


def test_alias_map_split_form_of_word_is_kept():
    # مدل «میکنم» را «می کنم» می‌شنود — چندواژه‌ای با همان کلید چسبیده
    amap = enroll.build_alias_map([
        {"word": "میکنم", "variants": ["می کنم"], "enabled": True},
    ])
    assert amap[enroll.norm_word("می کنم")] == "میکنم"


def test_alias_map_conflicting_variant_is_dropped():
    entries = [
        {"word": "دیکته‌یار", "variants": ["دکت یار"], "enabled": True},
        {"word": "دکتر یار", "variants": ["دکت یار"], "enabled": True},
    ]
    amap = enroll.build_alias_map(entries)
    assert enroll.norm_word("دکت یار") not in amap


def test_alias_map_disabled_entries_are_skipped_by_active():
    store = enroll.EnrollStore([
        {"word": "روشن", "variants": ["روشون"], "enabled": True},
        {"word": "خاموش", "variants": ["خامش"], "enabled": False},
    ])
    amap = enroll.build_alias_map(store.active())
    assert enroll.norm_word("روشون") in amap
    assert enroll.norm_word("خامش") not in amap


# ---------- apply_aliases ----------

def test_apply_single_token_alias():
    amap = {enroll.norm_word("کلاود"): "کلاه‌ور"}
    assert enroll.apply_aliases("این کلاود خوب است", amap) == "این کلاه‌ور خوب است"


def test_apply_joined_single_token_form_of_multi_variant():
    # شکل چسبیده با نیم‌فاصله («ای‌ای») هم باید گرفته شود
    amap = enroll.build_alias_map([
        {"word": "ای‌آی", "variants": ["ای ای"], "enabled": True},
    ])
    assert enroll.apply_aliases("میگم ای\u200cای چیست", amap) == "میگم ای‌آی چیست"


def test_apply_multi_token_span_alias():
    amap = enroll.build_alias_map([
        {"word": "ویسپرفلو", "variants": ["ویسپر فلو"], "enabled": True},
    ])
    assert enroll.apply_aliases("سلام ویسپر فلو جان", amap) == "سلام ویسپرفلو جان"


def test_apply_is_zwnj_and_diacritic_insensitive():
    amap = enroll.build_alias_map([
        {"word": "ای‌آی", "variants": ["ای ای"], "enabled": True},
    ])
    assert enroll.apply_aliases("میگم ای\u200cای چیست", amap) == "میگم ای‌آی چیست"


def test_apply_longest_span_wins():
    amap = {
        enroll.norm_word("ویسپر"): "کلمه اول",
        enroll.norm_word("ویسپر فلو"): "ویسپرفلو",
    }
    assert enroll.apply_aliases("ویسپر فلو", amap) == "ویسپرفلو"
    assert enroll.apply_aliases("فقط ویسپر", amap) == "فقط کلمه اول"


def test_apply_no_map_or_empty_text_is_identity():
    assert enroll.apply_aliases("متن", {}) == "متن"
    assert enroll.apply_aliases("", {"a": "b"}) == ""
    assert enroll.apply_aliases("متن", {"a": "b"}) == "متن"


def test_phrase_variant_does_not_fire_on_lone_fragment():
    # تکه‌ی «فلو» تنها نباید به واژه‌ی کامل تبدیل شود — فقط عبارت کامل
    amap = enroll.build_alias_map([
        {"word": "ویسپرفلو", "variants": ["ویسپر فلو"], "enabled": True},
    ])
    assert enroll.apply_aliases("فلو را بگو", amap) == "فلو را بگو"
    assert enroll.apply_aliases("ویسپر فلو را بگو", amap) == "ویسپرفلو را بگو"


def test_apply_split_word_from_rejoin_style():
    amap = enroll.build_alias_map([
        {"word": "نمیدانم", "variants": ["نمیدانم ها"], "enabled": True},
    ])
    assert amap, "واریانت باید نگه داشته شود"
    # تطبیق دو توکنیِ مجاور
    out = enroll.apply_aliases("من هم نمیدانم ها دیگر", amap)
    assert "نمیدانم ها" not in out
