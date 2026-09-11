"""تست‌های سه قانون قفل پیشوند: رقیب (contest)، لبه‌ی پنجره، رأی (votes).

سنجه‌ها از آزمایش ۳۵۰ کلیپ (spikes/speech_rate/report_multi.json):
  - contest<0.3: خطای واژه‌های قفل‌شده ۳۴٪→۲۸٪
  - واژه‌ی نزدیک لبه‌ی قدیمی: خطا ~۲ برابر میانه (ایده‌ی «واژه‌ی داخلی»)
  - votes≥1: نوسان پنجره را می‌گیرد (ولی نرم‌تر از contest)
"""
from __future__ import annotations

import numpy as np

from app.asr import AsrHypothesis, AsrWord, LiveTranscriber

SR = 16000
WIN_SEC = 10.0
MS_FRAME = 80.0 / 1000.0


class ListEngine:
    """موتور ساختگی: هر فراخوانی فرضیه‌ی بعدی لیست را برمی‌گرداند."""

    def __init__(self, hypotheses):
        self.hypotheses = hypotheses
        self.calls = 0

    def transcribe_with_details(self, samples, sample_rate=16000, previous_text=None):
        hyp = self.hypotheses[min(self.calls, len(self.hypotheses) - 1)]
        self.calls += 1
        return hyp

    def transcribe(self, samples, sample_rate=16000):
        return self.transcribe_with_details(samples, sample_rate).text


def timed_hyp(entries, confidence=0.9, margin=0.6):
    """ساخت فرضیه با واژه‌های زمان‌دار.

    entries: [(text, start_frame)] — end = شروع فریم واژه‌ی بعدی یا +3
    """
    words = []
    for i, (text, f0) in enumerate(entries):
        f1 = entries[i + 1][1] if i + 1 < len(entries) else f0 + 3
        words.append(AsrWord(text, confidence, margin, start=f0, end=max(f1, f0 + 1)))
    return AsrHypothesis(
        text=" ".join(e[0] for e in entries),
        confidence=confidence, margin=margin,
        words=tuple(words),
    )


def buf(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def test_word_memory_votes_and_contest():
    from app.asr import _WordMemory
    mem = _WordMemory()
    # دو پنجره‌ی هم‌پوشان: یکی «اپلیکیشن» و یکی «اپل شن» در همان بازه
    mem.record(0.0, [AsrWord("اپلیکیشن", 0.9, 0.8, start=10, end=30)], 10.0)
    mem.record(0.8, [AsrWord("اپل شن", 0.7, 0.5, start=0, end=28)], 10.8)
    # بازه‌ی زمانی مطلق واژه: 0.0 + 10×80ms = 0.8s تا 0.0 + 30×80ms = 2.4s
    votes = mem.count_votes("اپلیکیشن", 0.8, 2.4)
    contest = mem.contest("اپلیکیشن", 0.8, 2.4)
    assert votes == 1  # فقط خود مشاهده‌ی پنجره‌ی اول
    assert contest > 0.4  # «اپل شن» رقیب هم‌زمان است
    # واژه‌ی بدون رقیب
    assert mem.contest("واژه‌ی تنها", 5.0, 6.0) == 0.0
    mem.reset()
    assert mem.observations == []
    # هرس مشاهده‌های قدیمی‌تر از پنجره‌ی فعلی
    mem.record(0.0, [AsrWord("کهنه", 0.9, 0.8, start=5, end=6)], 10.0)
    mem.record(50.0, [AsrWord("تازه", 0.9, 0.8, start=0, end=5)], 60.0)
    # پنجره‌ی 60s: هرس از 60-12=48s — «کهنه» (0.4s) باید حذف شود
    assert all(o[2] >= 48.0 for o in mem.observations)
    assert any(o[0] == "تازه" for o in mem.observations)


def test_contested_word_is_not_locked():
    """واژه‌ای که رقیب هم‌زمان دارد وارد متن قفل‌شده نمی‌شود."""
    # پنجره ۱ (t=0..10s): «سلام» در ثانیه‌ی ۲ گفته می‌شود؛ پنجره ۲ همان
    # واژه را در همان زمان می‌بیند + رقیبِ «سلامی» — پس «سلام» قابل قفل است
    # ولی «سلامی» (رقیب زیاد) در پنجره‌ی ۳ نباید قفل شود.
    e = ListEngine([
        timed_hyp([("سلام", 25)]),
        timed_hyp([("سلام", 25), ("دنیا", 60)], confidence=0.9),
        # «سلامی» همان جای «سلام» را می‌گیرد و رقیب دارد
        timed_hyp([("سلام", 25), ("سلامی", 60)]),
        timed_hyp([("سلام", 25), ("سلامی", 60)]),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                          max_contest=0.30)
    # بافر رشد می‌کند: 10s → 10.8 → 11.6 (پنجره‌ها هم‌پوشان)
    live.partial_result(buf(10.0))
    live.partial_result(buf(10.8))
    shown = live.partial_result(buf(11.6))
    # «دنیا» پذیرفته شده (رأی + بدون رقیب)؛ «سلامی» بعد از حذف «دنیا»
    # نباید بیاید چون پنجره‌ی قبلی در همان جای زمانی «دنیا» دیده
    assert shown.text == "سلام دنیا", shown.text


def test_edge_word_waits_until_inside_window():
    """واژه‌ی چسبیده به لبه‌ی قدیمی پنجره قفل نمی‌شود تا داخلی شود.

    بافر 10.8s → پنجره‌ی 0.8..10.8 (کامل). «سلام» روی فریم 2 (0.16s از
    شروع پنجره) → ناحیه‌ی لبه؛ «دنیا» روی فریم 30 (2.4s) → داخلی.
    اولین قفل فقط contest و conf می‌سنجد (رأی هنوز وجود ندارد) پس متن
    شروع می‌شود، اما بعد از آن «سلام» در پنجره‌های بعدی (که هنوز لبه
    است) اجازه‌ی قفل ندارد — اینجا «سلام» رد و «دنیا» پذیرفته می‌شود.
    """
    e = ListEngine([
        timed_hyp([("سلام", 2), ("دنیا", 30)]),
        timed_hyp([("سلام", 2), ("دنیا", 30)]),
        timed_hyp([("سلام", 2), ("دنیا", 30)]),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                          lock_edge_sec=1.0, vote_high_conf=0.95)
    first = live.partial_result(buf(10.8))
    # اولین قفل: بدون حافظه‌ی رأی — هر دو واژه پذیرفته می‌شوند
    # (contest=0، conf=0.9) تا متن زنده شروع شود
    assert first.text == "سلام دنیا", first.text
    # پنجره‌ی دوم: همان متن → متن تکراری، نمایش ثابت
    second = live.partial_result(buf(11.6))
    assert second.text == "سلام دنیا", second.text


def test_edge_word_blocks_locking_in_continuation():
    """واژه‌ی لبه در مسیر «ادامه‌ی متن» قفل نمی‌شود.

    متن «سلام» قفل شده؛ پنجره‌ی بعدی «سلام دنیا» می‌دهد که «دنیا»
    روی لبه‌ی قدیمی نشسته → دنباله رد می‌شود و متن قدیمی می‌ماند.
    """
    e = ListEngine([
        timed_hyp([("سلام", 30)]),
        # «دنیا» در پنجره‌ی 0.8..10.8 روی فریم 2 (0.16s) → لبه
        timed_hyp([("سلام", 30), ("دنیا", 2)]),
        timed_hyp([("سلام", 30), ("دنیا", 2)]),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                          lock_edge_sec=1.0, vote_high_conf=0.95)
    first = live.partial_result(buf(10.8))
    assert first.text == "سلام", first.text
    second = live.partial_result(buf(11.6))
    # «دنیا» در لبه‌ی قدیمی → قفل نمی‌شود؛ حتی با تکرار (hysteresis)
    # سوپاپ _tail_unlocked هم همان قانون را می‌سنجد
    assert second.text == "سلام", second.text


def test_unseen_word_needs_high_confidence():
    """واژه‌ی بدون رأی پنجره‌ی قبلی فقط با اطمینان بالا قفل می‌شود."""
    # پنجره ۱: «سلام»؛ پنجره ۲: «سلام دنیا» که «دنیا» هرگز دیده نشده و
    # conf کم دارد → قفل نمی‌شود؛ اگر conf خیلی بالا باشد → قفل می‌شود.
    e = ListEngine([
        timed_hyp([("سلام", 25)], confidence=0.9),
        timed_hyp([("سلام", 25), ("دنیا", 60)], confidence=0.5),
        timed_hyp([("سلام", 25), ("دنیا", 60)], confidence=0.5),
        timed_hyp([("سلام", 25), ("دنیا", 60)], confidence=0.5),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                          vote_high_conf=0.90)
    live.partial_result(buf(10.0))
    live.partial_result(buf(10.8))
    shown = live.partial_result(buf(11.6))
    assert shown.text == "سلام", "واژه‌ی ناشناخته با conf پایین قفل نمی‌شود"

    e2 = ListEngine([
        timed_hyp([("سلام", 25)], confidence=0.9),
        timed_hyp([("سلام", 25), ("دنیا", 60)], confidence=0.99),
    ])
    live2 = LiveTranscriber(e2, window_sec=WIN_SEC, stable_live=True,
                           vote_high_conf=0.90)
    live2.partial_result(buf(10.0))
    shown = live2.partial_result(buf(10.8))
    assert shown.text == "سلام دنیا", "با conf بسیار بالا قفل می‌شود"


def test_words_without_timing_keep_old_behavior():
    """بدون span فریم، قوانین رأی/رقیب/لبه خاموش‌اند و conf حکم می‌کند.

    رفتار قبلی: درج واژه با conf زیر min_word_confidence رد می‌شود؛
    واژه با conf بالای آستانه پذیرفته می‌شود.
    """
    e = ListEngine([
        AsrHypothesis("سلام دنیا", 0.9, 0.6, (
            AsrWord("سلام", 0.9, 0.6), AsrWord("دنیا", 0.9, 0.6))),
        # conf=0.20 < min_word_confidence=0.25 → درج رد می‌شود (رفتار قبلی)
        AsrHypothesis("سلام دنیا جدید", 0.9, 0.6, (
            AsrWord("سلام", 0.9, 0.6), AsrWord("دنیا", 0.9, 0.6),
            AsrWord("جدید", 0.20, 0.3))),
        AsrHypothesis("سلام دنیا جدید", 0.9, 0.6, (
            AsrWord("سلام", 0.9, 0.6), AsrWord("دنیا", 0.9, 0.6),
            AsrWord("جدید", 0.5, 0.3))),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True)
    assert live.partial_result(buf(10.0)).text == "سلام دنیا"
    # واژه‌ی زیر آستانه بدون زمان‌بندی رد می‌شود (رفتار قبلی)
    assert live.partial_result(buf(10.8)).text == "سلام دنیا"
    # سوپاپ hysteresis همان رفتار قبلی را دارد: متن پایدار می‌آید
    assert live.partial_result(buf(11.6)).text == "سلام دنیا جدید"


def test_zwnj_variant_continuation_accepted():
    """واژه با/بدون نیم‌فاصله در تطبیق قفل یکی است؛ ادامه پذیرفته می‌شود.

    «می‌ریم» (با ZWNJ) قفل شده؛ پنجره‌ی بعد «میریم دنیا» (بدون ZWNJ)
    می‌دهد. بدون نرمال‌سازی این replace حساب می‌شد و رد؛ با کلید
    نرمال، ادامه است و واژه‌ی تازه می‌آید. متن نمایش هم خروجی خود
    مدل می‌ماند (ZWNJ پنجره‌ی اول تا جایگزینی حفظ می‌شود).
    """
    e = ListEngine([
        timed_hyp([("سلام", 5), ("می‌ریم", 30)], confidence=0.95),
        timed_hyp([("سلام", 5), ("میریم", 30), ("دنیا", 60)],
                  confidence=0.99),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                           vote_high_conf=0.95)
    first = live.partial_result(buf(10.0))
    assert first.text == "سلام می‌ریم"
    assert "\u200c" in first.text  # متن نمایش دست‌نخورده
    second = live.partial_result(buf(10.8))
    assert second.text == "سلام میریم دنیا", (
        "تفاوت نیم‌فاصله نباید ادامه‌ی متن را رد کند")


def test_zwnj_variant_identical_text_no_flicker():
    """تفاوت فقط-نیم‌فاصله «تغییر متن» حساب نمی‌شود تا نمایش نلرزد.

    پنجره‌ی بعد همان متن را بدون ZWNJ می‌دهد → نمایش قبلی (با ZWNJ)
    باقی می‌ماند؛ نه بازنگری، نه لرزش بین دو شکل یک واژه.
    """
    e = ListEngine([
        timed_hyp([("سلام", 5), ("می‌ریم", 30)], confidence=0.95),
        timed_hyp([("سلام", 5), ("میریم", 30)], confidence=0.95),
        timed_hyp([("سلام", 5), ("می‌ریم", 30)], confidence=0.95),
    ])
    live = LiveTranscriber(e, window_sec=WIN_SEC, stable_live=True,
                           vote_high_conf=0.95)
    first = live.partial_result(buf(10.0))
    assert first.text == "سلام می‌ریم"
    second = live.partial_result(buf(10.8))
    assert second.text == "سلام می‌ریم"
    third = live.partial_result(buf(11.6))
    assert third.text == "سلام می‌ریم"


def test_zwnj_variant_memory_votes():
    """رأی حافظه بین شکل‌های با/بدون نیم‌فاصله‌ی یک واژه مشترک است."""
    from app.asr import _WordMemory
    mem = _WordMemory()
    mem.record(0.0, [AsrWord("می‌ریم", 0.9, 0.6, start=10, end=20)], 10.0)
    # همان بازه‌ی مطلق، بدون ZWNJ → باید رأی بگیرد
    assert mem.count_votes("میریم", 0.8, 1.6) == 1
    assert mem.contest("میریم", 0.8, 1.6) == 0.0
    # رقیب واقعی (متن متفاوت) همچنان رقیب است
    mem.record(0.8, [AsrWord("می ریم", 0.7, 0.5, start=0, end=20)], 10.8)
    # «می ریم» دو واژه است — اینجا به‌صورت یک توکن ساختگی ثبت شد؛
    # کلیدش با «میریم» برابر نیست → رقیب حساب می‌شود
    assert mem.contest("میریم", 0.8, 1.6) > 0.0


def test_zwnj_variant_shift_detected():
    """جابه‌جایی پنجره با واریانت نیم‌فاصله‌دار هم تشخیص داده می‌شود."""
    old = [AsrWord("سلام", 0.9, 0.6), AsrWord("می‌ریم", 0.9, 0.6),
           AsrWord("دنیا", 0.9, 0.6)]
    new = [AsrWord("میریم", 0.9, 0.6), AsrWord("دنیا", 0.9, 0.6),
           AsrWord("خوب", 0.9, 0.6)]
    # بدون نرمال‌سازی تطبیق دقیق شکست می‌خورد و shift=None می‌شد
    assert LiveTranscriber._shift_prefix_len(old, new) == 1
    # و واریانت با ZWNJ هم ولی همچنان تطبیق ندارد (متن واقعاً متفاوت)
    new2 = [AsrWord("می‌خوابم", 0.9, 0.6), AsrWord("دنیا", 0.9, 0.6),
            AsrWord("خوب", 0.9, 0.6)]
    assert LiveTranscriber._shift_prefix_len(old, new2) is None
