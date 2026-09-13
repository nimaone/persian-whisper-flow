"""تست‌های رفتار متن زنده: قفل پیشوند پایدار + مارجین اطمینان + هیسترزیس."""
from __future__ import annotations

import numpy as np
import pytest

from app.asr import AsrHypothesis, AsrWord, LiveTranscriber


class FakeEngine:
    """موتور ساختگی که ترتیب فرضیه‌ها را از قبل می‌گیرد."""

    def __init__(self, hypotheses: list[AsrHypothesis]):
        self.hypotheses = hypotheses
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        hyp = self.hypotheses[min(self.calls, len(self.hypotheses) - 1)]
        self.calls += 1
        return hyp.text

    def transcribe_with_details(
        self, samples: np.ndarray, sample_rate: int = 16000, previous_text=None
    ) -> AsrHypothesis:
        hyp = self.hypotheses[min(self.calls, len(self.hypotheses) - 1)]
        self.calls += 1
        return hyp


def hyp(text: str, confidence: float = 0.9, margin: float = 0.5) -> AsrHypothesis:
    words = text.split()
    return AsrHypothesis(
        text=text,
        confidence=confidence,
        margin=margin,
        words=tuple(AsrWord(w, confidence, margin) for w in words),
    )


def test_live_extension_is_accepted_without_waiting():
    live = LiveTranscriber(FakeEngine([
        hyp("سلام همگی قراره"),
        hyp("سلام همگی قراره اپلی", confidence=0.92),
    ]), stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)

    assert live.partial_result(buf).text == "سلام همگی قراره"
    assert live.partial_result(buf).text == "سلام همگی قراره اپلی"


def test_live_weak_revision_is_locked():
    live = LiveTranscriber(FakeEngine([
        hyp("سلام همگی قراره اپلی", confidence=0.90, margin=0.70),
        hyp("سلام همگی قراره اپفی", confidence=0.88, margin=0.45),
        hyp("سلام همگی قراره اپلی", confidence=0.91, margin=0.72),
    ]), stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)

    assert live.partial_result(buf).text == "سلام همگی قراره اپلی"
    assert live.partial_result(buf).text == "سلام همگی قراره اپلی"
    assert live.partial_result(buf).text == "سلام همگی قراره اپلی"


def test_live_confident_revision_replaces_locked_prefix():
    live = LiveTranscriber(FakeEngine([
        hyp("این اپ اسمش ای سیار نام داره", confidence=0.82, margin=0.35),
        # همان واژه‌ی مشکل‌دار، ولی با اطمینان و مارجین روشن‌تر
        hyp("این اپ اسمش دیکت یار نام داره", confidence=0.94, margin=0.80),
    ]), stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)

    assert live.partial_result(buf).text == "این اپ اسمش ای سیار نام داره"
    assert live.partial_result(buf).text == "این اپ اسمش دیکت یار نام داره"


def test_live_hysteresis_accepts_repeated_revision_without_big_margin():
    live = LiveTranscriber(FakeEngine([
        hyp("این اپ اسمش ای سیار نام داره", confidence=0.86, margin=0.40),
        hyp("این اپ اسمش دیکت یار نام داره", confidence=0.87, margin=0.46),
        hyp("این اپ اسمش دیکت یار نام داره", confidence=0.87, margin=0.46),
    ]), stable_live=True, hysteresis=2)
    buf = np.zeros(1600, dtype=np.float32)

    assert live.partial_result(buf).text == "این اپ اسمش ای سیار نام داره"
    assert live.partial_result(buf).text == "این اپ اسمش ای سیار نام داره"
    assert live.partial_result(buf).text == "این اپ اسمش دیکت یار نام داره"


def test_live_reset_clears_prefix_between_recordings():
    live = LiveTranscriber(FakeEngine([
        hyp("جمله‌ی ضبط قبلی"),
        hyp("شروع تازه"),
    ]), stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)
    assert live.partial_result(buf).text == "جمله‌ی ضبط قبلی"
    live.reset()
    assert live.partial_result(buf).text == "شروع تازه"


def test_default_live_uses_plain_path_and_does_not_load_detail_decoder():
    class DetailEngine(FakeEngine):
        def transcribe_with_details(
            self, samples: np.ndarray, sample_rate: int = 16000
        ) -> AsrHypothesis:
            raise AssertionError("مسیر پیش‌فرض نباید امتیازدهی را لود کند")

    engine = DetailEngine([hyp("سلام همگی", confidence=0.9, margin=0.5)])
    live = LiveTranscriber(engine)
    assert live.stable_live is False
    assert live.partial(np.zeros(1600, dtype=np.float32)) == "سلام همگی"


def test_live_partial_falls_back_to_plain_engine():
    class PlainEngine:
        def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
            return "متن ساده"

    live = LiveTranscriber(PlainEngine())
    assert live.partial(np.zeros(1600, dtype=np.float32)) == "متن ساده"


def test_shift_prefix_len_is_fuzzy_on_noisy_scroll():
    """اسکرول پنجره با ۱-۲ واژه‌ی بدشنیده باید هم‌چنان shift را پیدا کند.

    روی صدای نویزدار، decode ناحیه‌ی هم‌پوشان دقیقاً یکسان نمی‌آید؛
    تطبیق دقیق هرگز نمی‌خورد و نمایش فریز می‌شد (آزمایش ۱۹).
    """
    live = LiveTranscriber(FakeEngine([]), stable_live=True)
    old = [AsrWord(w, 0.9, 0.5) for w in "الف ب پ ت ث ج چ".split()]
    new = [AsrWord(w, 0.9, 0.5) for w in "پ ت س ج چ ح خ".split()]
    assert live._shift_prefix_len(old, new) == 2


def test_scroll_with_noisy_decode_updates_display():
    """بعد از اسکرول پنجره، کاندیدِ بدشنیده‌ی هم‌محتوا باید پذیرفته شود."""
    live = LiveTranscriber(FakeEngine([
        hyp("الف ب پ ت ث ج چ ح"),
        hyp("ب پ ت ث ج چ ح خ"),   # قبل از اسکرول — حذف ابتدای نمایش نیست
        hyp("پ ت س ج چ ح خ ز", confidence=0.8),
    ]), stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)
    assert live.partial_result(buf, t_offset=11.0).text == "الف ب پ ت ث ج چ ح"
    # هنوز اسکرولی رخ نداده — کاندید کوتاه‌تر نباید نمایش را کوچک کند
    assert live.partial_result(buf, t_offset=12.0).text == "الف ب پ ت ث ج چ ح"
    # اسکرول واقعی (buf_end=13.5 ≥ 12.8) — همان محتوا با واژه‌ی بدشنیده
    out = live.partial_result(buf, t_offset=13.0)
    assert out.text == "پ ت س ج چ ح خ ز"


def test_first_lock_failure_still_seeds_display_after_hysteresis():
    """اگر قفل اولین فرضیه به‌خاطر واژه‌ی بحث‌برانگیز شکست بخورد، سوپاپ
    hysteresis باید نمایش را پس از دو تیک پایدار بسازد — وگرنه نمایش
    برای همیشه خالی می‌ماند (آزمایش ۱۹: kooshiar-live1)."""
    from app.asr import _cmp_key

    timed = AsrHypothesis(
        text="سلام سلام",
        confidence=0.9,
        margin=0.5,
        words=(
            AsrWord("سلام", 0.9, 0.5, start=10, end=15),
            AsrWord("سلام", 0.9, 0.5, start=16, end=21),
        ),
    )
    live = LiveTranscriber(FakeEngine([timed]), stable_live=True)
    # حافظه پر از رقیبِ هم‌بازه — قفل اولین فرضیه را رد می‌کند
    live._memory.observations.append((_cmp_key("دیگر"), 0.8, 2.2))
    buf = np.zeros(1600, dtype=np.float32)
    assert live.partial_result(buf, t_offset=0.0).text == ""   # تیک اول: رد
    assert live.partial_result(buf, t_offset=1.0).text == "سلام سلام"  # سوپاپ
