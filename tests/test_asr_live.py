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
