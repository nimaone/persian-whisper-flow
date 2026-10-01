"""تست‌های gating سکوت برای حلقه‌ی ترنسکرایپ زنده."""
from __future__ import annotations

import pytest

from app.asr import SpeechGate


def test_speech_gate_skips_initial_silence():
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.0, now=0.0) is False
    assert gate.should_decode(rms=0.001, now=0.8) is False


def test_speech_gate_decodes_while_speech_is_active():
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.010, now=0.0) is True
    assert gate.should_decode(rms=0.010, now=0.8) is True


def test_speech_gate_keeps_hangover_after_last_speech():
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.010, now=0.0) is True
    # تا ۱.۵ ثانیه پس از آخرین صدای واقعی decode ادامه دارد
    assert gate.should_decode(rms=0.0, now=0.8) is True
    assert gate.should_decode(rms=0.0, now=1.4) is True
    assert gate.should_decode(rms=0.0, now=1.6) is False


def test_speech_gate_resumes_on_new_speech_after_pause():
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.010, now=0.0) is True
    assert gate.should_decode(rms=0.0, now=2.0) is False
    assert gate.should_decode(rms=0.010, now=2.4) is True


def test_speech_gate_reset_clears_last_speech():
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.010, now=0.0) is True
    gate.reset()
    assert gate.should_decode(rms=0.0, now=0.1) is False


def test_speech_gate_uses_wall_clock_when_now_is_omitted(monkeypatch):
    times = iter([10.0, 10.1])
    monkeypatch.setattr("app.asr.time.perf_counter", lambda: next(times))
    gate = SpeechGate(threshold=0.003, hangover=1.5)
    assert gate.should_decode(rms=0.010) is True
    assert gate.should_decode(rms=0.010) is True
