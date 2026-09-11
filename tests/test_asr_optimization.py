"""تست‌های قرارداد بهینه‌سازی CPU/RAM مسیر زنده."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pytest

import app.asr as asr_module
from app.asr import (
    AsrEngine,
    AsrHypothesis,
    AsrWord,
    LiveTranscriber,
    _word_confidences,
    ctc_confidence,
)


# ---------- confidence / word span ----------

def test_ctc_confidence_uses_nonblank_frames():
    # three normalized log-probability rows; blank id is 0
    log_probs = np.log(np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.1, 0.8],
    ], dtype=np.float64))
    confidence, margin = ctc_confidence(log_probs, blank_id=0)
    assert confidence == pytest.approx((0.5 + 0.8) / 2)
    # row0: top1=0.3، top2=0.1 / row2: top1=0.8، top2=0.1
    assert margin == pytest.approx((0.2 + 0.7) / 2)


def test_ctc_confidence_blank_only_does_not_crash():
    log_probs = np.log(np.full((2, 4), 0.25, dtype=np.float64))
    confidence, margin = ctc_confidence(log_probs, blank_id=0)
    assert confidence == pytest.approx(0.25)
    assert margin == pytest.approx(0.0)


def test_word_confidences_maps_requested_frames():
    blank = 0
    a, b = 1, 2
    log_probs = np.log(np.array([
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90],
    ], dtype=np.float64))
    tokens = [("سلام", (0, 1)), ("همگی", (1, 3))]
    words = _word_confidences(log_probs, tokens, blank_id=blank)
    assert [w.text for w in words] == ["سلام", "همگی"]
    assert words[0].confidence == pytest.approx(0.90)
    assert words[0].margin == pytest.approx(0.85)
    assert words[1].confidence == pytest.approx(0.90)
    assert words[1].margin == pytest.approx(0.85)


# ---------- AsrEngine detail decoder ----------

@dataclass
class DummyHypothesis:
    text: str
    confidence: float = 0.9
    margin: float = 0.8
    words: tuple[AsrWord, ...] = ()


class DummyDetailDecoder:
    instances = []
    calls = []

    def __init__(self, model_dir, num_threads=4, beam_width=4):
        self.model_dir = model_dir
        self.num_threads = num_threads
        self.beam_width = beam_width
        DummyDetailDecoder.instances.append(self)

    def decode(self, samples, sample_rate=16000, previous_text=None):
        DummyDetailDecoder.calls.append((previous_text, samples.size, sample_rate))
        return DummyHypothesis("متن")


class ReleasableDetail:
    released = False

    def release(self):
        self.released = True


def make_engine_without_model():
    engine = AsrEngine.__new__(AsrEngine)
    engine._model_dir = "model"
    engine._num_threads = 4
    engine._detail_lock = threading.Lock()
    engine._detail_decoder = None
    return engine


def test_detail_decoder_is_created_once_with_beam_width_2(monkeypatch):
    DummyDetailDecoder.instances.clear(); DummyDetailDecoder.calls.clear()
    monkeypatch.setattr(asr_module, "_CtcHypothesisDecoder", DummyDetailDecoder)
    engine = make_engine_without_model()

    samples = np.zeros(1600, dtype=np.float32)
    first = engine.transcribe_with_details(samples)
    second = engine.transcribe_with_details(samples, previous_text="متن")

    assert first.text == "متن"
    assert second.text == "متن"
    assert len(DummyDetailDecoder.instances) == 1
    assert DummyDetailDecoder.instances[0].beam_width == 2
    # متن تکراری اجازه دارد detail را رد کند، اما API باید آن را پاس دهد
    assert DummyDetailDecoder.calls == [(None, 1600, 16000), ("متن", 1600, 16000)]


def test_release_detail_decoder_drops_session():
    engine = make_engine_without_model()
    detail = ReleasableDetail()
    engine._detail_decoder = detail
    engine.release_detail_decoder()
    assert engine._detail_decoder is None
    assert detail.released is True

    # آزادسازی تکراری نباید خطا بدهد
    engine.release_detail_decoder()
    assert engine._detail_decoder is None


# ---------- DirectCtcAsrEngine ----------

def test_direct_engine_uses_single_decoder_for_both_paths(monkeypatch):
    DummyDetailDecoder.instances.clear(); DummyDetailDecoder.calls.clear()
    monkeypatch.setattr(asr_module, "_CtcHypothesisDecoder", DummyDetailDecoder)

    engine = asr_module.DirectCtcAsrEngine("model", num_threads=4, beam_width=2)
    samples = np.zeros(1600, dtype=np.float32)
    details = engine.transcribe_with_details(samples, previous_text=None)
    text = engine.transcribe(samples)

    assert len(DummyDetailDecoder.instances) == 1
    assert DummyDetailDecoder.instances[0].beam_width == 2
    assert details.text == "متن"
    assert text == "متن"
    assert DummyDetailDecoder.calls == [(None, 1600, 16000), (None, 1600, 16000)]


def test_direct_engine_release_keeps_primary_decoder(monkeypatch):
    DummyDetailDecoder.instances.clear(); DummyDetailDecoder.calls.clear()
    monkeypatch.setattr(asr_module, "_CtcHypothesisDecoder", DummyDetailDecoder)
    engine = asr_module.DirectCtcAsrEngine("model", beam_width=2)
    engine.release_detail_decoder()
    # در موتور تک‌مدلی، session اصلی نباید آزاد شود
    assert engine._decoder is DummyDetailDecoder.instances[0]


def test_direct_engine_chunks_long_final_transcription(monkeypatch):
    calls = []

    class ChunkSpyDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def decode(self, samples, sample_rate=16000, previous_text=None):
            calls.append(samples.size)
            return DummyHypothesis(f"part-{samples.size}")

    monkeypatch.setattr(asr_module, "_CtcHypothesisDecoder", ChunkSpyDecoder)
    engine = asr_module.DirectCtcAsrEngine("model", beam_width=2)
    long_audio = np.zeros(int(17.5 * 16000), dtype=np.float32)
    text = engine.transcribe(long_audio)

    chunk = int(asr_module.FINAL_CHUNK_SEC * 16000)
    assert calls == [chunk, long_audio.size - chunk]
    assert text == f"part-{chunk} part-{long_audio.size - chunk}"


def test_live_transcriber_keeps_previous_result_on_identical_text():
    class SameTextEngine:
        def __init__(self):
            self.previous_texts = []

        def transcribe(self, samples, sample_rate=16000):
            return "متن ثابت"

        def transcribe_with_details(
            self, samples, sample_rate=16000, previous_text=None
        ):
            self.previous_texts.append(previous_text)
            return AsrHypothesis("متن ثابت")

    engine = SameTextEngine()
    live = LiveTranscriber(engine, stable_live=True)
    buf = np.zeros(1600, dtype=np.float32)

    first = live.partial_result(buf)
    second = live.partial_result(buf)

    assert first.text == "متن ثابت"
    assert second is first
    assert engine.previous_texts == [None, "متن ثابت"]


def test_app_engine_selection_and_fallback(monkeypatch):
    from app import main as main_module
    from app.config import Config

    calls = {}
    app = main_module.App()
    app.cfg = Config({})

    def fake_plain(**kwargs):
        calls["plain"] = kwargs
        return "plain-engine"

    def fake_direct(model_dir, num_threads=4, beam_width=2):
        calls["direct"] = {"model_dir": model_dir, "num_threads": num_threads,
                           "beam_width": beam_width}
        return "direct-engine"

    def fake_hotword(**kwargs):
        calls["hotword"] = kwargs
        return "hotword-engine"

    monkeypatch.setattr(main_module, "load_engine", fake_plain)
    monkeypatch.setattr(main_module, "DirectCtcAsrEngine", fake_direct)
    monkeypatch.setattr("app.hotword_asr.load_hotword_engine", fake_hotword)

    app.cfg.data["stable_live"] = True
    assert app._make_engine() == "direct-engine"
    assert calls["direct"]["beam_width"] == 2

    app.cfg.data["stable_live"] = False
    assert app._make_engine() == "plain-engine"

    app.cfg.data["hotword_boost"] = True
    app.cfg.data["hotwords"] = ["اپلیکیشن"]
    assert app._make_engine() == "hotword-engine"

    def broken_direct(*args, **kwargs):
        raise RuntimeError("boom")

    app.cfg.data["hotword_boost"] = False
    app.cfg.data["stable_live"] = True
    monkeypatch.setattr(main_module, "DirectCtcAsrEngine", broken_direct)
    assert app._make_engine() == "plain-engine"


class RebuildSpyEngine:
    def __init__(self, name="old"):
        self.name = name
        self.released = False

    def release_detail_decoder(self):
        self.released = True


class DummyConfig:
    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls):
        return cls({})

    def get(self, key):
        return self.data.get(key)


def make_rebuild_app(engine_name="old", stable_live=False):
    from app import main as main_module

    app = main_module.App()
    app.cfg = DummyConfig({"input_device": 0, "stable_live": stable_live})
    app.state = main_module.STATE_IDLE
    app.engine = RebuildSpyEngine(engine_name)
    app.live = object()
    app.apply_hotkey = lambda: None
    app._ui_set_state = lambda state: None
    return app


def test_apply_config_defers_engine_rebuild(monkeypatch):
    from app import main as main_module

    app = make_rebuild_app(stable_live=False)
    new_cfg = DummyConfig({"input_device": 0, "stable_live": True})
    monkeypatch.setattr(main_module.Config, "load", classmethod(lambda cls: new_cfg))
    monkeypatch.setattr(main_module, "set_autostart", lambda enabled: None)

    app.apply_config()

    assert app._engine_rebuild_requested is True
    assert app.engine.name == "old"


def test_rebuild_engine_releases_old_engine():
    app = make_rebuild_app(engine_name="old", stable_live=True)
    old_engine = app.engine
    new_engine = RebuildSpyEngine("new")
    app._make_engine = lambda: new_engine

    app._rebuild_engine()

    assert app.engine is new_engine
    assert old_engine.released is True
    assert app._engine_dirty is False


def test_rebuild_engine_failure_keeps_previous_engine(monkeypatch):
    app = make_rebuild_app(engine_name="old", stable_live=True)

    def broken_make_engine():
        raise RuntimeError("model failed")

    app._make_engine = broken_make_engine
    app._rebuild_engine()

    assert app.engine.name == "old"
    assert app.engine.released is False


# ---------- LiveTranscriber lifecycle ----------

class DetailAwareEngine:
    def __init__(self):
        self.detail_calls = 0
        self.released = False

    def transcribe(self, samples, sample_rate=16000):
        return "plain"

    def transcribe_with_details(self, samples, sample_rate=16000, previous_text=None):
        self.detail_calls += 1
        return AsrHypothesis("stable")

    def release_detail_decoder(self):
        self.released = True


def test_live_transcriber_configure_off_releases_detail_decoder():
    engine = DetailAwareEngine()
    live = LiveTranscriber(engine, stable_live=True)
    live.configure(False)
    assert live.stable_live is False
    assert engine.released is True


def test_live_transcriber_passes_previous_text_only_in_stable_mode():
    engine = DetailAwareEngine()
    live = LiveTranscriber(engine, stable_live=False)
    buf = np.zeros(1600, dtype=np.float32)
    assert live.partial(buf) == "plain"
    assert engine.detail_calls == 0

    live.configure(True)
    assert live.partial(buf) == "stable"
    live.reset()
    assert live.partial(buf) == "stable"
    # بعد از reset پیشوند پایدار خالی است
    assert engine.detail_calls == 2
