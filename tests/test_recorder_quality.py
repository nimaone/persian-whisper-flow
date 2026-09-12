"""تست‌های کیفیت ورودی و تشخیص خودکار میکروفون."""
from __future__ import annotations

import numpy as np
import pytest

from app import recorder
from app.recorder import detect_best_device, input_quality


# ---------- input_quality ----------

def test_quality_empty_shows_hint():
    text, level = input_quality([])
    assert level == "none" and "تست" in text


def test_quality_no_signal_is_bad():
    text, level = input_quality([0.0001] * 20)
    assert level == "bad" and "نمی‌آید" in text


def test_quality_good_when_snr_high():
    vals = [0.0002] * 30 + [0.05] * 10   # کف نویز خیلی زیر اوج
    text, level = input_quality(vals)
    assert level == "good" and "خوب" in text
    assert "SNR" in text


def test_quality_bad_when_noise_near_speech_level():
    vals = [0.03] * 30 + [0.04] * 10     # نویز تقریباً هم‌سطح صدا
    text, level = input_quality(vals)
    assert level == "bad" and "ضعیف" in text


def test_quality_warn_in_between():
    vals = [0.004] * 30 + [0.04] * 10    # SNR = 20dB → مرز خوب
    _, level = input_quality(vals)
    assert level in ("good", "warn")


# ---------- detect_best_device با sd شبیه‌سازی‌شده ----------

class _FakeStream:
    """استریم فیک — دستگاه شکسته/داده‌ی خراب را شبیه‌سازی می‌کند."""

    def __init__(self, broken=False, level=0.0, clip=False):
        self._broken = broken
        self._level = level
        self._clip = clip

    def __enter__(self):
        if self._broken:
            raise RuntimeError("Unanticipated host error")
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        v = 1.5 if self._clip else self._level
        data = np.full((frames, 1), v, dtype=np.float32)
        return data, "none"


@pytest.fixture
def fake_sd(monkeypatch):
    class FakeSd:
        devices = {}          # index -> dict(name, rate, broken, level, clip)
        default = {"device": [None, None]}

        def query_devices(self, device=None, kind=None):
            if device is None:
                return [dict(index=i, max_input_channels=1,
                             default_samplerate=info["rate"],
                             name=info["name"])
                        for i, info in self.devices.items()]
            info = self.devices[device]
            return {"name": info["name"], "default_samplerate": info["rate"],
                    "max_input_channels": 1, "hostapi": 0}

        def InputStream(self, device, **kwargs):
            info = self.devices[device]
            return _FakeStream(broken=info["broken"], level=info["level"],
                               clip=info.get("clip", False))

    fake = FakeSd()
    monkeypatch.setattr(recorder, "sd", fake)
    return fake


def test_detect_skips_broken_device_and_picks_working_loudest(fake_sd):
    fake_sd.devices = {0: dict(name="broken-ghost", rate=44100, broken=True),
                       1: dict(name="good-loud", rate=44100, broken=False, level=0.03),
                       2: dict(name="quiet", rate=44100, broken=False, level=0.0001)}
    assert detect_best_device() == 1


def test_detect_returns_none_when_all_silent(fake_sd):
    fake_sd.devices = {0: dict(name="quiet-a", rate=44100, broken=False, level=0.00001),
                       1: dict(name="quiet-b", rate=44100, broken=False, level=0.00002)}
    assert detect_best_device() is None


def test_detect_skips_garbage_clipped_data(fake_sd):
    # دستگاهی که باز می‌شود ولی دامنه‌اش فراتر از ۱٫۰ است = داده‌ی خراب
    fake_sd.devices = {0: dict(name="clipped-ghost", rate=44100,
                               broken=False, level=0.0, clip=True),
                       1: dict(name="good", rate=44100, broken=False, level=0.02)}
    assert detect_best_device() == 1
