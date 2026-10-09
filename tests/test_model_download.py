"""تست‌های دانلود مدل — mock شبکه، بدون اینترنت.

تمرکز روی «دانلود ناقص»: پروکسی/ISP گاهی پاسخ را میان‌راه قطع می‌کند و
تراکنش بی‌خطا تمام به نظر می‌رسد؛ فایل ناقص نباید جای‌گذاری شود.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import model_download  # noqa: E402


class FakeResponse:
    """پاسخ جعلی urllib — محتوای اعلام‌شده در هدر و واقعی جدا از هم."""

    def __init__(self, payload: bytes, content_length: int | None):
        self._payload = payload
        self._pos = 0
        self.headers = {"Content-Length": str(content_length)} \
            if content_length is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int) -> bytes:
        chunk = self._payload[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "model.onnx"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, payload: bytes, content_length: int | None):
        resp = FakeResponse(payload, content_length)
        with mock.patch.object(model_download.urllib.request, "urlopen",
                               return_value=resp):
            return model_download._download(
                "https://example/model.onnx", self.dest)

    def test_complete_response_is_placed(self):
        payload = b"M" * (model_download.EXPECTED_MIN["model.onnx"] + 10)
        ok = self._run(payload, len(payload))
        self.assertTrue(ok)
        self.assertTrue(self.dest.exists())
        self.assertEqual(self.dest.stat().st_size, len(payload))
        # فایل موقت نباید باقی بماند
        self.assertFalse(self.dest.with_name(self.dest.name + ".part").exists())

    def test_truncated_response_with_length_is_rejected(self):
        """Content-Length میگوید ۵۰۰MB ولی پاسخ در ۴MB تمام شد — قطعی ISP."""
        announced = 500_000_000
        payload = b"M" * 4_233_777
        ok = self._run(payload, announced)
        self.assertFalse(ok)
        self.assertFalse(self.dest.exists(), "فایل ناقص نباید جای‌گذاری شود")
        self.assertFalse(self.dest.with_name(self.dest.name + ".part").exists())

    def test_below_floor_without_length_is_rejected(self):
        """بدون Content-Length، کف سایز نگهبان است."""
        ok = self._run(b"M" * 1_000, None)
        self.assertFalse(ok)
        self.assertFalse(self.dest.exists())

    def test_size_floor_matches_real_model(self):
        """کف باید خیلی زیر مدل واقعی (~۴۴۰MB) ولی بالای قطعی‌های کوتاه باشد."""
        floor = model_download.EXPECTED_MIN["model.onnx"]
        self.assertGreater(floor, 10_000_000)
        self.assertLess(floor, 400_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
