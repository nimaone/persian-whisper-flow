"""تست سمت CTk تنظیم «پوشه مدل» — همان قابلیت نسخه فلت، در اپ اصلی.

بدون ساخت پنجره سنجیده می‌شود:
- model_dir_status: متن/رنگ وضعیت در سه حالت (کامل، دلخواه خالی، پیش‌فرض)
- ensure_model (اول اجرای CTk): پوشه دلخواه از تنظیمات خوانده می‌شود و
  دانلود/کپی و دیالوگ اول اجرا روی همان پوشه انجام می‌شود
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import first_run, settings_ui, theme
from app.config import model_dir


class CtkModelDirStatus(unittest.TestCase):
    """ردیف وضعیت کارت — متن و رنگ هر حالت."""

    def test_complete_is_accent(self):
        txt, color = settings_ui.model_dir_status(None, True)
        self.assertEqual(txt, "مدل کامل ✓")
        self.assertEqual(color, theme.ACCENT)

    def test_custom_missing_is_warn(self):
        txt, color = settings_ui.model_dir_status("D:\\models\\shenava", False)
        self.assertEqual(color, theme.WARN)
        self.assertIn("model.onnx یا tokens.txt نیست", txt)
        self.assertIn("دانلود در همین پوشه", txt)

    def test_default_missing_is_dim(self):
        txt, color = settings_ui.model_dir_status(None, False)
        self.assertEqual(color, theme.FG_DIM)
        self.assertIn("پوشه کنار برنامه", txt)

    def test_empty_string_counts_as_default(self):
        # رشته خالی مثل None است — پیش‌فرض
        txt, color = settings_ui.model_dir_status("", False)
        self.assertEqual(color, theme.FG_DIM)


class FirstRunUsesCustomDir(unittest.TestCase):
    """اول اجرای CTk باید پوشه دلخواه تنظیمات را بخواند."""

    def test_complete_in_custom_dir(self):
        with TemporaryDirectory() as td:
            d = Path(td) / "models"
            d.mkdir()
            with mock.patch("app.config.Config.load",
                            staticmethod(lambda: {"model_dir": str(d)})):
                with mock.patch("app.model_download.is_complete",
                                return_value=True) as comp:
                    self.assertTrue(first_run.ensure_model())
            self.assertEqual(comp.call_args[0][0], d)

    def test_source_run_without_model_returns_false(self):
        with TemporaryDirectory() as td:
            d = Path(td) / "models"
            d.mkdir()
            with mock.patch("app.config.Config.load",
                            staticmethod(lambda: {"model_dir": str(d)})):
                with mock.patch("app.model_download.is_complete",
                                return_value=False):
                    with mock.patch.object(sys, "frozen", False, create=True):
                        self.assertFalse(first_run.ensure_model())

    def test_frozen_dialog_gets_custom_dir(self):
        with TemporaryDirectory() as td:
            d = Path(td) / "models"
            d.mkdir()
            with mock.patch("app.config.Config.load",
                            staticmethod(lambda: {"model_dir": str(d)})):
                with mock.patch("app.model_download.is_complete",
                                return_value=False):
                    with mock.patch.object(sys, "frozen", True, create=True):
                        with mock.patch.object(first_run, "_dialog",
                                               return_value=True) as dlg:
                            self.assertTrue(first_run.ensure_model())
            self.assertEqual(dlg.call_args[0][0], d)

    def test_no_setting_falls_back_to_default(self):
        default = model_dir()
        with mock.patch("app.config.Config.load", staticmethod(lambda: {})):
            with mock.patch("app.model_download.is_complete",
                            return_value=True) as comp:
                self.assertTrue(first_run.ensure_model())
        self.assertEqual(comp.call_args[0][0], default)


if __name__ == "__main__":
    unittest.main()
