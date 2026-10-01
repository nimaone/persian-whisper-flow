"""تست صفحه‌ی «اولین اجرا»ی UI فلت — بدون شبکه و بدون دیالوگ سیستم.

دانلود و انتخاب پوشه تزریق می‌شوند، پس تست‌ها قطعی و آفلاین‌اند؛ فقط
وضعیت ویجت‌ها، کپی فایل‌ها و فراخوانی on_ready بررسی می‌شود.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flet as ft

from app.config import APP_TITLE
from flet_ui import first_run_window as fr
from flet_ui.first_run_window import FirstRunWindow, model_ready, TITLE, BODY
from tests.test_flet_ui import MockPage, _btn_texts, _find_all


def wait_for(pred, timeout=5.0):
    """انتظار برای پایان ترد دانلود در تست."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def write_model(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for n in ("model.onnx", "tokens.txt"):
        (folder / n).write_bytes(b"x" * 32)


def fake_download(ok=True, msg="مدل کامل است", pct_values=(0, 50, 100)):
    """جانشین model_download.download_model — فایل‌ها را می‌سازد و پیشرفت می‌دهد."""
    calls = []

    def fn(md: Path, progress_cb=None):
        calls.append(Path(md))
        for pct in pct_values:
            if progress_cb:
                progress_cb("model.onnx", pct, 100)
        if ok:
            write_model(Path(md))
        return ok, msg

    fn.calls = calls
    return fn


class ModelReady(unittest.TestCase):
    def test_true_when_both_files_exist(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            with mock.patch.object(fr, "model_dir", lambda: Path(d)):
                self.assertTrue(model_ready())

    def test_false_when_a_file_is_missing(self):
        with TemporaryDirectory() as d:
            (Path(d) / "model.onnx").write_bytes(b"x")
            with mock.patch.object(fr, "model_dir", lambda: Path(d)):
                self.assertFalse(model_ready())


class FirstRunBuild(unittest.TestCase):
    def setUp(self):
        self.page = MockPage()
        self.ready = []
        self.quit_calls = []

    def _win(self, **kw):
        self.tmp = TemporaryDirectory()
        kw.setdefault("md", Path(self.tmp.name))
        return FirstRunWindow(self.page, on_ready=lambda: self.ready.append(1),
                              on_quit=lambda: self.quit_calls.append(1), **kw)

    def tearDown(self):
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_panel_contents(self):
        win = self._win(download_fn=fake_download())
        self.assertEqual(self.page.title, APP_TITLE)
        self.assertTrue(self.page.rtl)
        texts = []
        for c in self.page.added:
            texts.extend(t.value for t in _find_all(c, ft.Text))
        self.assertIn(TITLE, texts)
        self.assertIn(BODY, texts)
        self.assertEqual(win.prog.value, 0)
        self.assertEqual(_btn_texts(self.page.added[-1]),
                         ["دانلود خودکار", "انتخاب دستی فایل‌ها…", "خروج"])
        self.assertFalse(win.result)

    def test_quit_button_calls_on_quit(self):
        win = self._win(download_fn=fake_download())
        win._fire_quit()
        self.assertEqual(self.quit_calls, [1])

    # ---------- دانلود خودکار ----------
    def test_download_success_marks_accent_and_readies(self):
        win = self._win(download_fn=fake_download())
        win._fire_download()
        self.assertTrue(wait_for(lambda: win.result), "دانلود باید تمام شود")
        self.assertEqual(win.prog.value, 1)
        self.assertEqual(win.status.color, "#22c55e")   # ACCENT
        self.assertIn("کامل شد", win.status.value)
        self.assertTrue(win.dl_btn.disabled and win.pick_btn.disabled)
        # بدون run_task (MockPage) مستقیم صدا زده می‌شود
        self.assertEqual(self.ready, [1])

    def test_download_failure_reenables_buttons(self):
        win = self._win(download_fn=fake_download(ok=False, msg="دانلود ناموفق"))
        win._fire_download()
        self.assertTrue(wait_for(lambda: not win._busy), "دانلود باید تمام شود")
        self.assertFalse(win.result)
        self.assertEqual(self.ready, [])
        self.assertEqual(win.status.color, "#e5484d")   # DANGER
        self.assertIn("دانلود ناموفق", win.status.value)
        self.assertFalse(win.dl_btn.disabled)
        self.assertFalse(win.pick_btn.disabled)

    def test_download_exception_does_not_crash(self):
        def boom(md, progress_cb=None):
            raise ConnectionError("قطعی شبکه")

        win = self._win(download_fn=boom)
        win._fire_download()
        self.assertTrue(wait_for(lambda: not win._busy))
        self.assertFalse(win.result)
        self.assertIn("قطعی شبکه", win.status.value)

    def test_progress_percent_is_deduped(self):
        win = self._win(download_fn=fake_download())
        win._on_progress("model.onnx", 10, 100)
        self.assertEqual(win.status.value, "model.onnx — 10٪")
        win.status.value = "دست‌نخورده"
        win._on_progress("model.onnx", 10, 100)   # همان درصد → پچ دوباره نه
        self.assertEqual(win.status.value, "دست‌نخورده")
        win._on_progress("model.onnx", 25, 100)
        self.assertEqual(win.status.value, "model.onnx — 25٪")
        self.assertAlmostEqual(win.prog.value, 0.25)

    # ---------- انتخاب دستی ----------
    def test_apply_folder_copies_files(self):
        win = self._win(download_fn=fake_download())
        with TemporaryDirectory() as src:
            write_model(Path(src))
            ok = win._apply_folder(Path(src))
        self.assertTrue(ok)
        self.assertTrue(win.result)
        for n in ("model.onnx", "tokens.txt"):
            self.assertTrue((win.md / n).exists(), f"{n} کپی نشد")
        self.assertEqual(self.ready, [1])

    def test_apply_folder_missing_files_reports_error(self):
        win = self._win(download_fn=fake_download())
        with TemporaryDirectory() as src:
            (Path(src) / "model.onnx").write_bytes(b"x")
            ok = win._apply_folder(Path(src))
        self.assertFalse(ok)
        self.assertFalse(win.result)
        self.assertEqual(win.status.color, "#e5484d")
        self.assertIn("tokens.txt", win.status.value)
        self.assertEqual(self.ready, [])

    def test_pick_folder_applies_selected_dir(self):
        with TemporaryDirectory() as src:
            write_model(Path(src))
            win = self._win(download_fn=fake_download(), pick_dir_fn=lambda: src)
            asyncio.run(win._pick_folder())
        self.assertTrue(win.result)

    def test_pick_folder_cancel_changes_nothing(self):
        win = self._win(download_fn=fake_download(), pick_dir_fn=lambda: None)
        asyncio.run(win._pick_folder())
        self.assertFalse(win.result)
        self.assertEqual(win.status.value, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
