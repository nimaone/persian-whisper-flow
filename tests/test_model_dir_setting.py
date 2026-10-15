"""تست تنظیم «پوشه مدل» — انتخاب/تعویض مسیر مدل از تب عمومی.

پوشش:
- app.config.model_dir(custom): مقدار دلخواه مطلق، نادیدهگرفتن نسبی، پیشفرض
- تب عمومی تنظیمات فلت: کادر مسیر، دکمههای مرور/پیشفرض، وضعیت کامل/ناقص،
  رفتوبرگشت ذخیره (_collect/_apply) و هندلر مرور با دیالوگ جعلی
- بکاند فلت: کلید موتور با تغییر پوشه عوض میشود و موتور از پوشه تازه لود میشود
- صفحهی اول اجرا: model_ready با مسیر صریح و با تنظیم دلخواه
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flet as ft

from app import models_catalog
from app.config import DEFAULTS, Config, model_dir
from flet_ui import first_run_window as fr
from flet_ui.backend import DictationApp
from tests.test_flet_ui import MockPage, _btn_texts, _find_all


def write_model(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for n in ("model.onnx", "tokens.txt"):
        (folder / n).write_bytes(b"x" * 32)


class ModelDirConfig(unittest.TestCase):
    def test_default_is_project_model(self):
        d = model_dir()
        self.assertTrue(d.is_absolute())
        self.assertEqual(d.name, "model")

    def test_custom_absolute_wins(self):
        d = model_dir(r"D:\models\persian")
        self.assertEqual(Path(r"D:\models\persian"), d)

    def test_empty_uses_default(self):
        self.assertEqual(model_dir(None), model_dir())
        self.assertEqual(model_dir(""), model_dir())

    def test_relative_ignored(self):
        # مسیر نسبی معنیی ثابتی ندارد (به cwd وابسته است) — پیشفرض میماند
        self.assertEqual(model_dir("model"), model_dir())

    def test_default_key_is_none(self):
        self.assertIn("model_dir", DEFAULTS)
        self.assertIsNone(DEFAULTS["model_dir"])


class SettingsModelDirBase(unittest.TestCase):
    """پایه: SettingsWindow با config قطعی (DEFAULTS) مثل SettingsTestBase."""

    def setUp(self):
        from flet_ui.settings_window import SettingsWindow
        page = MockPage()
        self.page = page

        class Fixed(SettingsWindow):
            def _load(self):
                return dict(DEFAULTS)

        self.win = Fixed(page)
        self.root = self.page.added[0]


class GeneralTabModelCard(SettingsModelDirBase):
    def _general_views(self):
        # تبها داخل Tabs > Column > TabBarView
        return _find_all(self.root, ft.TabBarView)

    def test_model_card_in_general_tab(self):
        views = self._general_views()
        self.assertTrue(views, "TabBarView پیدا نشد")
        general = views[0].controls[0]
        # متنهای کارت مدل باید در تب عمومی باشند
        texts = [t.value for t in _find_all(general, ft.Text) if t.value]
        joined = " ".join(texts)
        self.assertIn("پوشه مدل تشخیص گفتار", joined)
        # کادر مسیر + دو دکمه
        field = _find_all(general, ft.TextField)
        self.assertTrue(any(getattr(f, "read_only", False) for f in field),
                        "کادر فقفف خواندنی مسیر مدل باید باشد")
        btns = _btn_texts(general)
        self.assertIn("مرور…", btns)
        self.assertIn("پیش‌فرض", btns)

    def test_field_shows_active_path_default(self):
        self.assertEqual(self.win.model_path.value, str(model_dir()))
        self.assertIsNone(self.win._model_dir_custom)

    def test_status_complete_on_full_default_folder(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            with mock.patch("flet_ui.settings_window.model_dir",
                            lambda custom=None: Path(d)):
                self.win._model_status_update()
                self.assertEqual(self.win.model_status.value, "مدل کامل ✓")
                self.assertEqual(self.win.model_status.color, "#22c55e")

    def test_status_custom_empty_warns(self):
        with TemporaryDirectory() as d:
            self.win._set_model_dir(d)
            self.assertIn("نیست", self.win.model_status.value)
            self.assertEqual(self.win.model_status.color, "#d9a13c")

    def test_status_default_missing_is_dim(self):
        with TemporaryDirectory() as d:
            with mock.patch("flet_ui.settings_window.model_dir",
                            lambda custom=None: Path(d)):
                self.win._model_status_update()
                self.assertIn("پیش‌فرض", self.win.model_status.value)
                self.assertEqual(self.win.model_status.color, "#9b9b9b")


class SettingsModelDirFlow(SettingsModelDirBase):
    def test_collect_roundtrip_default_none(self):
        got = self.win._collect()
        self.assertIn("model_dir", got)
        self.assertIsNone(got["model_dir"])

    def test_set_custom_is_collected(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            self.win._set_model_dir(d)
            self.assertEqual(self.win.model_path.value, str(Path(d)))
            self.assertEqual(self.win.model_status.value, "مدل کامل ✓")
            self.assertEqual(self.win._collect()["model_dir"], d)

    def test_default_button_clears_custom(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            self.win._set_model_dir(d)
            self.win._model_default_dir()
            self.assertIsNone(self.win._model_dir_custom)
            self.assertEqual(self.win.model_path.value, str(model_dir()))
            self.assertIsNone(self.win._collect()["model_dir"])

    def test_apply_roundtrip_with_custom(self):
        data = dict(DEFAULTS)
        data["model_dir"] = r"D:\models\persian"
        self.win._apply(data)
        self.assertEqual(self.win._model_dir_custom, r"D:\models\persian")
        self.assertEqual(self.win.model_path.value, r"D:\models\persian")
        self.assertEqual(self.win._collect()["model_dir"], r"D:\models\persian")

    def test_reset_returns_default_folder(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            self.win._set_model_dir(d)
            self.win._reset(None)
            self.assertIsNone(self.win._model_dir_custom)
            self.assertEqual(self.win._collect()["model_dir"], None)

    def test_browse_sets_picked_folder(self):
        import asyncio
        with TemporaryDirectory() as d:
            write_model(Path(d))
            picker = mock.AsyncMock(return_value=str(Path(d)))
            self.win._pick_model_dir = picker
            asyncio.run(self.win._browse_model_dir())
            self.assertEqual(self.win._model_dir_custom, str(Path(d)))
            self.assertEqual(picker.await_count, 1)

    def test_browse_cancel_keeps_previous(self):
        import asyncio
        with TemporaryDirectory() as d:
            write_model(Path(d))
            self.win._set_model_dir(d)
            self.win._pick_model_dir = mock.AsyncMock(return_value=None)
            asyncio.run(self.win._browse_model_dir())
            self.assertEqual(self.win._model_dir_custom, str(Path(d)))

    def test_picker_is_attached_to_page_services(self):
        """جعبه‌ی دیالوگ بدون وصلشدن به صفحه خطا میدهد (RuntimeError
        «Control must be added to the page first») — پس باید به
        page.services اضافه و بعد از استفاده برداشته شود."""
        import asyncio
        from flet_ui import settings_window as sw

        win = self.win
        win.page.services = []
        attached = []

        class FakePicker:
            async def get_directory_path(self, dialog_title=None,
                                         initial_directory=None):
                # در زمان فراخوانی باید روی صفحه ثبت شده باشد
                attached.append(self in win.page.services)
                return r"D:\models\persian"

        with mock.patch.object(sw.ft, "FilePicker", FakePicker):
            got = asyncio.run(win._pick_model_dir())
        self.assertEqual(got, r"D:\models\persian")
        self.assertEqual(attached, [True])
        # بعد از بستن دیالوگ از صفحه برداشته شده باشد
        self.assertEqual(win.page.services, [])

    def test_save_writes_model_dir_to_disk(self):
        import asyncio
        from app.config import settings_path
        with TemporaryDirectory() as cfg_dir, TemporaryDirectory() as md:
            write_model(Path(md))
            with mock.patch.dict(os.environ, {"APPDATA": cfg_dir}):
                self.win._set_model_dir(md)
                self.win.on_save = lambda data: None
                asyncio.run(self.win._save())
                loaded = Config.load()
                self.assertEqual(loaded.get("model_dir"), md)


class BackendModelDir(unittest.TestCase):
    def _bare_app(self, **overrides):
        app = object.__new__(DictationApp)
        cfg = Config({**DEFAULTS, **overrides})
        app.cfg = cfg
        return app

    def test_engine_key_changes_with_folder(self):
        app = self._bare_app()
        k1 = app._engine_key
        app.cfg.set("model_dir", r"D:\models\persian")
        k2 = app._engine_key
        self.assertNotEqual(k1, k2)

    def test_engine_loader_uses_custom_folder(self):
        app = self._bare_app(model_dir=r"D:\models\persian")
        seen = {}
        with mock.patch("app.engine_factory.load_engine", side_effect=lambda **kw: seen.update(kw) or object()):
            app._default_engine_loader()
        self.assertEqual(seen.get("model_dir"), Path(r"D:\models\persian"))

    def test_engine_loader_default_folder(self):
        app = self._bare_app()
        seen = {}
        with mock.patch("app.engine_factory.load_engine", side_effect=lambda **kw: seen.update(kw) or object()):
            app._default_engine_loader()
        self.assertEqual(seen.get("model_dir"), model_dir())


class FirstRunWithCustomDir(unittest.TestCase):
    def test_model_ready_uses_explicit_dir(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            # مدل فعال قطعی (shenava) — مستقل از settings.json واقعی ماشین
            with mock.patch.object(fr, "active_model_spec",
                                   models_catalog.default_model):
                self.assertTrue(fr.model_ready(Path(d)))

    def test_model_ready_uses_setting(self):
        with TemporaryDirectory() as d:
            write_model(Path(d))
            cfg = Config({**DEFAULTS, "model_dir": str(Path(d))})
            # active_model_dir مقدار را با Config.load() میخواند (متد کلاس)
            fake = type("C", (), {"load": staticmethod(lambda: cfg)})
            with mock.patch("app.config.Config", fake):
                self.assertTrue(fr.model_ready())

    def test_active_model_dir_reads_setting(self):
        with TemporaryDirectory() as d:
            cfg = Config({**DEFAULTS, "model_dir": str(Path(d))})
            fake = type("C", (), {"load": staticmethod(lambda: cfg)})
            with mock.patch("app.config.Config", fake):
                self.assertEqual(fr.active_model_dir(), Path(d))

    def test_active_model_dir_falls_back_on_error(self):
        def boom():
            raise RuntimeError("no disk")

        fake = type("C", (), {"load": staticmethod(boom)})
        with mock.patch("app.config.Config", fake):
            self.assertEqual(fr.active_model_dir(), model_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
