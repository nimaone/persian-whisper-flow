"""تست آیکون سینی اپ فلت — با pystray جعلی، بدون ترد/پیام‌لوپ واقعی ویندوز."""
from __future__ import annotations

import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import APP_TITLE
from flet_ui.tray import TITLES, Tray


# ---------- pystray جعلی ----------

class FakeMenuItem:
    def __init__(self, text, action, default=False):
        self.text = text
        self.action = action
        self.default = default


class FakeMenu(list):
    def __init__(self, *items):
        super().__init__(items)


class FakeIcon:
    def __init__(self, name, image, title, menu):
        self.name = name
        self.image = image
        self.title = title
        self.menu = menu
        self.notifications = []
        self.stopped = False
        self.ran = threading.Event()

    def run(self):
        self.ran.set()

    def notify(self, msg, title=None):
        self.notifications.append((msg, title))

    def stop(self):
        self.stopped = True


def fake_pystray():
    return SimpleNamespace(Menu=FakeMenu, MenuItem=FakeMenuItem, Icon=FakeIcon)


class TrayBase(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.tray = Tray(
            on_show=lambda: self.calls.append("show"),
            on_toggle=lambda: self.calls.append("toggle"),
            on_settings=lambda: self.calls.append("settings"),
            on_quit=lambda: self.calls.append("quit"),
            pystray_mod=fake_pystray(),
            image=object(),  # بدون PIL
        )

    def _item(self, text):
        return next(i for i in self.tray.build().menu if i.text == text)


class TrayMenu(TrayBase):
    def test_menu_items_match_ctk(self):
        menu = self.tray.build().menu
        self.assertEqual([i.text for i in menu],
                         ["نمایش پنجره", "شروع/توقف ضبط", "تنظیمات", "خروج"])
        self.assertTrue(menu[0].default, "کلیک دوبار = نمایش پنجره")
        self.assertFalse(any(i.default for i in menu[1:]))

    def test_callbacks_dispatch(self):
        for text, expected in (("نمایش پنجره", "show"), ("شروع/توقف ضبط", "toggle"),
                               ("تنظیمات", "settings"), ("خروج", "quit")):
            self._item(text).action()
        self.assertEqual(self.calls, ["show", "toggle", "settings", "quit"])

    def test_callback_exception_is_swallowed(self):
        tray = Tray(on_show=lambda: None, on_toggle=lambda: 1 / 0,
                    on_settings=lambda: None, on_quit=lambda: None,
                    pystray_mod=fake_pystray(), image=object())
        menu = tray.build().menu
        item = next(i for i in menu if i.text == "شروع/توقف ضبط")
        item.action()  # نباید استثنا بیرون بزند (ترد pystray می‌خوابد)

    def test_initial_tooltip_is_loading(self):
        self.assertEqual(self.tray.build().title, TITLES["loading"])


class TrayRuntime(TrayBase):
    def test_start_runs_icon_on_a_thread(self):
        self.assertTrue(self.tray.start())
        icon = self.tray._icon
        self.assertTrue(icon.ran.wait(2), "run() باید صدا زده شود")
        self.assertTrue(self.tray.started)
        self.assertEqual(icon.name, "WhisperFlowFarsi")

    def test_set_state_updates_tooltip(self):
        self.tray.build()
        self.tray.set_state("recording")
        self.assertEqual(self.tray._icon.title, TITLES["recording"])
        self.tray.set_state("idle")
        self.assertEqual(self.tray._icon.title, TITLES["idle"])

    def test_unknown_state_falls_back_to_app_title(self):
        self.tray.build()
        self.tray.set_state("چیز-عجیب")
        self.assertEqual(self.tray._icon.title, APP_TITLE)

    def test_long_title_is_truncated_to_63(self):
        self.tray.build()
        self.tray.set_title("ب" * 200)
        self.assertEqual(len(self.tray._icon.title), 63)

    def test_notify_calls_balloon(self):
        self.tray.build()
        self.tray.notify("میکروفون باز نشد")
        self.assertEqual(self.tray._icon.notifications, [("میکروفون باز نشد", APP_TITLE)])

    def test_stop_stops_icon(self):
        self.tray.build()
        self.tray.stop()
        self.assertTrue(self.tray._icon.stopped)
        self.assertFalse(self.tray.started)

    def test_calls_without_icon_are_noops(self):
        # build صدا زده نشده — هیچ‌کدام نباید AttributeError بدهند
        self.tray.set_title("x")
        self.tray.set_state("idle")
        self.tray.notify("x")
        self.tray.stop()


class TrayUnavailable(unittest.TestCase):
    def test_start_returns_false_without_pystray(self):
        tray = Tray(on_show=lambda: None, on_toggle=lambda: None,
                    on_settings=lambda: None, on_quit=lambda: None, image=object())
        with mock.patch.dict(sys.modules, {"pystray": None}):
            self.assertFalse(tray.start())
        self.assertFalse(tray.started)

    def test_start_returns_false_without_image(self):
        tray = Tray(on_show=lambda: None, on_toggle=lambda: None,
                    on_settings=lambda: None, on_quit=lambda: None,
                    pystray_mod=fake_pystray(), image=None)
        with mock.patch("flet_ui.tray.tray_image", lambda: None):
            self.assertFalse(tray.start())

    def test_start_is_idempotent(self):
        tray = Tray(on_show=lambda: None, on_toggle=lambda: None,
                    on_settings=lambda: None, on_quit=lambda: None,
                    pystray_mod=fake_pystray(), image=object())
        self.assertTrue(tray.start())
        with mock.patch.object(tray, "build") as build:
            self.assertTrue(tray.start())
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
