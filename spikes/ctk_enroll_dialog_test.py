"""تست دیالوگ ثبت واژه — بازتولید و تأیید دو باگ:

  ۱) دیالوگ نامرئی + فریز (grab روی پنجره‌ی مخفی) — اصلاح شد
  ۲) «main thread is not in main loop» — خواندن کمبوی CTk از ترد ضبط —
     اصلاح شد: دستگاه در ترد اصلی حل می‌شود

بدون تعامل: پنجره تنظیمات باز می‌شود، دکمه «+ ثبت واژه جدید» invoke
می‌شود، واژه نوشته و «ضبط ۱» فشار داده می‌شود (با موتور ساختگی) و
نتیجه‌ی ضبط بررسی می‌شود. خروجی سالم: «DONE — no freeze» در آخر.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk

import customtkinter as ctk

from app import settings_ui
from app.config import DEFAULTS

FAKE_TEXT = "ویسپر فلو"  # خروجی موتور ساختگی — واریانت کاندید می‌سازد


class FakeConfig:
    """تنظیمات مستقل از فایل واقعی کاربر — enrolalyz روشن و حالت خودکار."""

    data = {**DEFAULTS, "enroll_alias": True, "input_device": None,
            "input_device_key": None, "stable_live": False}

    @classmethod
    def load(cls):
        return cls()

    def get(self, key):
        return self.data.get(key)


settings_ui.Config = FakeConfig


class FakeEngine:
    def transcribe(self, samples, sample_rate: int = 16000) -> str:
        return FAKE_TEXT


class FakeApp:
    engine = FakeEngine()
    device = None  # «خودکار» — مسیر detect_best_device در ترد کارگر

    def refresh_alias_map(self):
        pass


def find_widget(widget, cls, needle: str):
    for c in widget.winfo_children():
        try:
            txt = c.cget("text")
        except Exception:
            txt = None
        if isinstance(c, cls) and txt and needle in txt:
            return c
        r = find_widget(c, cls, needle)
        if r is not None:
            return r
    return None


def find_entry(widget):
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkEntry):
            return c
        r = find_entry(c)
        if r is not None:
            return r
    return None


def main():
    import tempfile
    from app import enroll as _enroll_mod

    tmp = tempfile.mkdtemp()  # enrollments واقعی کاربر آلوده نشود
    _enroll_mod.store_path = lambda: Path(tmp) / "enrollments.json"

    root = ctk.CTk()
    root.withdraw()

    def run():
        settings_ui.open_settings(root, app=FakeApp())
        root.update()
        win = [w for w in root.winfo_children()
               if isinstance(w, tk.Toplevel)][0]
        win.geometry("560x640+2600+80")  # بیرون از صفحه — تداخل با کاربر نداشته باشد
        btn = find_widget(win, ctk.CTkButton, "ثبت واژه جدید")
        print("دکمه پیدا شد:", btn is not None)
        btn.invoke()
        for _ in range(30):
            root.update()
            time.sleep(0.03)
        dlg = [w for w in win.winfo_children()
               if isinstance(w, tk.Toplevel)][0]
        dlg.geometry("470x430+2600+120")
        print("mapped:", dlg.winfo_ismapped(),
              "| viewable:", dlg.winfo_viewable(),
              "| grab_current == dlg:", dlg.grab_current() is dlg)

        entry = find_entry(dlg)
        entry.insert(0, "ویسپرفلو")

        # سه لیبل وضعیت همان‌هایی که اول با «—» هستند
        status_lbls = [c for c in dlg.winfo_children()
                       if isinstance(c, ctk.CTkLabel)
                       and c.cget("text") == "—"]
        if not status_lbls:
            status_lbls = find_status_labels(dlg)
        print("تعداد لیبل وضعیت:", len(status_lbls))
        rec1 = find_widget(dlg, ctk.CTkButton, "ضبط ۱")
        rec1.invoke()  # شروع ضبط — دکمه باید «توقف» شود
        root.update()
        time.sleep(0.5)
        for _ in range(20):
            root.update()
            time.sleep(0.03)
        print("متن دکمه حین ضبط:", rec1.cget("text"))
        rec1.invoke()  # توقف ضبط — از این‌جا پردازش در ترد کارگر

        status = None
        for _ in range(300):  # حداکثر ~۱۵ ثانیه تا دیکد
            root.update()
            time.sleep(0.03)
            txt = status_lbls[0].cget("text")
            if "شنیده شد" in txt or "خطا" in txt or "کوتاه" in txt:
                status = txt
                break
        print("وضعیت نهایی ضبط ۱:", status)
        ok = status is not None and "main thread is not in main loop" not in status
        print("بدون خطای ترد:", ok, "| شنید موتور ساختگی:",
              status is not None and "شنیده شد" in status)

        # --- مسیر پخش: فشار پخش باید sd.play را اجرا و دکمه را «قطع» کند ---
        import sounddevice as sd
        played = {"n": 0, "peak": 0.0}
        orig_play = sd.play

        def spy_play(data, *a, **kw):
            import numpy as _np
            played["n"] += 1
            played["peak"] = float(_np.abs(_np.asarray(data)).max())
            return orig_play(data, *a, **kw)

        sd.play = spy_play
        pbtn = find_widget(dlg, ctk.CTkButton, "پخش")
        print("دکمه پخش پیدا شد:", pbtn is not None,
              "| state:", pbtn.cget("state") if pbtn else None)
        pbtn.invoke()
        for _ in range(20):
            root.update()
            time.sleep(0.03)
        print("sd.play فراخوانی شد:", played["n"] > 0,
              "| peak بافر پخش:", round(played["peak"], 3),
              "| متن دکمه:", pbtn.cget("text"))
        pbtn.invoke()  # قطع
        for _ in range(10):
            root.update()
            time.sleep(0.03)
        print("بعد از قطع، متن دکمه:", pbtn.cget("text"))
        sd.play = orig_play

        # --- ذخیره مدخل و سپس جریان ویرایش ---
        save_btn = find_widget(dlg, ctk.CTkButton, "ذخیره واژه")
        save_btn.invoke()
        root.update()
        edit_btn = find_widget(win, ctk.CTkButton, "ویرایش")
        print("دکمه ویرایش در لیست:", edit_btn is not None)
        edit_btn.invoke()
        for _ in range(30):
            root.update()
            time.sleep(0.03)
        dlg2 = [w for w in win.winfo_children()
                if isinstance(w, tk.Toplevel)][0]
        dlg2.geometry("470x430+2600+120")
        for _ in range(10):
            root.update()
            time.sleep(0.03)
        print("عنوان دیالوگ ویرایش:", dlg2.title())
        entry2 = find_entry(dlg2)
        print("واژه پیش‌پرشده:", entry2.get())
        checks = []

        def _collect_checks(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkCheckBox):
                    checks.append(c)
                _collect_checks(c)

        _collect_checks(dlg2)
        print("چک‌باکس واریانت‌های موجود:", len(checks),
              "| همه تیک‌خورده:", all(c.get() for c in checks))

        # --- افزودن دستی واریانت ---
        entries2 = []

        def _collect_entries(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkEntry):
                    entries2.append(c)
                _collect_entries(c)

        _collect_entries(dlg2)
        manual_entry = entries2[1] if len(entries2) > 1 else None
        print("ورودی افزودن دستی:", manual_entry is not None)
        manual_entry.insert(0, "تست دستی")
        mbtn = find_widget(dlg2, ctk.CTkButton, "افزودن دستی")
        mbtn.invoke()
        for _ in range(10):
            root.update()
            time.sleep(0.03)
        checks.clear()
        _collect_checks(dlg2)
        texts = []
        for c in checks:
            try:
                texts.append(c.cget("text"))
            except Exception:
                pass
        print("چک‌باکس‌ها بعد از افزودن دستی:", len(checks), texts)
        print("واریانت دستی تیک‌خورده:", "تست دستی" in " ".join(texts)
              and all(c.get() for c in checks))
        dlg2.destroy()
        win.destroy()
        root.destroy()

    def find_status_labels(widget):
        out = []
        for c in widget.winfo_children():
            if isinstance(c, ctk.CTkLabel) and c.cget("text") == "—":
                out.append(c)
            out.extend(find_status_labels(c))
        return out

    root.after(150, run)
    root.mainloop()
    print("DONE — no freeze")


if __name__ == "__main__":
    main()
