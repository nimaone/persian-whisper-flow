"""پنجره تنظیمات — Fluent تیره، تب‌بندی‌شده با کارت‌ها.

چهار تب (راست به چپ): عمومی | میکروفون | درج متن | پیشرفته
هر بخش داخل یک کارت با عنوان و توضیح کوتاه. باید از حلقه UI
(thread اصلی) باز شود.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk

import customtkinter as ctk
import numpy as np

from app import theme, smooth_ctk
from app.config import APP_TITLE, APP_TITLE_FULL, APP_VERSION, DEFAULTS, Config, set_autostart
from app.recorder import (Recorder, dedupe_input_devices, detect_best_device,
                          device_label, device_siblings)
from app.win32 import style_toplevel, smooth_show, disable_min_max
from app.theme import apply_icon

SPECS_BARS = 48
SPECS_H = 20

AUTO_STOP_LABELS = {"خاموش": 0, "۳ ثانیه": 3, "۵ ثانیه": 5, "۱۰ ثانیه": 10}


class MicTester:
    """ضبط کوتاه از دستگاه انتخابی + محاسبه RMS در thread جدا."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()  # ("rms", float) | ("err", str)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, device: int | None):
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(device,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self, device):
        import sounddevice as sd

        try:
            dev = sd.query_devices(device, "input")
            sr = int(dev["default_samplerate"])

            def cb(indata, frames, t, status):
                if self._stop.is_set():
                    raise sd.CallbackStop
                rms = float(np.sqrt((indata[:, 0].astype(np.float64) ** 2).mean()))
                try:
                    self.q.put_nowait(("rms", rms))
                except queue.Full:
                    pass

            with sd.InputStream(
                device=device, channels=1, samplerate=sr,
                dtype="float32", blocksize=int(sr * 0.05),
                callback=cb,
            ):
                while not self._stop.is_set():
                    threading.Event().wait(0.05)
        except Exception as e:
            self.q.put(("err", str(e)[:60]))


def open_settings(parent_root, app=None):
    smooth_ctk.apply()  # رندر نرم سراسری — اگر هنوز فعال نشده
    cfg = Config.load()
    fam = theme.family()

    win = tk.Toplevel(parent_root)
    win.title(f"تنظیمات — {APP_TITLE}")
    win.geometry("560x640")
    win.minsize(520, 560)
    win.attributes("-topmost", True)
    win.grab_set()
    win.configure(bg=theme.BG)
    style_toplevel(win)
    apply_icon(win)
    disable_min_max(win)  # دیالوگ ثابت — دکمه‌های مین/ماکس خاکستری می‌شوند

    # هات‌کی سراسری حین باز بودن تنظیمات تعلیق می‌شود تا فشردن همان ترکیب
    # برای capture، ضبط را شروع نکند
    hotkey_suspended = False
    if app is not None:
        try:
            app.suspend_hotkey()
            hotkey_suspended = True
        except Exception:
            hotkey_suspended = False

    def card(parent, title):
        """کارت با عنوان — inner frame را برمی‌گرداند."""
        c = ctk.CTkFrame(parent, fg_color=theme.SURFACE, corner_radius=10)
        c.pack(fill="x", pady=(7, 0), padx=2)
        inner = ctk.CTkFrame(c, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=12)
        if title:
            ctk.CTkLabel(inner, text=title, font=(fam, 13, "bold"),
                         text_color=theme.FG, anchor="e").pack(fill="x", pady=(0, 8))
        return inner

    def dim(parent, text):
        """متن راهنمای کم‌رنگ — با wraplength ثابت.

        بدون wraplength، CTkLabel تکخطی میماند: متنهای بلند از لبهی کارت
        بریده میشدند و پاراگرافها بهصورت خطهای بریدهی ناهماهنگ دیده
        میشدند (ریشهی «نامرتبی» گزارششده).

        نکته: wraplength عمداً ثابت است، نه bind به <Configure> — حلقهی
        Configure↔re-wrap آزمایشی، پنجره را در Not Responding برده بود.
        عرض لیبل راهنما با پنجرهی ۵۶۰ ثابت ≈ ۴۵۶px است؛ ۴۴۰ حاشیهی امن.
        """
        return ctk.CTkLabel(parent, text=text, font=(fam, 12),
                            text_color=theme.FG_DIM, anchor="e", justify="right",
                            wraplength=440).pack(fill="x", pady=(2, 3))

    # === تب‌ها (ترتیب add = چپ به راست؛ «عمومی» باید راست‌ترین باشد) ===
    tabview = ctk.CTkTabview(
        win, corner_radius=10,
        fg_color=theme.BG,
        segmented_button_fg_color=theme.SURFACE,
        segmented_button_selected_color=theme.SURFACE_3,
        segmented_button_selected_hover_color=theme.SURFACE_3,
        segmented_button_unselected_color=theme.BG,
        segmented_button_unselected_hover_color=theme.SURFACE_2,
        text_color=theme.FG,
        segmented_button_font=(fam, 13),
    )
    # نوار دکمه‌های ثابت — باید «قبل از» tabview بسته شود تا نوار پایین
    # اول صاحب جای خودش شود؛ در غیر این صورت ارتفاع درخواستی tabview که با
    # هر تب عوض می‌شود جای دکمه‌ها را جابه‌جا می‌کند (تب میکروفون بلندتر است).
    btn_bar = tk.Frame(win, bg=theme.BG, padx=18, pady=8)
    btn_bar.pack(fill="x", side="bottom")
    tabview.pack(fill="both", expand=True, padx=18, pady=(14, 10))
    tab = tabview.add
    for name in ("راهنما", "پیشرفته", "درج متن", "میکروفون", "عمومی"):
        tab(name)
        tabview.tab(name).configure(fg_color="transparent")
    tabview.set("عمومی")

    check_style = dict(font=(fam, 13), text_color=theme.FG,
                       fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                       border_color=theme.SURFACE_3,
                       checkmark_color=theme.ON_ACCENT)
    switch_style = dict(font=(fam, 13), text_color=theme.FG,
                        fg_color=theme.SURFACE_3,   # ریل خاموش
                        progress_color=theme.ACCENT,  # ریل روشن = سبز
                        button_color=theme.FG, button_hover_color="#ffffff",
                        border_width=0)  # دایره هم‌اندازه‌ی ریل

    def switch_row(parent, text, variable, pady=(0, 0), command=None):
        """ردیف سوییچ: متن راست، کلید آن/آف در لبه‌ی چپ کارت — همه‌ی
        سوییچ‌ها همتراز (قرینه‌ی t.switch در نسخه‌ی فلت).

        خروجی: (سوییچ، برچسب) — برچسب برای خاکستری‌شدنِ همزمان با
        غیرفعال‌شدن سوییچ (مثل «اصلاح واژه‌های ثبت‌شده» در حالت پایدار).
        """
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=pady)
        lbl = ctk.CTkLabel(row, text=text, font=(fam, 13),
                           text_color=theme.FG, anchor="e")
        lbl.pack(side="right", fill="x", expand=True)
        sw = ctk.CTkSwitch(row, text="", variable=variable, command=command,
                           **switch_style)
        sw.pack(side="left")
        return sw, lbl
    radio_style = dict(font=(fam, 13), text_color=theme.FG,
                       fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                       border_color=theme.SURFACE_3)
    menu_style = dict(font=(fam, 13), dropdown_font=(fam, 13),
                      fg_color=theme.SURFACE_2, button_color=theme.SURFACE_3,
                      button_hover_color=theme.SURFACE_3, text_color=theme.FG,
                      dropdown_fg_color=theme.SURFACE_2,
                      dropdown_hover_color=theme.SURFACE_3,
                      dropdown_text_color=theme.FG)

    # ================= تب عمومی =================
    t_general = tabview.tab("عمومی")

    # --- کارت کلید میانبر ---
    ck = card(t_general, "کلید میانبر شروع/توقف ضبط")
    var_hotkey = tk.StringVar(value=cfg.get("hotkey"))
    prev_hotkey = var_hotkey.get()
    MODIFIER_KEYS = ("control_l", "control_r", "shift_l", "shift_r",
                     "alt_l", "alt_r", "win_l", "win_r")

    def _valid_hotkey(hk: str) -> bool:
        parts = [p for p in hk.split("+") if p]
        if not parts:
            return False
        if parts[-1] in ("ctrl", "shift", "alt"):
            return False  # کلید نهایی نمی‌تواند خودِ modifier باشد
        if len(parts) == 1:
            # تک‌کلیدی فقط برای F-keyها مجاز است
            return parts[0].startswith("f") and parts[0][1:].isdigit()
        return True

    HINT_TXT = "برای ثبت میان‌بر جدید، روی کادر کلیک کن و ترکیب دلخواه را بفشار (لغو: Esc)"

    def capture_hotkey(event):
        key = event.keysym.lower()
        if key == "escape":  # لغو capture — بازگشت به مقدار قبلی
            var_hotkey.set(prev_hotkey)
            hk_hint.configure(text=HINT_TXT, text_color=theme.FG_DIM)
            return "break"
        if key in MODIFIER_KEYS:
            return "break"
        mods = []
        if event.state & 0x4:
            mods.append("ctrl")
        if event.state & 0x1:
            mods.append("shift")
        if event.state & 0x20000:
            mods.append("alt")
        # تک‌کلیدیِ بدون modifier تایپ عادی ویندوز را می‌شکند — فقط F-key مجاز است
        if not mods and not (key.startswith("f") and key[1:].isdigit()):
            var_hotkey.set(prev_hotkey)
            hk_hint.configure(text="ترکیب باید شامل کلید ترکیبی (کنترل، آلت یا شیفت) باشد، یا یک کلید F",
                              text_color=theme.DANGER)
            return "break"
        var_hotkey.set("+".join(mods + [key]))
        hk_hint.configure(text=HINT_TXT, text_color=theme.FG_DIM)
        return "break"

    hk = ctk.CTkEntry(ck, textvariable=var_hotkey, font=(fam, 14), height=40,
                      corner_radius=8, fg_color=theme.SURFACE_2,
                      border_color=theme.BORDER, text_color=theme.FG)
    hk.bind("<Key>", capture_hotkey)
    hk.pack(fill="x")
    hk_hint = ctk.CTkLabel(ck, text=HINT_TXT, font=(fam, 12),
                           text_color=theme.FG_DIM, anchor="e")
    hk_hint.pack(fill="x", pady=(3, 0))

    # --- کارت پنجره زنده و سیستم ---
    cv = card(t_general, "پنجره زنده و سیستم")
    var_overlay = tk.BooleanVar(value=bool(cfg.get("overlay_enabled")))
    switch_row(cv, "نمایش پنجره زنده هنگام ضبط", var_overlay, pady=(0, 4))

    frow = ctk.CTkFrame(cv, fg_color="transparent")
    frow.pack(fill="x", pady=(2, 0))
    font_lbl = tk.StringVar(value=f"اندازه متن: {int(cfg.get('overlay_font_size') or 15)}")
    ctk.CTkLabel(frow, textvariable=font_lbl, font=(fam, 12),
                 text_color=theme.FG_DIM, anchor="e").pack(side="right")
    var_font = tk.IntVar(value=int(cfg.get("overlay_font_size") or 15))

    def _on_font_slide(v):
        font_lbl.set(f"اندازه متن: {int(float(v))}")
        sample.configure(font=(fam, int(float(v))))

    ctk.CTkSlider(frow, from_=12, to=20, number_of_steps=8, variable=var_font,
                  command=_on_font_slide, width=150,
                  progress_color=theme.ACCENT, button_color=theme.ACCENT,
                  button_hover_color=theme.ACCENT_HOVER).pack(side="left")

    sample = ctk.CTkLabel(cv, text="نمونه متن فارسی", font=(fam, int(var_font.get())),
                          text_color=theme.FG, anchor="e")
    sample.pack(fill="x", pady=(6, 0))

    var_autostart = tk.BooleanVar(value=bool(cfg.get("autostart")))
    switch_row(cv, "اجرای خودکار با ورود به ویندوز", var_autostart, pady=(10, 0))

    # ================= تب میکروفون =================
    t_mic = tabview.tab("میکروفون")
    import sounddevice as sd

    cm = card(t_mic, "دستگاه ورودی")
    # یک مدخل برای هر میکروفون فیزیکی — ویندوز هر دستگاه را به ازای هر
    # Host API یک بار فهرست می‌کند (۳ میکروفون → ۱۵+ مدخل پرتکرار)
    all_inputs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            all_inputs.append({"index": i, "name": d["name"],
                               "rate": int(d["default_samplerate"]),
                               "api": sd.query_hostapis(d["hostapi"])["name"]})
    devices = [(d["index"], device_label(d))
               for d in dedupe_input_devices(all_inputs)]
    # اگر دستگاه پین‌شده‌ی فعلی در فهرست یکدست نیامد (API کم‌ترجیح)، برای
    # دیده‌شدن انتخاب فعلی اضافه شود
    cur_pin = cfg.get("input_device")
    if cur_pin is not None and cur_pin not in (idx for idx, _ in devices):
        for e in all_inputs:
            if e["index"] == cur_pin:
                devices.append((e["index"], device_label(e)))
                break

    auto = {"label": "خودکار (پرسیگنال‌ترین)"}
    cur = cfg.get("input_device")
    current_name = auto["label"]
    if cur is not None:
        match = [name for idx, name in devices if idx == cur]
        if match:
            current_name = match[0]

    def _auto_device_name(dev: int | None) -> str | None:
        """برچسب میکروفون انتخاب‌شده‌ی حالت خودکار — از فهرست فیزیکی."""
        if dev is None:
            return None
        e = next((x for x in all_inputs if x["index"] == dev), None)
        return device_label(e) if e else None

    if cur is None:
        # در حالت خودکار، اسم میکروفونی که تشخیص برگزیده کنار «خودکار» می‌آید
        found = getattr(app, "device", None) if app is not None else None
        if found is not None:
            lbl = _auto_device_name(found)
            if lbl:
                auto["label"] = f"خودکار — {lbl}"
        elif app is not None:
            # تشخیص پس‌زمینه هنوز تمام نشده — تمام که شد، برچسب زنده به‌روز می‌شود.
            # ترد فقط محاسبه می‌کند و نتیجه را در holder می‌گذارد؛ هر فراخوانی
            # Tk (win.after/...) باید از ترد اصلی باشد وگرنه
            # «main thread is not in main loop»
            holder = {"dev": None, "done": False}

            def _bg_detect():
                try:
                    holder["dev"] = detect_best_device()
                except Exception:
                    holder["dev"] = None
                finally:
                    holder["done"] = True

            threading.Thread(target=_bg_detect, daemon=True).start()

            def _apply_auto_label():
                if not win.winfo_exists():
                    return
                if not holder["done"]:
                    win.after(400, _apply_auto_label)
                    return
                lbl = _auto_device_name(holder["dev"])
                if not lbl:
                    return
                auto["label"] = f"خودکار — {lbl}"
                vals = [auto["label"]] + [name for _, name in devices]
                dev_combo.configure(values=vals)
                if dev_combo.get() not in vals:
                    dev_combo.set(auto["label"])

            win.after(400, _apply_auto_label)

    dev_values = [auto["label"]] + [name for _, name in devices]
    dev_combo = ctk.CTkOptionMenu(cm, values=dev_values, height=36,
                                  dynamic_resizing=False, anchor="e", **menu_style)
    dev_combo.set(current_name)
    dev_combo.pack(fill="x")
    dim(cm, "خودکار = پرسیگنال‌ترین میکروفون فعال در شروع هر ضبط")

    def selected_device():
        """دستگاه انتخابی در کمبو — None یعنی تشخیص خودکار."""
        v = dev_combo.get()
        if v == auto["label"] or not v:
            return None
        for idx, label in devices:
            if label == v:
                return idx
        return None

    def selected_device_key():
        """کلید پایدار انتخاب فعلی (نام — API) — برای بازیابی بعد از
        جابه‌جایی ایندکس‌ها بین بوت‌ها."""
        idx = selected_device()
        if idx is None:
            return None
        for i, label in devices:
            if i == idx:
                return label
        return None

    ct = card(t_mic, "تست صدا")
    tester = MicTester()
    var_testing = {"on": False}
    test_fallbacks: list[dict] = []   # مسیرهای جایگزین همان میکروفون — اگر مسیر اصلی باز نشود
    bars_hist: list[float] = [0.0] * SPECS_BARS
    hist_lock = threading.Lock()

    spec_canvas = tk.Canvas(ct, height=SPECS_H + 8,
                            bg=theme.DEEP, highlightthickness=0)
    spec_canvas.pack(fill="x", pady=(0, 2))

    test_btn_var = tk.StringVar(value="شروع تست")

    def toggle_test():
        if var_testing["on"]:
            var_testing["on"] = False
            tester.stop()
            test_btn_var.set("شروع تست")
            spec_canvas.delete("all")
            quality_lbl.configure(text="", text_color=theme.FG_DIM)
        else:
            dev = selected_device()
            if dev is None:
                # «خودکار» = همان دستگاهی که دیکته استفاده می‌کند؛
                # probe دوباره نه — نتایج detect ناپایدار است و ممکن است
                # به دستگاهی بیفتد که استریم باز نمی‌کند (بدون اسپاک)
                dev = getattr(app, "device", None) if app is not None else None
                if dev is None:
                    dev = detect_best_device()
                if dev is None:
                    dev = sd.default.device[0]
            tester.start(dev)
            test_fallbacks.clear()
            if dev is not None:
                test_fallbacks.extend(device_siblings(all_inputs, dev))
            test_vals.clear()
            var_testing["on"] = True
            test_btn_var.set("توقف تست")
            poll_spec()

    ctk.CTkButton(ct, textvariable=test_btn_var, font=(fam, 13, "bold"),
                  height=34, width=120, corner_radius=8,
                  fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
                  text_color=theme.FG, command=toggle_test).pack(pady=(0, 2))

    verdict_lbl = ctk.CTkLabel(ct, text="", font=(fam, 13, "bold"),
                               text_color=theme.FG, anchor="e")
    verdict_lbl.pack(fill="x")
    # نشانگر کیفیت ورودی — نویز پایه/اوج/SNR زنده حین تست
    test_vals: list[float] = []
    quality_lbl = ctk.CTkLabel(ct, text="", font=(fam, 12),
                               text_color=theme.FG_DIM, anchor="e",
                               wraplength=440, justify="right")
    quality_lbl.pack(fill="x")

    QUALITY_COLORS = {"good": theme.ACCENT, "warn": theme.WARN,
                      "bad": theme.DANGER, "none": theme.FG_DIM}

    def _friendly_audio_error(msg: str) -> str:
        low = (msg or "").lower()
        if "unanticipated host error" in low or "error starting stream" in low:
            return ("این مسیر دستگاه روی این سیستم باز نمی‌شود — اگر میکروفون مجازی "
                    "است برنامه‌اش را اجرا کن، یا مسیر دیگری (مثلاً WASAPI) همان "
                    "میکروفون را انتخاب کن")
        return msg

    def update_quality():
        from app.recorder import input_quality
        text, level = input_quality(test_vals)
        quality_lbl.configure(text=text, text_color=QUALITY_COLORS.get(level, theme.FG_DIM))

    # --- کارت رفتار ضبط ---
    cr = card(t_mic, "رفتار ضبط")
    row = ctk.CTkFrame(cr, fg_color="transparent")
    row.pack(fill="x")
    ctk.CTkLabel(row, text="توقف خودکار پس از سکوت", font=(fam, 13),
                 text_color=theme.FG, anchor="e").pack(side="right")
    auto_stop_labels = {v: k for k, v in AUTO_STOP_LABELS.items()}
    var_auto_stop = tk.StringVar(value=auto_stop_labels.get(int(cfg.get("auto_stop_sec") or 0), "خاموش"))
    ctk.CTkOptionMenu(row, values=list(AUTO_STOP_LABELS.keys()), variable=var_auto_stop,
                      width=110, height=34, **menu_style).pack(side="left")
    dim(cr, "اگر بعد از صحبت، N ثانیه سکوت کنی ضبط خودکار تمام و متن درج می‌شود")

    var_sound = tk.BooleanVar(value=bool(cfg.get("sound_feedback")))
    switch_row(cr, "بوق کوتاه هنگام شروع و پایان ضبط", var_sound, pady=(6, 0))

    def poll_spec():
        if not var_testing["on"]:
            return
        try:
            got_err = None
            vals = []
            while True:
                try:
                    kind, v = tester.q.get_nowait()
                except queue.Empty:
                    break
                if kind == "err":
                    got_err = v
                else:
                    vals.append(v)
            if got_err:
                if test_fallbacks:
                    # مسیر اصلی/قبلی باز نشد — خودکار روی مسیر دیگر همان میکروفون
                    nxt = test_fallbacks.pop(0)
                    verdict_lbl.configure(
                        text="این مسیر دستگاه باز نشد — تست روی مسیر جایگزین: "
                             f"{nxt['name'][:40]} ({nxt['api']})",
                        text_color=theme.WARN)
                    tester.start(nxt["index"])
                else:
                    verdict_lbl.configure(text=f"خطا: {_friendly_audio_error(got_err)}",
                                          text_color=theme.DANGER)
                    toggle_test()
                    return
            if vals:
                test_vals.extend(vals)
                if len(test_vals) > 400:
                    del test_vals[:-400]
                update_quality()
                rms = max(vals)
                norm = min(1.0, rms / 0.04)
                with hist_lock:
                    bars_hist[:-1] = bars_hist[1:]
                    bars_hist[-1] = norm
                if rms > 0.004:
                    verdict_lbl.configure(text="میکروفون کار می‌کند — صدای واضح",
                                          text_color=theme.ACCENT)
                elif rms > 0.0005:
                    verdict_lbl.configure(text="صدای خیلی کم — تقویت ورودی را بالا ببر",
                                          text_color=theme.WARN)
                else:
                    verdict_lbl.configure(text="سیگنالی نمی‌آید — دستگاه دیگری را امتحان کن",
                                          text_color=theme.DANGER)
            # رندر — حتی بدون داده جدید، موج نفس می‌کشد
            c = spec_canvas
            c.delete("all")
            w = c.winfo_width() or 480
            bw = 6
            with hist_lock:
                hist = list(bars_hist)
            for i, v in enumerate(reversed(hist)):  # جدیدترین در راست
                x = w - 8 - (i + 1) * (bw + 3)
                h = max(3, v * SPECS_H)
                if v > 0.5:
                    color = theme.ACCENT
                elif v > 0.15:
                    color = "#4f8f68"
                elif v > 0.02:
                    color = theme.SURFACE_3
                else:
                    color = "#2e2e2e"  # سکوت: مرئی اما خنثی
                c.create_rectangle(x, SPECS_H + 4 - h, x + bw, SPECS_H + 4,
                                   fill=color, outline="")
        except tk.TclError:
            var_testing["on"] = False  # پنجره بسته شد
            return
        win.after(80, poll_spec)

    # ================= تب درج متن =================
    t_insert = tabview.tab("درج متن")

    ci = card(t_insert, "روش درج")
    dim(ci, "متن تشخیص‌داده‌شده چگونه در برنامه مقصد برسد؟")
    var_paste = tk.StringVar(value=cfg.get("paste_method"))
    ctk.CTkRadioButton(ci, text="کلیپ‌بورد (پیشنهادی)",
                       variable=var_paste, value="clipboard",
                       **radio_style).pack(anchor="e", pady=(0, 2))
    ctk.CTkRadioButton(ci, text="تایپ مستقیم (کندتر)",
                       variable=var_paste, value="type",
                       **radio_style).pack(anchor="e")

    cv2 = card(t_insert, "کلیپ‌بورد و فرمان‌ها")
    var_restore = tk.BooleanVar(value=bool(cfg.get("restore_clipboard")))
    switch_row(cv2, "بازیابی محتوای قبلی کلیپ‌بورد بعد از درج",
               var_restore, pady=(0, 2))
    dim(cv2, "اگر غیرفعال شود، متن دیکته در کلیپ‌بورد می‌ماند")
    var_commands = tk.BooleanVar(value=bool(cfg.get("voice_commands")))
    switch_row(cv2, "فرمان‌های صوتی", var_commands, pady=(8, 0))
    dim(cv2, "نقطه، ویرگول، علامت سوال، گیومه باز/بسته، نقطه ویرگول، خط جدید، حذف آخرین کلمه")
    var_itn = tk.BooleanVar(value=bool(cfg.get("persian_itn")))
    switch_row(cv2, "تبدیل اعداد حروفی به رقم", var_itn, pady=(8, 0))
    dim(cv2, "اعداد حروفی خودکار به رقم تبدیل می‌شوند؛ اعداد تکی مثل «یک» حروفی می‌مانند")

    # ================= تب پیشرفته (اسکرول‌شونده — محتوای بلند) =================
    t_adv = tabview.tab("پیشرفته")
    t_adv = ctk.CTkScrollableFrame(
        t_adv, fg_color="transparent", scrollbar_fg_color="transparent",
        scrollbar_button_color=theme.SURFACE_3,
        scrollbar_button_hover_color=theme.SURFACE_2,
    )
    t_adv.pack(fill="both", expand=True)

    ca = card(t_adv, "پردازش")
    arow = ctk.CTkFrame(ca, fg_color="transparent")
    arow.pack(fill="x")
    ctk.CTkLabel(arow, text="تعداد هسته پردازش مدل (با ری‌استارت اعمال می‌شود)",
                 font=(fam, 13), text_color=theme.FG, anchor="e").pack(side="right")
    var_threads = tk.StringVar(value=str(int(cfg.get("num_threads") or 4)))
    ctk.CTkOptionMenu(arow, values=[str(i) for i in range(1, 9)], variable=var_threads,
                      width=80, height=34, **menu_style).pack(side="left")

    cs = card(t_adv, "متن زنده پایدار — آزمایشی")
    var_stable_live = tk.BooleanVar(value=bool(cfg.get("stable_live")))
    switch_row(cs, "قفل واژه‌های قطعی (رأی بین‌پنجره‌ای + امتیاز اطمینان)",
               var_stable_live, pady=(0, 6),
               command=lambda: _sync_enroll_ui())
    dim(cs, "واژه فقط وقتی قطعی می‌شود که در پنجره‌های پیاپی پایدار باشد، رقیب هم‌زمان نداشته باشد و از لبه خارج نشده باشد؛ نوسان نمایش کمتر می‌شود")
    dim(cs, "نمایش زنده = کل متن قفل‌شده + پنجره‌ی جاری؛ واژه‌های هنوز قطعی‌نشده کم‌رنگ‌تر دیده می‌شوند")
    dim(cs, "متن نهایی از مسیر جداگانه ساخته می‌شود و تحت تأثیر نیست؛ خروجی ممکن است کمی با حالت پیش‌فرض متفاوت باشد")
    dim(cs, "در این حالت اصلاح واژه‌های ثبت‌شده (تب پیشرفته) روی خروجی زنده و نهایی اعمال نمی‌شود")
    dim(cs, "پیش‌فرض خاموش است؛ اگر وسط ضبط تغییرش دهید، بعد از پایان ضبط اعمال می‌شود")

    ch_hw = card(t_adv, "واژه‌های حساس (هات‌وورد) — آزمایشی")
    var_hotword = tk.BooleanVar(value=bool(cfg.get("hotword_boost")))
    switch_row(ch_hw, "تقویت واژه‌های مشخص هنگام تشخیص", var_hotword, pady=(0, 6))
    txt_hotwords = ctk.CTkTextbox(ch_hw, height=110, font=(fam, 13))
    txt_hotwords.pack(fill="x")
    txt_hotwords.insert("1.0", "\n".join(str(w) for w in (cfg.get("hotwords") or [])))
    dim(ch_hw, "هر خط یک واژه، حداقل ۲ حرف — اسم‌ها و برندهایی که مدل مدام اشتباه می‌گیرد")
    dim(ch_hw, "با روشن‌کردن، پردازش کمی کندتر می‌شود و ممکن است نشانه‌های پایانی جمله (مثل نقطه) هم درج شوند")

    ce = card(t_adv, "ثبت صوتی واژه‌ها — آزمایشی")

    def _sync_enroll_ui():
        # تعریف قبل از دکمه، ولی بدنه در زمان فراخوانی resolve می‌شود
        stable = bool(var_stable_live.get())
        # در حالت «متن زنده پایدار» این لایه روی خروجی زنده و نهایی اعمال
        # نمی‌شود (گویش دیکد موتور پایدار با گویش واریانت‌ها فرق دارد)، پس
        # کلید به‌جای روشن‌بودنِ بی‌اثر، غیرفعال نشان داده می‌شود.
        sw_enroll.configure(state="disabled" if stable else "normal")
        # برچسب متن هم با سوییچ خاکستری شود — سوییچ دیگر متن ندارد
        lbl_enroll.configure(state="disabled" if stable else "normal")
        add_btn.configure(
            state="normal" if (var_enroll.get() and not stable) else "disabled"
        )
        if stable:
            lbl_stable_note.pack(fill="x", pady=(0, 4), before=enroll_list)
        else:
            lbl_stable_note.pack_forget()

    var_enroll = tk.BooleanVar(value=bool(cfg.get("enroll_alias")))
    sw_enroll, lbl_enroll = switch_row(ce, "اصلاح واژه‌های ثبت‌شده در خروجی",
                                       var_enroll, pady=(0, 6),
                                       command=_sync_enroll_ui)
    dim(ce, "واژه‌ای که مدل مدام اشتباه می‌شنود را ضبط کن؛ شکل‌های شنیده‌شده را تیک بزن تا در خروجی به واژه‌ی درست تبدیل شوند")
    dim(ce, "اثر هم روی متن زنده و هم روی متن نهایی دارد؛ در حالت «متن زنده پایدار» اعمال نمی‌شود")

    lbl_stable_note = ctk.CTkLabel(
        ce, text="«متن زنده پایدار» روشن است — تا خاموشش نکنی این لایه روی خروجی زنده و نهایی اعمال نمی‌شود",
        font=(fam, 12), text_color=theme.WARN, justify="right", anchor="e",
        wraplength=440,
    )

    from app import enroll as enroll_mod

    enroll_store = enroll_mod.EnrollStore.load()
    enroll_list = ctk.CTkFrame(ce, fg_color="transparent")
    enroll_list.pack(fill="x")

    def _enroll_changed():
        enroll_store.save()
        if app is not None:
            try:
                app.refresh_alias_map()
            except Exception:
                pass
        rebuild_enroll_list()

    def rebuild_enroll_list():
        for w in enroll_list.winfo_children():
            w.destroy()
        if not enroll_store.entries:
            dim(enroll_list, "هنوز واژه‌ای ثبت نشده")
            return
        for e in enroll_store.entries:
            row = ctk.CTkFrame(enroll_list, fg_color="transparent")
            row.pack(fill="x", pady=1)
            n_var = len(e.get("variants", []))
            ctk.CTkLabel(row, text=f"«{e['word']}» — {n_var} واریانت تأییدشده",
                         font=(fam, 13), text_color=theme.FG,
                         anchor="e").pack(side="right")
            ctk.CTkButton(row, text="ویرایش", width=60, height=26,
                          corner_radius=6, font=(fam, 12),
                          fg_color=theme.SURFACE_2,
                          hover_color=theme.SURFACE_3, text_color=theme.FG,
                          command=lambda wd=e: open_enroll_dialog(wd)
                          ).pack(side="left", padx=(4, 0))
            ctk.CTkButton(row, text="حذف", width=56, height=26, corner_radius=6,
                          font=(fam, 12), fg_color=theme.SURFACE_2,
                          hover_color=theme.DANGER, text_color=theme.FG,
                          command=lambda wd=e["word"]: (
                              enroll_store.remove_entry(wd), _enroll_changed())
                          ).pack(side="left")

    def open_enroll_dialog(entry: dict | None = None):
        """entry=None → ثبت واژه جدید؛ dict → ویرایش همان واژه."""
        dlg = tk.Toplevel(win)
        dlg.title("ویرایش واژه" if entry else "ثبت واژه جدید")
        dlg.geometry("470x430")
        dlg.attributes("-topmost", True)
        dlg.configure(bg=theme.BG)
        # style_toplevel پنجره را مخفی نگه می‌دارد (ضد فلش سفید)؛ نمایش
        # در پایان با smooth_show — و grab بعد از نمایان‌شدن، وگرنه رویدادها
        # به پنجره‌ی نامرئی می‌رود و تنظیمات فریز می‌شود
        style_toplevel(dlg)
        apply_icon(dlg)

        body = ctk.CTkFrame(dlg, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        ctk.CTkLabel(body, text="واژه‌ی درست — همان‌طور که باید نوشته شود:",
                     font=(fam, 13), text_color=theme.FG,
                     anchor="e").pack(fill="x", pady=(0, 3))
        var_word = tk.StringVar()
        if entry:
            var_word.set(str(entry.get("word", "")))
        ctk.CTkEntry(body, textvariable=var_word, font=(fam, 14), height=38,
                     corner_radius=8, fg_color=theme.SURFACE_2,
                     border_color=theme.BORDER,
                     text_color=theme.FG).pack(fill="x")

        ctk.CTkLabel(body, text="سه بار واضح بگو — ضبط را شروع کن، بگو، و قطع کن؛ بعد با پخش گوش بده:",
                     font=(fam, 13), text_color=theme.FG,
                     anchor="e").pack(fill="x", pady=(10, 3))

        heard_forms: list[str] = [str(v) for v in (entry or {}).get("variants", [])]
        var_checks: dict[str, tk.BooleanVar] = {}
        check_frame = ctk.CTkFrame(body, fg_color="transparent")
        check_frame.pack(fill="x", pady=(2, 0))
        # افزودن دستی واریانت — شکل شنیده‌شده را که در متن زنده دیدی،
        # بدون ضبط مجدد همین‌جا تایپ کن؛ شنیدنِ مدل در دیکته با ضبطِ
        # تنها فرق می‌کند و دقیق‌ترین منبع واریانت همان متن زنده است
        manual_row = ctk.CTkFrame(body, fg_color="transparent")
        manual_row.pack(fill="x", pady=(0, 6))
        var_manual = tk.StringVar()
        ctk.CTkEntry(manual_row, textvariable=var_manual, font=(fam, 13),
                     height=32, corner_radius=6, fg_color=theme.SURFACE_2,
                     border_color=theme.BORDER,
                     text_color=theme.FG).pack(side="right", fill="x",
                                               expand=True, padx=(6, 0))
        ctk.CTkButton(manual_row, text="+ افزودن دستی", width=110, height=30,
                      corner_radius=6, font=(fam, 12),
                      fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
                      text_color=theme.FG,
                      command=lambda: add_manual_variant()).pack(side="left")
        slots: list[dict] = []               # per-slot: rec/samples/timer
        status_lbls: list[ctk.CTkLabel] = []
        rec_btns: list[ctk.CTkButton] = []
        play_btns: list[ctk.CTkButton] = []
        busy = {"slot": -1}                  # اسلات در حال ضبط
        playing = {"slot": -1}               # اسلات در حال پخش
        result_q: queue.Queue = queue.Queue()
        REC_MAX_SEC = 10.0                   # سقف ایمنی — توقف خودکار

        engine_ok = app is not None and getattr(app, "engine", None) is not None

        def rebuild_checks():
            for w in check_frame.winfo_children():
                w.destroy()
            for v in heard_forms:
                if v in var_checks:
                    continue
                var_checks[v] = tk.BooleanVar(value=True)
            if not heard_forms:
                dim(check_frame, "هنوز واریانتی نیست — ضبط کن یا دستی اضافه کن")
                return
            dim(check_frame, "شکل‌های شنیده‌شده — هر کدام را تأیید می‌کنی در خروجی جای واژه‌ی درست می‌نشیند:")
            for v, var in var_checks.items():
                ctk.CTkCheckBox(check_frame, text=f"«{v}»", variable=var,
                                **check_style).pack(anchor="e", pady=1)

        def add_manual_variant():
            v = var_manual.get().strip()
            if len(v) < 2:
                return
            if v not in heard_forms:
                heard_forms.append(v)
            var_manual.set("")
            rebuild_checks()

        def _decode_worker(slot: int, data, word: str):
            # هیچ دسترسی Tk اینجا ممنوع — word و data از ترد اصلی آمده‌اند
            try:
                text = str(app.engine.transcribe(data, 16000) or "")
                variants = enroll_mod.harvest_variants(
                    app.engine, data, word, text=text)
                result_q.put(("done", slot, text, variants))
            except Exception as e:
                result_q.put(("err", slot, str(e)[:60], []))

        def stop_rec(slot: int):
            s = slots[slot]
            if s["timer"] is not None:
                dlg.after_cancel(s["timer"])
                s["timer"] = None
            rec, s["rec"] = s["rec"], None
            busy["slot"] = -1
            rec.stop()
            data = rec.get_buffer_16k()
            s["samples"] = data
            # بدون این، دکمه پخش برای همیشه disabled می‌ماند — ریشه‌ی
            # «پخش کار نمی‌کند»؛ حتی ضبط کوتاه برای تشخیص قابل پخش است
            play_btns[slot].configure(state="normal")
            rec_btns[slot].configure(
                text=f"ضبط {'۱۲۳'[slot]}", state="normal",
                fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3)
            for j, b in enumerate(rec_btns):
                if j != slot and slots[j]["rec"] is None:
                    b.configure(state="normal")
            if data.size < 0.3 * 16000:
                status_lbls[slot].configure(
                    text="ضبط خیلی کوتاه بود — دوباره ضبط کن",
                    text_color=theme.WARN)
                return
            status_lbls[slot].configure(text="در حال پردازش…",
                                        text_color=theme.WARN)
            # word همین‌جا در ترد اصلی خوانده می‌شود — StringVar.get از
            # ترد کارگر «main thread is not in main loop» می‌دهد
            word = var_word.get().strip()
            threading.Thread(target=_decode_worker,
                             args=(slot, data, word), daemon=True).start()

        def toggle_rec(slot: int):
            if busy["slot"] == slot:
                stop_rec(slot)
                return
            if busy["slot"] >= 0:
                return
            if not var_word.get().strip():
                status_lbls[slot].configure(text="اول واژه‌ی درست را بنویس",
                                            text_color=theme.DANGER)
                return
            if not engine_ok:
                status_lbls[slot].configure(
                    text="موتور تشخیص هنوز بارگذاری نشده",
                    text_color=theme.DANGER)
                return
            # دستگاه در ترد اصلی حل می‌شود — خواندن کمبوی CTk از ترد کارگر خطا می‌دهد
            dev = selected_device()
            if dev is None and app is not None:
                dev = getattr(app, "device", None)
            try:
                rec = Recorder(device=dev, block_ms=50)
                rec.start()  # همان مسیر ضبط دیکته — سریع و بی‌probe
            except Exception as e:
                status_lbls[slot].configure(text=f"خطا: {str(e)[:50]}",
                                            text_color=theme.DANGER)
                return
            busy["slot"] = slot
            slots[slot]["rec"] = rec
            rec_btns[slot].configure(
                text="توقف", fg_color=theme.DANGER, hover_color=theme.DANGER)
            play_btns[slot].configure(state="disabled")
            status_lbls[slot].configure(text="در حال ضبط…", text_color=theme.WARN)
            slots[slot]["timer"] = dlg.after(
                int(REC_MAX_SEC * 1000), lambda: stop_rec(slot))

        def play_slot(slot: int):
            import sounddevice as sd

            data = slots[slot]["samples"]
            if data is None or not data.size:
                return
            btn = play_btns[slot]
            if playing["slot"] == slot:  # در حال پخش — قطع
                sd.stop()
                playing["slot"] = -1
                btn.configure(text="پخش")
                return
            sd.stop()
            # ضبط میکروفون معمولاً خیلی کم‌صدا است (peak ~۰٫۰۱) — مدل ASR
            # آن را راحت می‌شنود ولی پخش مستقیمش تقریباً نامرئی است؛
            # برای پخش به peak نرمال می‌شود (سقف تقویت ×۳۰)
            out = data
            peak = float(np.abs(data).max())
            if 0.0 < peak < 0.15:
                out = np.clip(data * min(30.0, 0.5 / peak), -1.0, 1.0)
            try:
                sd.play(out, 16000)
            except Exception as e:
                status_lbls[slot].configure(text=f"خطای پخش: {str(e)[:40]}",
                                            text_color=theme.DANGER)
                return
            playing["slot"] = slot
            btn.configure(text="قطع")

            def _reset():
                # پایان طبیعی پخش — بدون این، فشار بعدی «قطع» می‌شد و صدا نمی‌داد
                if playing["slot"] == slot:
                    playing["slot"] = -1
                    if btn.winfo_exists():
                        btn.configure(text="پخش")

            dlg.after(int(len(data) / 16000 * 1000) + 300, _reset)

        def poll_results():
            try:
                while True:
                    kind, slot, payload, variants = result_q.get_nowait()
                    if kind == "err":
                        status_lbls[slot].configure(text=f"خطا: {payload}",
                                                    text_color=theme.DANGER)
                    else:
                        new = [v for v in variants if v not in heard_forms]
                        heard_forms.extend(new)
                        shown = payload.strip() or "چیزی شنیده نشد"
                        status_lbls[slot].configure(
                            text=f"شنیده شد: {shown}",
                            text_color=theme.ACCENT if new else theme.WARN)
                        rebuild_checks()
            except queue.Empty:
                pass
            if dlg.winfo_exists():
                dlg.after(100, poll_results)

        for i in range(3):
            slots.append({"rec": None, "samples": None, "timer": None})
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=2)
            st = ctk.CTkLabel(row, text="—", font=(fam, 12),
                              text_color=theme.FG_DIM, anchor="e")
            st.pack(side="right", fill="x", expand=True, padx=(6, 0))
            pbtn = ctk.CTkButton(row, text="پخش", width=60, height=30,
                                 corner_radius=6, font=(fam, 12),
                                 fg_color=theme.SURFACE_2,
                                 hover_color=theme.SURFACE_3,
                                 text_color=theme.FG, state="disabled",
                                 command=lambda s=i: play_slot(s))
            pbtn.pack(side="left", padx=(6, 0))
            btn = ctk.CTkButton(row, text=f"ضبط {'۱۲۳'[i]}", width=80, height=30,
                                corner_radius=6, font=(fam, 12, "bold"),
                                fg_color=theme.SURFACE_2,
                                hover_color=theme.SURFACE_3, text_color=theme.FG,
                                command=lambda s=i: toggle_rec(s))
            btn.pack(side="left")
            status_lbls.append(st)
            rec_btns.append(btn)
            play_btns.append(pbtn)
        if not engine_ok:
            dim(body, "موتور تشخیص هنوز بارگذاری نشده — بعد از آماده‌شدن اپ دوباره باز کن")

        rebuild_checks()
        dlg.after(100, poll_results)
        smooth_show(dlg)  # نمایش نرم بعد از ساخت کامل — ویندوز از قبل مخفی بود
        dlg.grab_set()    # فقط بعد از نمایان‌شدن؛گرنه grab روی پنجره مخفی می‌ماند

        def save_entry():
            word = var_word.get().strip()
            if len(word) < 2:
                return
            checked = [v for v, var in var_checks.items() if var.get()]
            if entry is not None and \
                    enroll_mod.norm_word(str(entry.get("word", ""))) != \
                    enroll_mod.norm_word(word):
                # متن واژه عوض شده — مدخل با نام قبلی حذف شود
                enroll_store.remove_entry(str(entry.get("word", "")))
            enroll_store.add_entry(word, checked)
            _enroll_changed()
            dlg.destroy()

        btnrow = ctk.CTkFrame(body, fg_color="transparent")
        btnrow.pack(side="bottom", fill="x", pady=(8, 0))
        ctk.CTkButton(btnrow, text="ذخیره واژه", font=(fam, 13, "bold"),
                      height=36, width=130, corner_radius=8,
                      fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                      text_color=theme.ON_ACCENT,
                      command=save_entry).pack(side="right")
        ctk.CTkButton(btnrow, text="انصراف", font=(fam, 13),
                      height=36, width=100, corner_radius=8,
                      fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
                      text_color=theme.FG,
                      command=dlg.destroy).pack(side="left")

    add_btn = ctk.CTkButton(ce, text="+ ثبت واژه جدید", font=(fam, 13, "bold"),
                            height=34, width=140, corner_radius=8,
                            fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
                            text_color=theme.FG,
                            command=open_enroll_dialog)
    add_btn.pack(anchor="e", pady=(6, 0))
    _sync_enroll_ui()
    rebuild_enroll_list()

    cm_info = card(t_adv, "درباره موتور تشخیص")
    ctk.CTkLabel(cm_info, text="Shenava-Koochik v1.0", font=(fam, 13, "bold"),
                 text_color=theme.FG, anchor="e").pack(fill="x")
    dim(cm_info, "کاملاً آفلاین — ۱۱۴ میلیون پارامتر (معماری FastConformer)")
    dim(cm_info, "کیفیت روی جملات دیکته‌شده بهتر از مکالمه آزاد است")

    # ================= تب راهنما (اسکرول‌شونده — محتوای بلند) =================
    t_help = tabview.tab("راهنما")
    t_help = ctk.CTkScrollableFrame(
        t_help, fg_color="transparent", scrollbar_fg_color="transparent",
        scrollbar_button_color=theme.SURFACE_3,
        scrollbar_button_hover_color=theme.SURFACE_2,
    )
    t_help.pack(fill="both", expand=True)

    _fa_ver = str(APP_VERSION).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
    ch = card(t_help, f"{APP_TITLE_FULL} — نسخه {_fa_ver}")
    dim(ch, "دیکته صوتی فارسی، کاملاً آفلاین — تجربه‌ای شبیه ویسپر فلو")
    dim(ch, "مدل شنوا کوچیک، ۱۱۴ میلیون پارامتر — هیچ داده‌ای از سیستم شما خارج نمی‌شود")

    c1 = card(t_help, "استفاده سریع")
    dim(c1, "گام اول — در هر برنامه‌ای (نوت‌پد، تلگرام، مرورگر…) کلید میان‌بر را بزن")
    dim(c1, "گام دوم — صحبت کن؛ پنجره زنده کنار موس متن را همزمان نشان می‌دهد")
    dim(c1, "گام سوم — همان کلید را دوباره بزن تا متن در محل کرسر درج شود")

    c2 = card(t_help, "فرمان‌های صوتی")
    dim(c2, "بگو «نقطه» یا «ویرگول» یا «علامت سوال» تا نشانه درج شود")
    dim(c2, "«گیومه باز» و «گیومه بسته» برای « »، «نقطه ویرگول» برای ؛")
    dim(c2, "برای رفتن به خط بعد، «خط جدید» را بگو")
    dim(c2, "«حذف آخرین کلمه» آخرین کلمه درج‌شده را پاک می‌کند")

    c3 = card(t_help, "نکته‌ها")
    dim(c3, "کلید میان‌بر فقط برای همین اپ مصرف می‌شود و به برنامه مقصد فرستاده نمی‌شود")
    dim(c3, "اگر با میان‌بر برنامه دیگری تداخل داشت، از تب عمومی یک ترکیب تازه بگیر")
    dim(c3, "در برنامه‌هایی که با دسترسی مدیر باز شده‌اند درج کار نمی‌کند؛ اپ را هم مدیر اجرا کن یا روش درج را عوض کن")
    dim(c3, "اگر میکروفون را عوض کردی، از تب میکروفون دستگاه را انتخاب کن یا حالت خودکار را نگه دار")
    dim(c3, "اعداد حروفی خودکار به رقم تبدیل می‌شوند؛ خاموش یا روشن‌کردنش از تب درج متن است")
    dim(c3, "اگر مدل واژه‌ای را مدام غلط می‌شنود، از تب پیشرفته آن را صوتی ثبت کن تا از این پس درست نوشته شود")

    # ================= دکمه‌های ثابت پایین (pack در بالای فایل انجام شد) =================
    def _sync(data: dict):
        """بارگذاری مقادیر روی ویجت‌ها — برای بازنشانی."""
        nonlocal prev_hotkey
        var_hotkey.set(data.get("hotkey"))
        prev_hotkey = data.get("hotkey")
        hk_hint.configure(text=HINT_TXT, text_color=theme.FG_DIM)
        dev_combo.set(auto["label"])
        var_paste.set(data.get("paste_method"))
        var_commands.set(bool(data.get("voice_commands")))
        var_itn.set(bool(data.get("persian_itn")))
        var_restore.set(bool(data.get("restore_clipboard")))
        var_sound.set(bool(data.get("sound_feedback")))
        var_overlay.set(bool(data.get("overlay_enabled")))
        var_autostart.set(bool(data.get("autostart")))
        var_threads.set(str(int(data.get("num_threads") or 4)))
        var_font.set(int(data.get("overlay_font_size") or 15))
        font_lbl.set(f"اندازه متن: {int(var_font.get())}")
        sample.configure(font=(fam, int(var_font.get())))
        var_auto_stop.set(auto_stop_labels.get(int(data.get("auto_stop_sec") or 0), "خاموش"))
        var_hotword.set(bool(data.get("hotword_boost")))
        var_stable_live.set(bool(data.get("stable_live")))
        var_enroll.set(bool(data.get("enroll_alias")))
        _sync_enroll_ui()
        txt_hotwords.delete("1.0", "end")
        txt_hotwords.insert("1.0", "\n".join(str(w) for w in (data.get("hotwords") or [])))

    def reset():
        _sync(dict(DEFAULTS))

    def save():
        nonlocal hotkey_suspended
        new_hotkey = var_hotkey.get().strip()
        if not _valid_hotkey(new_hotkey):
            hk_hint.configure(text="کلید میانبر نامعتبر است", text_color=theme.DANGER)
            return
        cfg.set("hotkey", new_hotkey)
        cfg.set("paste_method", var_paste.get())
        cfg.set("voice_commands", var_commands.get())
        cfg.set("persian_itn", var_itn.get())
        cfg.set("restore_clipboard", var_restore.get())
        cfg.set("sound_feedback", var_sound.get())
        cfg.set("overlay_enabled", var_overlay.get())
        cfg.set("autostart", var_autostart.get())
        try:
            cfg.set("num_threads", max(1, min(8, int(var_threads.get()))))
        except ValueError:
            pass
        cfg.set("overlay_font_size", int(var_font.get()))
        cfg.set("auto_stop_sec", AUTO_STOP_LABELS.get(var_auto_stop.get(), 0))
        cfg.set("input_device", selected_device())
        cfg.set("input_device_key", selected_device_key())
        hw_on = bool(var_hotword.get())
        hw_list = [ln.strip() for ln in txt_hotwords.get("1.0", "end").splitlines()
                   if len(ln.strip()) >= 2]
        if hw_on and not hw_list:
            hk_hint.configure(text="حالت واژه‌های حساس روشن است ولی هیچ واژه‌ای وارد نشده",
                              text_color=theme.DANGER)
            return
        cfg.set("hotword_boost", hw_on)
        cfg.set("hotwords", hw_list)
        cfg.set("stable_live", bool(var_stable_live.get()))
        cfg.set("enroll_alias", bool(var_enroll.get()))
        cfg.save()
        set_autostart(var_autostart.get())
        tester.stop()
        if app is not None:
            app.apply_config()
        # apply_config هات‌کی جدید را ثبت کرده — دیگر resume لازم نیست
        hotkey_suspended = False
        close()

    def close():
        tester.stop()
        _resume_hotkey_once()
        win.destroy()

    def _resume_hotkey_once():
        """برگرداندن میانبر در هر مسیر بسته‌شدن — حتی نابهنجار."""
        nonlocal hotkey_suspended
        if hotkey_suspended and app is not None:
            hotkey_suspended = False
            try:
                app.resume_hotkey()
            except Exception:
                pass

    def _on_destroy(event):
        if event.widget is win:
            _resume_hotkey_once()

    # destroy بدون close (خطای نیمه‌راه در ساخت/کد خارجی) هم پوشش داده می‌شود
    win.bind("<Destroy>", _on_destroy)

    # سه دکمهی همعرض ۱۱۰ — مثل نسخهی فلت؛ ردیف بعد از چیدمان با لبه‌ی
    # کارت‌های تب تراز می‌شود: «ذخیره» از سمت شروع (راست) و «بازنشانی»
    # تا انتهای ردیف (چپ).
    btn_reset = ctk.CTkButton(btn_bar, text="بازنشانی", font=(fam, 13), height=40,
                              width=110, corner_radius=8, fg_color=theme.SURFACE_2,
                              hover_color=theme.SURFACE_3, text_color=theme.FG,
                              command=reset)
    btn_reset.pack(side="left")
    btn_save = ctk.CTkButton(btn_bar, text="ذخیره", font=(fam, 13, "bold"), height=40,
                             width=110, corner_radius=8, fg_color=theme.ACCENT,
                             hover_color=theme.ACCENT_HOVER, text_color=theme.ON_ACCENT,
                             command=save)
    btn_save.pack(side="right", padx=(8, 0))
    btn_close = ctk.CTkButton(btn_bar, text="انصراف", font=(fam, 13), height=40,
                              width=110, corner_radius=8, fg_color=theme.SURFACE_2,
                              hover_color=theme.SURFACE_3, text_color=theme.FG,
                              command=close)
    btn_close.pack(side="right")

    def _align_btn_bar():
        """تراز لبه‌ی ردیف دکمه‌ها با کارت‌های تب —
        بعد از اینکه چیدمان واقعی پنجره نشست.

        عرض دکمه‌ها دیگر اینجا بازنویسی نمیشود: قبلاً هر سه به یکسوم عرض
        پنجره (~۱۶۰px) بزرگ میشدند که با ۱۱۰ ثابتِ نسخهی فلت تفاوت داشت.
        """
        try:
            inset = (tabview.tab("عمومی").winfo_rootx()
                     - win.winfo_rootx()) + 2  # +2: padx کارت داخل تب
            if inset <= 0:
                return
            btn_bar.configure(padx=inset)
        except Exception:
            pass

    win.after(80, _align_btn_bar)
    win.after(450, _align_btn_bar)  # بعد از settle نهایی smooth_show

    win.protocol("WM_DELETE_WINDOW", close)
    smooth_ctk.flush_pending(win)  # پرکردن بوم‌های خالی — دکمه‌ها از اولین فریم کامل
    smooth_show(win)  # نمایش نرم بعد از ساخت کامل محتوا — بدون فریم سفید
