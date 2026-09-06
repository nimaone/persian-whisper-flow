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

from app import theme
from app.config import APP_NAME, APP_VERSION, DEFAULTS, Config, set_autostart
from app.recorder import detect_best_device
from app.win32 import style_toplevel, smooth_show, disable_min_max
from app.theme import apply_icon

SPECS_BARS = 48
SPECS_H = 40

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
    cfg = Config.load()
    fam = theme.family()

    win = tk.Toplevel(parent_root)
    win.title(f"تنظیمات — {APP_NAME}")
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
        c.pack(fill="x", pady=(10, 0), padx=2)
        inner = ctk.CTkFrame(c, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=12)
        if title:
            ctk.CTkLabel(inner, text=title, font=(fam, 13, "bold"),
                         text_color=theme.FG, anchor="e").pack(fill="x", pady=(0, 8))
        return inner

    def dim(parent, text):
        ctk.CTkLabel(parent, text=text, font=(fam, 12), text_color=theme.FG_DIM,
                     anchor="e", justify="right").pack(fill="x", pady=(2, 3))

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
                        button_color=theme.FG, button_hover_color="#ffffff")
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

    HINT_TXT = "روی کادر کلیک کن و ترکیب دلخواه را بفشار (Esc = لغو)"

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
            hk_hint.configure(text="ترکیب باید Ctrl یا Alt یا Shift داشته باشد (یا کلید F)",
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
    ctk.CTkSwitch(cv, text="نمایش پنجره زنده هنگام ضبط",
                  variable=var_overlay, **switch_style).pack(anchor="e", pady=(0, 4))

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
    ctk.CTkSwitch(cv, text="اجرای خودکار با ورود به ویندوز",
                  variable=var_autostart, **switch_style).pack(anchor="e", pady=(10, 0))

    # ================= تب میکروفون =================
    t_mic = tabview.tab("میکروفون")
    import sounddevice as sd

    cm = card(t_mic, "دستگاه ورودی")
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append((i, f"[{i}] {d['name']}"))

    auto_label = "خودکار (پرسیگنال‌ترین)"
    cur = cfg.get("input_device")
    current_name = auto_label
    if cur is not None:
        match = [name for idx, name in devices if idx == cur]
        if match:
            current_name = match[0]
    dev_values = [auto_label] + [name for _, name in devices]
    dev_combo = ctk.CTkOptionMenu(cm, values=dev_values, height=40,
                                  dynamic_resizing=False, anchor="e", **menu_style)
    dev_combo.set(current_name)
    dev_combo.pack(fill="x")
    dim(cm, "خودکار = پرسیگنال‌ترین میکروفون فعال در شروع هر ضبط")

    def selected_device():
        """دستگاه انتخابی در کمبو — None یعنی تشخیص خودکار."""
        v = dev_combo.get()
        if v == auto_label or not v:
            return None
        try:
            return int(v.split("]")[0][1:])
        except Exception:
            return None

    ct = card(t_mic, "تست صدا")
    tester = MicTester()
    var_testing = {"on": False}
    bars_hist: list[float] = [0.0] * SPECS_BARS
    hist_lock = threading.Lock()

    spec_canvas = tk.Canvas(ct, height=SPECS_H + 8,
                            bg=theme.DEEP, highlightthickness=0)
    spec_canvas.pack(fill="x", pady=(0, 6))

    test_btn_var = tk.StringVar(value="شروع تست")

    def toggle_test():
        if var_testing["on"]:
            var_testing["on"] = False
            tester.stop()
            test_btn_var.set("شروع تست")
            spec_canvas.delete("all")
        else:
            dev = selected_device()
            if dev is None:
                # همان دستگاهی که دیکته استفاده می‌کند — پیش‌فرض سیستم
                # روی خیلی از سیستم‌ها دستگاهی ساکت است و تست بی‌اثر می‌شود
                dev = detect_best_device()
                if dev is None:
                    dev = sd.default.device[0]
            tester.start(dev)
            var_testing["on"] = True
            test_btn_var.set("توقف تست")
            poll_spec()

    ctk.CTkButton(ct, textvariable=test_btn_var, font=(fam, 13, "bold"),
                  height=34, width=120, corner_radius=8,
                  fg_color=theme.SURFACE_2, hover_color=theme.SURFACE_3,
                  text_color=theme.FG, command=toggle_test).pack(pady=(0, 4))

    verdict_lbl = ctk.CTkLabel(ct, text="", font=(fam, 13, "bold"),
                               text_color=theme.FG, anchor="e")
    verdict_lbl.pack(fill="x")

    # --- کارت رفتار ضبط ---
    cr = card(t_mic, "رفتار ضبط")
    row = ctk.CTkFrame(cr, fg_color="transparent")
    row.pack(fill="x")
    ctk.CTkLabel(row, text="توقف خودکار پس از سکوت:", font=(fam, 13),
                 text_color=theme.FG, anchor="e").pack(side="right")
    auto_stop_labels = {v: k for k, v in AUTO_STOP_LABELS.items()}
    var_auto_stop = tk.StringVar(value=auto_stop_labels.get(int(cfg.get("auto_stop_sec") or 0), "خاموش"))
    ctk.CTkOptionMenu(row, values=list(AUTO_STOP_LABELS.keys()), variable=var_auto_stop,
                      width=110, height=34, **menu_style).pack(side="left")
    dim(cr, "اگر بعد از صحبت، N ثانیه سکوت کنی ضبط خودکار تمام و متن درج می‌شود")

    var_sound = tk.BooleanVar(value=bool(cfg.get("sound_feedback")))
    ctk.CTkSwitch(cr, text="بوق کوتاه هنگام شروع و پایان ضبط",
                  variable=var_sound, **switch_style).pack(anchor="e", pady=(10, 0))

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
                verdict_lbl.configure(text=f"خطا: {got_err}", text_color=theme.DANGER)
                toggle_test()
                return
            if vals:
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
    ctk.CTkSwitch(cv2, text="بازیابی محتوای قبلی کلیپ‌بورد بعد از درج",
                  variable=var_restore, **switch_style).pack(anchor="e", pady=(0, 2))
    dim(cv2, "اگر غیرفعال شود، متن دیکته در کلیپ‌بورد می‌ماند")
    var_commands = tk.BooleanVar(value=bool(cfg.get("voice_commands")))
    ctk.CTkSwitch(cv2, text="فرمان‌های صوتی", variable=var_commands,
                  **switch_style).pack(anchor="e", pady=(8, 0))
    dim(cv2, "نقطه، ویرگول، علامت سوال، گیومه باز/بسته، نقطه ویرگول، خط جدید، حذف آخرین کلمه")
    var_itn = tk.BooleanVar(value=bool(cfg.get("persian_itn")))
    ctk.CTkSwitch(cv2, text="تبدیل اعداد حروفی به رقم", variable=var_itn,
                  **switch_style).pack(anchor="e", pady=(8, 0))
    dim(cv2, "«بیست و سه» → ۲۳ — اعداد تکی مثل «یک» حروفی می‌مانند")

    # ================= تب پیشرفته =================
    t_adv = tabview.tab("پیشرفته")

    ca = card(t_adv, "پردازش")
    arow = ctk.CTkFrame(ca, fg_color="transparent")
    arow.pack(fill="x")
    ctk.CTkLabel(arow, text="تعداد thread پردازش مدل (با ری‌استارت اعمال می‌شود):",
                 font=(fam, 13), text_color=theme.FG, anchor="e").pack(side="right")
    var_threads = tk.StringVar(value=str(int(cfg.get("num_threads") or 4)))
    ctk.CTkOptionMenu(arow, values=[str(i) for i in range(1, 9)], variable=var_threads,
                      width=80, height=34, **menu_style).pack(side="left")

    cm_info = card(t_adv, "درباره موتور تشخیص")
    ctk.CTkLabel(cm_info, text="Shenava-Koochik v1.0", font=(fam, 13, "bold"),
                 text_color=theme.FG, anchor="e").pack(fill="x")
    dim(cm_info, "FastConformer NeMo CTC — ۱۱۴M پارامتر — کاملاً آفلاین")
    dim(cm_info, "کیفیت روی جملات دیکته‌شده بهتر از مکالمه آزاد است")

    # ================= تب راهنما (اسکرول‌شونده — محتوای بلند) =================
    t_help = tabview.tab("راهنما")
    t_help = ctk.CTkScrollableFrame(
        t_help, fg_color="transparent", scrollbar_fg_color="transparent",
        scrollbar_button_color=theme.SURFACE_3,
        scrollbar_button_hover_color=theme.SURFACE_2,
    )
    t_help.pack(fill="both", expand=True)

    ch = card(t_help, f"وِیس‌فلو فارسی — نسخه {APP_VERSION}")
    dim(ch, "دیکته صوتی فارسی، کاملاً آفلاین — مثل Wispr Flow")
    dim(ch, "مدل: Shenava-Koochik v1.0 (sherpa-onnx) — هیچ داده‌ای از سیستم شما خارج نمی‌شود")

    c1 = card(t_help, "استفاده سریع")
    dim(c1, "۱ — در هر برنامه‌ای (نوت‌پد، تلگرام، مرورگر…) کلید میانبر را بزن")
    dim(c1, "۲ — صحبت کن؛ پنجره زنده کنار موس متن را همزمان نشان می‌دهد")
    dim(c1, "۳ — همان کلید را دوباره بزن تا متن در محل کرسر درج شود")

    c2 = card(t_help, "فرمان‌های صوتی")
    dim(c2, "بگو «نقطه» یا «ویرگول» یا «علامت سوال» تا نشانه درج شود")
    dim(c2, "«گیومه باز» و «گیومه بسته» برای « »، «نقطه ویرگول» برای ؛")
    dim(c2, "«خط جدید» کلید Enter را می‌زند")
    dim(c2, "«حذف آخرین کلمه» آخرین کلمه درج‌شده را پاک می‌کند")

    c3 = card(t_help, "نکته‌ها")
    dim(c3, "میان‌بر با suppress ثبت می‌شود؛ اگر با میان‌بر برنامه‌ای تداخل داشت از تب عمومی عوضش کن")
    dim(c3, "در برنامه‌های run-as-administrator درج کار نمی‌کند (محدودیت ویندوز)")
    dim(c3, "اگر دستگاه ورودی را عوض کردی، در تب میکروفون انتخاب یا «خودکار» را نگه دار")
    dim(c3, "اعداد حروفی («بیست و سه») خودکار به رقم (۲۳) تبدیل می‌شوند — از تب درج متن خاموشش کن")

    # ================= دکمه‌های ثابت پایین =================
    btn_bar = tk.Frame(win, bg=theme.BG, padx=18, pady=8)
    btn_bar.pack(fill="x", side="bottom")

    def _sync(data: dict):
        """بارگذاری مقادیر روی ویجت‌ها — برای بازنشانی."""
        nonlocal prev_hotkey
        var_hotkey.set(data.get("hotkey"))
        prev_hotkey = data.get("hotkey")
        hk_hint.configure(text=HINT_TXT, text_color=theme.FG_DIM)
        dev_combo.set(auto_label)
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
        if hotkey_suspended and app is not None:
            app.resume_hotkey()
        win.destroy()

    ctk.CTkButton(btn_bar, text="بازنشانی", font=(fam, 13), height=40,
                  width=100, corner_radius=8, fg_color=theme.SURFACE_2,
                  hover_color=theme.SURFACE_3, text_color=theme.FG,
                  command=reset).pack(side="left")
    ctk.CTkButton(btn_bar, text="ذخیره", font=(fam, 13, "bold"), height=40,
                  width=130, corner_radius=8, fg_color=theme.ACCENT,
                  hover_color=theme.ACCENT_HOVER, text_color=theme.ON_ACCENT,
                  command=save).pack(side="right", padx=(8, 0))
    ctk.CTkButton(btn_bar, text="انصراف", font=(fam, 13), height=40,
                  width=110, corner_radius=8, fg_color=theme.SURFACE_2,
                  hover_color=theme.SURFACE_3, text_color=theme.FG,
                  command=close).pack(side="right")

    win.protocol("WM_DELETE_WINDOW", close)
    smooth_show(win)  # نمایش نرم بعد از ساخت کامل محتوا — بدون فریم سفید
