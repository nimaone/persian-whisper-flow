"""دیالوگ اولین اجرا — وقتی فایل‌های مدل فعال کنار exe نیستند.

در بسته‌ی نصب‌شده (نصب‌کننده سبک) مدل همراه اپ نیست. کاربر مدل را انتخاب
می‌کند (شنوا کوچیک پیشنهادی / نیموترون دقیق‌تر) و دو راه دارد:
  «دانلود خودکار»  — از Release/HF با نوار پیشرفت
  «انتخاب دستی»    — پوشه‌ای که خودش فایل‌ها را در آن گذاشته (اینترنت ایران)

انتخاب مدل پس از موفقیت در تنظیمات ذخیره می‌شود تا اپ همان را بار کند.
در سورس (بدون frozen) این دیالوگ اجرا نمی‌شود — توسعه‌دهنده مدل را
خودش در پوشه‌ی مدل می‌گذارد.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from app import models_catalog, model_download, theme
from app.config import APP_TITLE


def ensure_model() -> bool:
    """اگر مدل فعال کامل است True. وگرنه دیالوگ می‌دهد و پس از تدارک True.

    خروج False یعنی کاربر انصراف داده و اپ باید ببندد.
    """
    from app.config import Config, model_dir

    cfg = Config.load()
    # پوشه‌ی دلخواهِ ست‌شده در تنظیمات — دانلود/کپی اول اجرا همان‌جا انجام می‌شود
    md = model_dir(cfg.get("model_dir"))
    spec = models_catalog.get(cfg.get("model_id"))
    if model_download.is_complete(md, spec):
        return True
    if not getattr(sys, "frozen", False):
        # سورس: بدون دیالوگ — همان پیام CLI
        print(f"[model] فایل‌های مدل در {spec.dir(md)} نیست — python download_model.py")
        return False
    return _dialog(md, spec)


def _persist_model_id(model_id: str):
    """ذخیره‌ی مدل انتخاب‌شده تا اپ در اجرای بعدِ همین فرآیند بخواند."""
    from app.config import Config

    try:
        cfg = Config.load()
        cfg.set("model_id", model_id)
        cfg.save()
    except Exception:
        pass


def _dialog(md, default_spec) -> bool:
    done_evt = threading.Event()
    result: list[bool] = [False]
    sel = [default_spec]

    win = ctk.CTk()
    win.title(APP_TITLE)
    win.geometry("500x430")
    win.resizable(False, False)
    win.configure(bg=theme.BG)
    win.attributes("-topmost", True)
    ctk.set_appearance_mode("dark")

    fam = theme.family()
    c = ctk.CTkFrame(win, fg_color="transparent")
    c.pack(fill="both", expand=True, padx=24, pady=20)

    ctk.CTkLabel(c, text="فایل‌های مدل یافت نشد", font=(fam, 16, "bold"),
                 text_color=theme.FG).pack(anchor="e")
    ctk.CTkLabel(
        c, justify="right", anchor="e",
        text=("مدل تشخیص گفتار همراه نصب‌کننده نیست تا حجم دانلود کم بماند.\n"
              "یک مدل را انتخاب و دانلود کنید، یا اگر فایل‌ها را دارید مسیرشان را بدهید."),
        font=(fam, 12), text_color=theme.FG_DIM,
    ).pack(fill="x", pady=(8, 12))

    var_choice = tk.StringVar(value=default_spec.id)

    def _label(spec) -> str:
        badge = "  (پیشنهادی)" if spec.recommended else ""
        return (f"{spec.name}{badge} — {spec.approx_size}\n"
                f"سرعت: {spec.speed} • دقت: {spec.accuracy}")

    for spec in models_catalog.all_models():
        ctk.CTkRadioButton(
            c, text=_label(spec), variable=var_choice, value=spec.id,
            font=(fam, 12), text_color=theme.FG, fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_HOVER, border_color=theme.SURFACE_3,
            command=lambda s=spec: sel.__setitem__(0, s),
        ).pack(anchor="e", fill="x", pady=(0, 6))

    prog = ctk.CTkProgressBar(c)
    prog.set(0)
    prog.pack(fill="x", pady=(6, 4))
    status = ctk.CTkLabel(c, text="", font=(fam, 11), text_color=theme.FG_DIM,
                          anchor="e", justify="right")
    status.pack(fill="x")

    def set_progress(name: str, done: int, total: int):
        pct = done / total if total else 0
        win.after(0, lambda: (prog.set(pct),
                              status.configure(text=f"{name} — {int(pct * 100)}٪")))

    def start_download():
        spec = sel[0]
        dl_btn.configure(state="disabled")
        pick_btn.configure(state="disabled")

        def worker():
            ok, msg = model_download.download_for(md, spec, set_progress)
            if ok:
                _persist_model_id(spec.id)
            result[0] = ok
            win.after(0, lambda: status.configure(
                text="دانلود کامل شد ✓" if ok else f"{msg} — دوباره تلاش کنید یا دستی انتخاب کنید",
                text_color=theme.ACCENT if ok else theme.DANGER))
            if ok:
                win.after(600, lambda: (win.destroy(), done_evt.set()))
            else:
                dl_btn.configure(state="normal")
                pick_btn.configure(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    def pick_folder():
        spec = sel[0]
        names = [f.name for f in spec.files]
        folder = filedialog.askdirectory(
            title=f"پوشه‌ای که {', '.join(names)} در آن است را انتخاب کنید")
        if not folder:
            return
        import shutil
        from pathlib import Path
        src = Path(folder)
        missing = [n for n in names if not (src / n).exists()]
        if missing:
            status.configure(text=f"در این پوشه {', '.join(missing)} نیست", text_color=theme.DANGER)
            return
        target = spec.dir(md)
        target.mkdir(parents=True, exist_ok=True)
        for n in names:
            if src.resolve() != target.resolve():
                shutil.copy2(src / n, target / n)
        _persist_model_id(spec.id)
        result[0] = True
        status.configure(text="فایل‌ها آماده شد ✓", text_color=theme.ACCENT)
        win.after(400, lambda: (win.destroy(), done_evt.set()))

    btns = ctk.CTkFrame(c, fg_color="transparent")
    btns.pack(fill="x", pady=(10, 0))
    dl_btn = ctk.CTkButton(btns, text="دانلود خودکار", font=(fam, 13, "bold"),
                           height=40, corner_radius=8, fg_color=theme.ACCENT,
                           hover_color=theme.ACCENT_HOVER,
                           text_color=theme.ON_ACCENT, command=start_download)
    dl_btn.pack(side="right", padx=(8, 0))
    pick_btn = ctk.CTkButton(btns, text="انتخاب دستی فایل‌ها…", font=(fam, 13),
                             height=40, corner_radius=8, fg_color=theme.SURFACE_2,
                             hover_color=theme.SURFACE_3, text_color=theme.FG,
                             command=pick_folder)
    pick_btn.pack(side="right")
    quit_btn = ctk.CTkButton(btns, text="خروج", font=(fam, 13), height=40,
                             width=80, corner_radius=8, fg_color=theme.SURFACE_2,
                             hover_color=theme.SURFACE_3, text_color=theme.FG_DIM,
                             command=lambda: (win.destroy(), done_evt.set()))
    quit_btn.pack(side="left")

    win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(), done_evt.set()))
    win.mainloop()
    done_evt.wait(timeout=2)
    return result[0]
