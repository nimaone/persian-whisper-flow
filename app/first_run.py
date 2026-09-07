"""دیالوگ اولین اجرا — وقتی فایل‌های مدل کنار exe نیستند.

در بسته‌ی نصب‌شده (نصب‌کننده سبک) مدل همراه اپ نیست. کاربر دو راه دارد:
  «دانلود خودکار»  — ~۴۳۸MB از Release/HF با نوار پیشرفت
  «انتخاب دستی»    — پوشه‌ای که خودش فایل‌ها را در آن گذاشته (اینترنت ایران)

در سورس (بدون frozen) این دیالوگ اجرا نمی‌شود — توسعه‌دهنده مدل را
خودش در model/ پروژه می‌گذارد.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from app import model_download, theme
from app.config import APP_TITLE
from app.paths import install_root


def ensure_model() -> bool:
    """اگر مدل کامل است True. وگرنه دیالوگ می‌دهد و پس از تدارک True.

    خروج False یعنی کاربر انصراف داده و اپ باید ببندد.
    """
    from app.config import model_dir

    md = model_dir()
    if model_download.is_complete(md):
        return True
    if not getattr(__import__("sys"), "frozen", False):
        # سورس: بدون دیالوگ — همان پیام CLI
        print(f"[model] فایل‌های مدل در {md} نیست — python download_model.py")
        return False
    return _dialog(md)


def _dialog(md) -> bool:
    done_evt = threading.Event()
    result: list[bool] = [False]

    win = ctk.CTk()
    win.title(APP_TITLE)
    win.geometry("480x300")
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
        text=("مدل تشخیص گفتار (~۴۴۰MB) همراه نصب‌کننده نیست تا حجم دانلود کم بماند.\n"
              "الان دانلودش می‌کنم، یا اگر خودتان فایل‌ها را دارید مسیرشان را بدهید."),
        font=(fam, 12), text_color=theme.FG_DIM,
    ).pack(fill="x", pady=(8, 12))

    prog = ctk.CTkProgressBar(c)
    prog.set(0)
    prog.pack(fill="x", pady=(0, 4))
    status = ctk.CTkLabel(c, text="", font=(fam, 11), text_color=theme.FG_DIM,
                          anchor="e", justify="right")
    status.pack(fill="x")

    def set_progress(name: str, done: int, total: int):
        pct = done / total if total else 0
        win.after(0, lambda: (prog.set(pct),
                              status.configure(text=f"{name} — {int(pct * 100)}٪")))

    def start_download():
        dl_btn.configure(state="disabled")
        pick_btn.configure(state="disabled")

        def worker():
            ok, msg = model_download.download_model(md, set_progress)
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
        folder = filedialog.askdirectory(
            title="پوشه‌ای که model.onnx و tokens.txt در آن است را انتخاب کنید")
        if not folder:
            return
        import shutil
        from pathlib import Path
        src = Path(folder)
        missing = [n for n in model_download.FILE_NAMES if not (src / n).exists()]
        if missing:
            status.configure(text=f"در این پوشه {', '.join(missing)} نیست", text_color=theme.DANGER)
            return
        for n in model_download.FILE_NAMES:
            target = md / n
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != md.resolve():
                shutil.copy2(src / n, target)
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
