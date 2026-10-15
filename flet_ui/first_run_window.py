"""صفحه‌ی «اولین اجرا» در UI فلت — قرینه‌ی app/first_run.py بدون Tk.

اگر فایل‌های مدل سر جایشان نباشند، نسخه‌ی CTk یک دیالوگ Tk باز می‌کند.
در فلت همان جریان داخل خود پنجره پیاده شده است، با دو گزینه‌ی دستی:
  «دانلود خودکار»  — ~۴۳۸MB از Release گیت‌هاب با فالبک HuggingFace
  «انتخاب دستی»    — پوشه‌ای که کاربر خودش فایل‌ها را در آن گذاشته

تفاوت عمدی با نسخه‌ی CTk: آنجا دیالوگ فقط در حالت frozen اجرا می‌شود و
در سورس فقط یک پیام CLI چاپ می‌شود؛ اینجا صفحه در هر دو حالت نمایش
داده می‌شود — UI فلت اپ گرافیکی است و پیام چاپی جای گزینه‌های واقعی را
نمی‌گیرد.
"""
from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import flet as ft

from app import models_catalog, model_download
from app.config import APP_TITLE, model_dir
from flet_ui import theme as t

TITLE = "فایل‌های مدل یافت نشد"
BODY = ("مدل تشخیص گفتار همراه نصب‌کننده نیست تا حجم دانلود کم بماند.\n"
        "یک مدل را انتخاب و دانلود کنید، یا اگر فایل‌ها را دارید مسیرشان را بدهید.")
PICK_TITLE = "پوشه‌ای که فایل‌های مدل در آن است را انتخاب کنید"


def active_model_spec():
    """مدل فعال از تنظیمات — ناشناخته/خطا → مدل پیش‌فرض."""
    try:
        from app.config import Config
        return models_catalog.get(Config.load().get("model_id"))
    except Exception:
        return models_catalog.default_model()


def active_model_dir() -> Path:
    """پوشه مدل فعال — تنظیم دلخواه کاربر اگر باشد، وگرنه پیشفرض."""
    try:
        from app.config import Config
        return model_dir(Config.load().get("model_dir"))
    except Exception:
        return model_dir()


def model_ready(md: Path | None = None, spec=None) -> bool:
    """آیا فایل‌های مدل فعال کنار پروژه/نصب موجودند؟"""
    d = Path(md) if md is not None else active_model_dir()
    return model_download.is_complete(d, spec or active_model_spec())


class FirstRunWindow:
    """جایگزین پنجره‌ی کنترل تا وقتی مدل آماده شود.

    on_ready پس از آماده‌شدن مدل (دانلود یا انتخاب دستی) صدا زده می‌شود؛
    run.py از همان‌جا اپ اصلی را بالا می‌آورد.
    """

    def __init__(self, page: ft.Page, on_ready, on_quit=None,
                 download_fn=None, pick_dir_fn=None, md: Path | None = None):
        self.page = page
        self.on_ready = on_ready
        self.on_quit = on_quit
        self.md = Path(md) if md is not None else active_model_dir()
        self._sel = active_model_spec()
        # تزریق وابستگی — تست بدون شبکه و بدون دیالوگ سیستم.
        # امضای download_fn قرینه‌ی قبل (md, progress_cb) می‌ماند؛ مدل فعال
        # از self._sel خوانده می‌شود.
        self._download = download_fn or (
            lambda m, cb: model_download.download_for(m, self._sel, cb))
        self._pick_dir = pick_dir_fn  # sync: () -> str | None (تست)
        self._busy = False
        self.result = False
        self._last_pct = -1
        self._build()

    def _persist_model_id(self):
        """ذخیره‌ی مدل انتخاب‌شده — اپ در اجرای همین فرآیند می‌خواند."""
        try:
            from app.config import Config
            cfg = Config.load()
            cfg.set("model_id", self._sel.id)
            cfg.save()
        except Exception:
            pass

    def _model_label(self, spec) -> str:
        badge = "  (پیشنهادی)" if spec.recommended else ""
        return f"{spec.name}{badge} — {spec.approx_size} • {spec.speed}، {spec.accuracy}"

    def _on_model_change(self, e):
        self._sel = models_catalog.get(e.control.value)

    # ---------- ساخت صفحه ----------
    def _build(self):
        self.page.title = APP_TITLE
        self.page.bgcolor = t.BG
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.rtl = True
        self.page.padding = 24
        try:
            self.page.window.width = 500
            self.page.window.height = 400
        except Exception:
            pass

        self.head = ft.Text(TITLE, style=t.fam("bold", 16), color=t.FG,
                            text_align=ft.TextAlign.RIGHT)
        self.body = ft.Text(BODY, style=t.fam("Regular", 12), color=t.FG_DIM,
                            text_align=ft.TextAlign.RIGHT)
        self.model_group = ft.RadioGroup(
            content=ft.Column(
                [ft.Radio(value=s.id, label=self._model_label(s),
                          label_style=t.fam("Regular", 12), active_color=t.ACCENT)
                 for s in models_catalog.all_models()],
                spacing=2,
            ),
            value=self._sel.id, on_change=self._on_model_change,
        )
        self.prog = ft.ProgressBar(value=0, color=t.ACCENT, bgcolor=t.SURFACE_2,
                                   height=6, border_radius=3)
        self.status = ft.Text("", style=t.fam("Regular", 11), color=t.FG_DIM,
                              text_align=ft.TextAlign.RIGHT)
        self.dl_btn = ft.Button(
            content="دانلود خودکار", on_click=self._fire_download, height=40,
            bgcolor=t.ACCENT, color=t.ON_ACCENT,
            style=t.btn_style(weight="bold", hpad=18),
        )
        self.pick_btn = ft.Button(
            content="انتخاب دستی فایل‌ها…", on_click=self._pick_folder, height=40,
            bgcolor=t.SURFACE_2, color=t.FG, style=t.btn_style(hpad=18),
        )
        self.quit_btn = ft.Button(
            content="خروج", on_click=self._fire_quit, height=40,
            bgcolor=t.SURFACE_2, color=t.FG_DIM, style=t.btn_style(hpad=18),
        )
        btns = ft.Row([self.dl_btn, self.pick_btn, self.quit_btn],
                      alignment=ft.MainAxisAlignment.START, spacing=8)

        try:
            self.page.controls.clear()
        except Exception:
            pass  # MockPage بدون controls
        self.page.add(
            ft.Column([self.head, self.body], spacing=8),
            ft.Container(self.model_group, padding=ft.Padding(left=0, top=6, right=0, bottom=6)),
            self.prog,
            self.status,
            ft.Container(btns, padding=ft.Padding(left=0, top=10)),
        )
        self._flush_sync()

    # ---------- به‌روزرسانی UI (از هر ترد) ----------
    def _flush_sync(self):
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            run_task(self._flush)
        else:
            try:
                self.page.update()  # MockPage
            except Exception:
                pass

    async def _flush(self):
        try:
            self.page.update()
        except Exception:
            pass  # پنجره بسته شده

    # ---------- دانلود خودکار ----------
    def _fire_download(self, e=None):
        if self._busy:
            return
        self._busy = True
        self.dl_btn.disabled = True
        self.pick_btn.disabled = True
        self.status.value = "شروع دانلود…"
        self.status.color = t.FG_DIM
        self._last_pct = -1
        self._flush_sync()
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self):
        try:
            ok, msg = self._download(self.md, self._on_progress)
        except Exception as ex:  # شبکه/دیسک — اپ نباید بیفتد
            ok, msg = False, str(ex)
        if ok:
            self._persist_model_id()
            self._succeed("دانلود کامل شد ✓")
        else:
            self._busy = False
            self.dl_btn.disabled = False
            self.pick_btn.disabled = False
            self.status.value = f"{str(msg)[:70]} — دوباره تلاش کنید یا دستی انتخاب کنید"
            self.status.color = t.DANGER
            self._flush_sync()

    def _on_progress(self, name: str, done: int, total: int):
        """از ترد دانلود — فقط وقتی درصد عوض شود پچ می‌فرستیم (≤۱۰۰ ارسال)."""
        pct = int(done / total * 100) if total else 0
        if pct == self._last_pct:
            return
        self._last_pct = pct
        self.prog.value = done / total if total else 0
        self.status.value = f"{name} — {pct}٪"
        self._flush_sync()

    # ---------- انتخاب دستی پوشه ----------
    async def _pick_dir_dialog(self) -> str | None:
        """دیالوگ انتخاب پوشه — انتخابگر باید اول به صفحه وصل شود
        (page.services) وگرنه get_directory_path خطای «کنترل ثبتنشده»
        می‌دهد."""
        picker = ft.FilePicker()
        try:
            self.page.services.append(picker)
            self.page.update()
            return await picker.get_directory_path(dialog_title=PICK_TITLE)
        except Exception as ex:
            self._fail(f"انتخاب پوشه ممکن نشد: {str(ex)[:50]}")
            return None
        finally:
            try:
                self.page.services.remove(picker)
            except Exception:
                pass

    async def _pick_folder(self, e=None):
        if self._busy:
            return
        folder = None
        if self._pick_dir is not None:
            folder = self._pick_dir()
        else:
            folder = await self._pick_dir_dialog()
        if not folder:
            return  # کاربر انصراف داد
        self._apply_folder(Path(folder))

    def _apply_folder(self, src: Path) -> bool:
        """اعتبارسنجی و کپی فایل‌های پوشه‌ی انتخابی — قرینه‌ی pick_folder در CTk."""
        names = [f.name for f in self._sel.files]
        missing = [n for n in names if not (src / n).exists()]
        if missing:
            self._fail(f"در این پوشه {', '.join(missing)} نیست")
            return False
        target_dir = self._sel.dir(self.md)
        target_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            target = target_dir / n
            if src.resolve() != target_dir.resolve():
                shutil.copy2(src / n, target)
        self._persist_model_id()
        self._succeed("فایل‌ها آماده شد ✓")
        return True

    # ---------- پایان/خطا ----------
    def _succeed(self, msg: str):
        self.result = True
        self.prog.value = 1
        self.status.value = msg
        self.status.color = t.ACCENT
        self.dl_btn.disabled = True
        self.pick_btn.disabled = True
        self._flush_sync()
        # مکث کوتاه تا پیام دیده شود، بعد اپ اصلی بالا بیاید
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            run_task(self._ready_soon)
        else:
            self.on_ready()

    async def _ready_soon(self):
        await asyncio.sleep(0.6)
        self.on_ready()

    def _fail(self, msg: str):
        self.status.value = msg
        self.status.color = t.DANGER
        self._flush_sync()

    def _fire_quit(self, e=None):
        if self.on_quit:
            self.on_quit()
            return
        try:
            self.page.window.destroy()
        except Exception:
            pass
