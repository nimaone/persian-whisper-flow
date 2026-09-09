"""تم دیکته‌یار برای Flet — بازتاب یک‌به‌یک app/theme.py.

قاعده همان است: سلسله‌مراتب با وزن/اندازه فونت، فقط دو رنگ عملکردی
(سبز accent و قرمز ضبط). فونت وزیرمتن از assets/fonts بار می‌شود.
"""
from __future__ import annotations

from pathlib import Path

import flet as ft

# ---------- پالت (قرینه‌ی app/theme.py) ----------
BG = "#1e1e1e"          # پس‌زمینه‌ی پنجره‌ها
SURFACE = "#2a2a2a"     # کارت
SURFACE_2 = "#333333"   # ورودی/دکمه‌ی ثانویه
SURFACE_3 = "#3c3c3c"   # hover ثانویه
BORDER = "#3f3f3f"
DEEP = "#181818"        # نواحی گرافیکی (اسپیکتر/موج)

FG = "#f3f3f3"
FG_DIM = "#9b9b9b"

ACCENT = "#22c55e"
ACCENT_HOVER = "#1eb457"
ON_ACCENT = "#081409"

DANGER = "#e5484d"
DANGER_HOVER = "#cd2d31"
ON_DANGER = "#ffffff"

WARN = "#d9a13c"

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

_installed = False


def install_fonts(page: ft.Page) -> None:
    """ثبت وزیرمتن برای پروسه — Flet کلاینت را با فونت آشنا می‌کند."""
    global _installed
    if _installed:
        return
    if page.fonts is None:
        page.fonts = {}
    for w in ("Regular", "Bold", "Medium", "Light"):
        p = _FONT_DIR / f"Vazirmatn-{w}.ttf"
        if p.exists():
            page.fonts["Vazirmatn-" + w] = str(p)
    _installed = True


def fam(weight: str = "Regular", size: int = 13) -> ft.TextStyle:
    """استایل متن وزیرمتن — قرینه‌ی (fam, size[, weight]) در CTk."""
    return ft.TextStyle(
        font_family="Vazirmatn-" + ("Bold" if weight == "bold" else weight),
        size=size,
        weight=ft.FontWeight.NORMAL,
    )


def apply_icon(page: ft.Page) -> None:
    """آیکون نوار عنوان/تسک‌بار از assets/logo.ico — قرینه‌ی theme.apply_icon.

    Window.icon در ویندوز مسیر .ico می‌گیرد؛ بی‌صدا رد می‌شود اگر نبود.
    """
    ico = _FONT_DIR.parent / "logo.ico"
    if ico.exists():
        page.window.icon = str(ico)


# ---------- هلپرهای مشترک ----------

def card(title: str | None, *controls, padding: int = 12) -> ft.Container:
    """کارت با عنوان اختیاری — قرینه‌ی card() در settings_ui.py."""
    inner: list[ft.Control] = []
    if title:
        inner.append(ft.Container(
            ft.Text(title, style=fam("bold", 13), color=FG),
            padding=ft.Padding(left=0, top=0, right=0, bottom=8),
        ))
    inner.extend(controls)
    return ft.Container(
        ft.Column(inner, spacing=0),
        bgcolor=SURFACE,
        border_radius=10,
        padding=padding,
        expand=True,
    )


def dim(text: str) -> ft.Container:
    """متن توضیح کم‌رنگ راست‌چین — قرینه‌ی dim()."""
    return ft.Container(
        ft.Text(text, style=fam("Regular", 11), color=FG_DIM, text_align=ft.TextAlign.RIGHT),
        alignment=ft.Alignment(1, 0),
        padding=ft.Padding(left=0, top=4),
    )


def row_label(text: str) -> ft.Container:
    """برچسب ردیف — راست‌چین."""
    return ft.Container(
        ft.Text(text, style=fam("Regular", 13), color=FG, text_align=ft.TextAlign.RIGHT),
        alignment=ft.Alignment(1, 0),
        expand=True,
    )


def btn_style(radius: int = 8, size: int = 13, weight: str = "Regular",
              hpad: int = 14) -> ft.ButtonStyle:
    """استایل دکمه — hover در Flet از طریق overlay_color (روی bgcolor)."""
    return ft.ButtonStyle(
        shape=ft.RoundedRectangleBorder(radius=radius),
        padding=ft.Padding(left=hpad, top=8, right=hpad, bottom=8),
        text_style=fam(weight, size),
        elevation=0,
    )


def secondary_button(text: str, on_click=None, height: int = 36, width: int | None = None,
                     bold: bool = False) -> ft.ElevatedButton:
    """دکمه‌ی ثانویه — قرینه‌ی دکمه‌های SURFACE_2/SURFACE_3."""
    return ft.ElevatedButton(
        text=text,
        on_click=on_click,
        height=height,
        width=width,
        bgcolor=SURFACE_2,
        color=FG,
        style=btn_style(weight="bold" if bold else "Regular"),
    )


def switch(label: str | None, value: bool = False, on_change=None) -> ft.Row:
    """سوییچ هم‌اندازه با ریل — قرینه‌ی switch_style با border_width=0."""
    sw = ft.Switch(
        value=value,
        active_color=ACCENT,
        on_change=on_change,
        scale=0.9,
    )
    r = ft.Row([], spacing=10)
    if label:
        r.controls.append(row_label(label))
    r.controls.append(sw)
    return r


def radio(label: str, group_value: str, value: str, on_change=None) -> ft.Radio:
    """رادیو — قرینه‌ی radio_style."""
    return ft.Radio(
        value=value,
        group_value=group_value,
        label=label,
        active_color=ACCENT,
        fill_color=ACCENT,
        label_style=fam("Regular", 13),
        label_position=ft.LabelPosition.LEFT,
        on_change=on_change,
    )


def dropdown(values: list[str], value: str, on_change=None, width: int | None = None,
             height: int = 34) -> ft.Dropdown:
    """دراپ‌داون — قرینه‌ی CTkOptionMenu با menu_style.

    نکته: ft.dropdown.Option(«متن») مقدار را فقط در key می‌گذارد و
    label خالی می‌ماند — پس key و text را صریح می‌دهیم.
    """
    return ft.Dropdown(
        options=[ft.dropdown.Option(key=v, text=v) for v in values],
        value=value,
        on_select=on_change,
        width=width,
        bgcolor=SURFACE_2,
        border_color=BORDER,
        focused_border_color=ACCENT,
        border_radius=8,
        dense=True,
        expand=width is None,
        text_style=fam("Regular", 13),
        scale=0.9,
    )
