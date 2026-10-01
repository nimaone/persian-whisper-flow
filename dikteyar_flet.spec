# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — دیکته‌یار نسخه فلت (onedir + windowed).

بیلد:
    .venv/Scripts/python.exe -m PyInstaller dikteyar_flet.spec --noconfirm

خروجی: dist/DikteYarFlet/  (DikteYarFlet.exe + _internal)

پیش‌نیاز: کلاینت دسکتاپ فلت — packaging/flet-windows.zip (نسخه هم‌خانواده
با flet نصب‌شده؛ اینجا 1.0.3). دانلود:
    https://github.com/flet-dev/flet/releases/download/v1.0.3/flet-windows.zip
این zip به flet_desktop/app/ باندل میشود تا در اولین اجرای سیستم مقصد
کلاینت از اینترنت دانلود نشود (فلت در نبودش به CDN میرود).

نکته‌ها:
- onedir نه onefile: استارت فوری، بدون اکسترکت به temp؛ هوک کیبورد و
  آنتی‌ویروس با onefile بد رفتار می‌کنند.
- کلاینت فلت در اولین اجرا از zip به ~/.flet/client اکسترکت میشود —
  یک بار برای هر کاربر، بدون اینترنت.
- مدل (model.onnx) عمداً باندل نمیشود — دیالوگ اولین اجرا دانلود میکند.
- tkinter حذف نمیشود: پنجره متن زنده (app/overlay.py) در هر دو UI
  با Tk کار میکند.
"""

a = Analysis(
    ["flet_ui\\run.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("assets", "assets"),                          # لوگو + فونت وزیرمتن
        ("packaging\\flet-windows.zip", "flet_desktop\\app"),
    ],
    hiddenimports=[
        "pystray._win32",               # بک‌اند ویندوز pystray (import دینامیک)
        "flet_desktop",                 # import دینامیک در flet.app
        "onnxruntime",                  # هات‌وورد اختیاری
        "pyctcdecode",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "customtkinter",                # فقط UI قبلی — در بسته فلت اضافی
        "tkinter.test", "test", "unittest", "pydoc_data",
        "matplotlib", "pandas", "scipy", "IPython", "jedi",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DikteYarFlet",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                      # بدون کنسول سیاه
    icon="assets\\logo.ico",
    version="version_info_flet.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DikteYarFlet",
)
