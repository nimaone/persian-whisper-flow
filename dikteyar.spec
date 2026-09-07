# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — دیکته‌یار (onedir + windowed).

بیلد:
    .venv/Scripts/python.exe -m PyInstaller dikteyar.spec --noconfirm

خروجی: dist/DikteYar/  (DikteYar.exe + _internal)

نکته‌ها:
- onedir نه onefile: استارت فوری، بدون اکسترکت به temp؛ هوک کیبورد و
  آنتی‌ویروس با onefile بد رفتار می‌کنند.
- onnxruntime برای حالت اختیاری هات‌وورد باندل می‌شود (~۹۰MB) — اگر
  خواستید بسته سبک‌تر شود، hiddenimports آن و collect آن را حذف کنید.
- مدل (model.onnx) عمداً باندل نمی‌شود — دیالوگ اولین اجرا دانلود می‌کند.
"""

a = Analysis(
    ["app\\main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("assets", "assets"),           # لوگو + فونت وزیرمتن
    ],
    hiddenimports=[
        "pystray._win32",               # بک‌اند ویندوز pystray (import دینامیک)
        "onnxruntime",                  # هات‌وورد اختیاری
        "pyctcdecode",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
    name="DikteYar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                      # بدون کنسول سیاه
    icon="assets\\logo.ico",
    version="version_info.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DikteYar",
)
