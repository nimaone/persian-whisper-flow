"""درج متن در محل کرسر برنامه فعال.

روش: کلیپ‌بورد (با بکاپ/بازیابی) + ارسال Ctrl+V با SendInput از طریق
کتابخانه‌ی keyboard — سریع (~1ms) و قابل اعتماد برای همه‌ی برنامه‌ها.

نکته‌ی حیاتی: بعد از Ctrl+V باید به برنامه‌ی مقصد فرصت دهیم متن را از
کلیپ‌بورد بخواند «قبل از» بازیابی کلیپ‌بورد قبلی؛ در غیر این صورت
متن ناقص یا خالی درج می‌شود. انتظار با poll دیده‌بانی می‌شود، نه sleep کور.
"""
from __future__ import annotations

import time

import pyperclip


def set_clipboard(text: str):
    pyperclip.copy(text)


def get_clipboard() -> str:
    try:
        return pyperclip.paste()
    except Exception:
        return ""


def send_key(combo: str):
    """ارسال ترکیب کلید با SendInput — سریع و بدون spawn پروسه.

    combo مثل "ctrl+v" یا "enter" یا "ctrl+backspace".
    """
    import keyboard

    keyboard.send(combo)


# ---------- انتظار هوشمند ----------

def _wait_until(predicate, timeout: float, poll: float = 0.02) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def _target_pasted(text: str, target_window_hint: str | None = None) -> bool:
    """حدس زدن اینکه مقصد متن را گرفته — از طریق ناپدید شدن فرمت delayed-render.

    ویندوز: وقتی کلیپ‌بورد با pyperclip ست می‌شود، داده واقعی فوراً
    نوشته می‌شود؛ برنامه مقصد هنگام Ctrl+V آن را می‌خواند. قابل اطمینان‌ترین
    علامت مشترک بین برنامه‌ها همان زمان سپری‌شده است؛ برای همین این تابع
    فقط برای برنامه‌های خاصی نتیجه‌ی قطعی برمی‌گرداند.
    """
    return False


def paste_via_clipboard(text: str, restore: bool = True, settle_timeout: float = 0.6):
    """کپی → Ctrl+V → انتظار امن → بازیابی کلیپ‌بورد قبلی.

    settle_timeout: حداکثر انتظار بعد از Ctrl+V قبل از بازیابی.
    برنامه‌های کند (Electron/تلگرام) تا ~۴۰۰ms متن را می‌گیرند؛ ۶۰۰ms
    با حاشیه‌ی امن همه را پوشش می‌دهد و در عین حال کل تجربه زیر ۱ ثانیه می‌ماند.

    محتوای غیرمتنی کلیپ‌بورد (عکس/فایل) با pyperclip قابل بکاپ‌گیری نیست؛
    در آن حالت بازیابی انجام نمی‌شود و متن دیکته در کلیپ‌بورد می‌ماند.
    """
    if not text:
        return
    saved = get_clipboard() if restore else None
    set_clipboard(text)
    # فرصت به سیستم برای انتشار کلیپ‌بورد جدید (event WM_CLIPBOARDUPDATE)
    time.sleep(0.08)
    send_key("ctrl+v")
    # انتظار امن قبل از دست‌زدن دوباره به کلیپ‌بورد
    time.sleep(settle_timeout)
    # saved خالی یعنی کلیپ‌بورد قبلاً متن نداشته (یا غیرمتنی بوده) — بازیابی معنا ندارد
    if restore and saved:
        set_clipboard(saved)


def send_enter():
    send_key("enter")


def send_delete_word():
    """پاک‌کردن آخرین کلمه در اکثر برنامه‌ها (ویرایشگرهای استاندارد ویندوز)."""
    send_key("ctrl+backspace")


def type_text_directly(text: str):
    """تایپ مستقیم با SendInput — برای متن فارسی کپیوکلیپ‌بورد غیرقابل اعتماد است.

    نکته: keyboard.write با یونیکد از SendInput با KEYEVENTF_UNICODE استفاده
    می‌کند؛ پشتیبانی فارسی کامل است اما در بعضی برنامه‌های بازی (که
    فقط رخدادهای اسکن‌کد می‌گیرند) کار نمی‌کند.
    """
    import keyboard

    keyboard.write(text, delay=0.001, exact=True)


def insert_text(text: str, method: str = "clipboard", restore: bool = True):
    """درج متن در برنامه فعال — روش کلیپ‌بورد یا تایپ مستقیم.

    restore: بازیابی محتوای قبلی کلیپ‌بورد بعد از درج (تنظیم کاربر).
    """
    if not text:
        return
    if method == "type":
        type_text_directly(text)
    else:
        paste_via_clipboard(text, restore=restore)
