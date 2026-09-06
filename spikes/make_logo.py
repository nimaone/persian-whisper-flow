"""تولید لوگوی وِیس‌فلو فارسی — میکروفون سبز روی زمینه تیره گوشه‌گرد.

خروجی: assets/logo.png (256px) و assets/logo.ico (چنداندازه برای نوار عنوان).
اجرا: python spikes/make_logo.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

S = 512
GREEN = (34, 197, 94, 255)     # #22c55e — همان accent تم
LIGHT = (232, 232, 238, 255)   # گهواره/ساقه/پایه
BG = (30, 30, 30, 255)         # #1e1e1e
BORDER = (63, 63, 63, 255)     # #3f3f3f

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def make_logo() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # زمینه‌ی گوشه‌گرد
    d.rounded_rectangle((16, 16, 496, 496), radius=112,
                        fill=BG, outline=BORDER, width=6)

    # کپسول میکروفون
    d.rounded_rectangle((196, 112, 316, 292), radius=60, fill=GREEN)
    # هایلایت ظریف داخل کپسول
    d.rounded_rectangle((216, 140, 244, 264), radius=14,
                        fill=(255, 255, 255, 40))

    # گهواره‌ی U شکل
    d.arc((156, 152, 356, 352), start=0, end=180, fill=LIGHT, width=24)
    # ساقه
    d.rounded_rectangle((244, 344, 268, 404), radius=12, fill=LIGHT)
    # پایه
    d.rounded_rectangle((180, 404, 332, 428), radius=12, fill=LIGHT)
    return img


def main():
    ASSETS.mkdir(exist_ok=True)
    logo = make_logo()
    png = logo.resize((256, 256), Image.LANCZOS)
    png.save(ASSETS / "logo.png")
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)]
    png.save(ASSETS / "logo.ico", sizes=ico_sizes)
    print("saved:", ASSETS / "logo.png", "and", ASSETS / "logo.ico")


if __name__ == "__main__":
    main()
