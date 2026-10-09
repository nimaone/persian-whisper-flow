"""جابه‌جایی کارت overlay با موس — آفست درگ + کلیپ روی کل دسکتاپ."""
from app.overlay import Overlay, clamp_pos, virtual_screen


class _FakeRoot:
    """جای root تکینتر — فقط چیزی که هندلرهای درگ لازم دارند."""

    def __init__(self, x=0, y=0, w=300, h=120):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.geo: list[str] = []

    def winfo_x(self):
        return self.x

    def winfo_y(self):
        return self.y

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h

    def geometry(self, spec):
        self.geo.append(spec)
        if spec.startswith("+"):
            parts = spec.split("+")
            self.x, self.y = int(parts[1]), int(parts[2])


def _make_overlay(screen=(0, 0, 1920, 1080), **root_kw):
    ov = object.__new__(Overlay)
    ov.root = _FakeRoot(**root_kw)
    ov._drag_off = None
    ov._manual_pos = None
    ov._screen_rect = lambda: screen
    return ov


class _Ev:
    def __init__(self, x_root, y_root):
        self.x_root, self.y_root = x_root, y_root


# ---------- clamp_pos ----------

def test_clamp_inside_screen_is_untouched():
    assert clamp_pos(100, 200, 300, 120, (0, 0, 1920, 1080)) == (100, 200)


def test_clamp_keeps_card_fully_inside():
    x, y = clamp_pos(5000, 4000, 300, 120, (0, 0, 1920, 1080))
    assert (x, y) == (1920 - 300 - 8, 1080 - 120 - 8)


def test_clamp_allows_negative_origin_second_monitor():
    """مانیتور چپ: مبدأ دسکتاپ منفی است — کلیپ باید همان را بپذیرد."""
    screen = (-1920, 0, 3840, 1080)
    assert clamp_pos(-1900, 100, 300, 120, screen) == (-1900, 100)
    assert clamp_pos(-5000, 100, 300, 120, screen) == (-1920 + 8, 100)
    assert clamp_pos(3000, 100, 300, 120, screen) == (3840 - 1920 - 300 - 8, 100)


def test_clamp_without_screen_returns_input():
    assert clamp_pos(9999, -9999, 300, 120, None) == (9999, -9999)


def test_virtual_screen_is_sane_or_none():
    rect = virtual_screen()
    if rect is not None:
        x, y, w, h = rect
        assert w > 0 and h > 0


# ---------- هندلرهای درگ ----------

def test_drag_keeps_grab_point_under_cursor():
    ov = _make_overlay(x=100, y=100)
    ov._drag_start(_Ev(200, 150))          # آفست = (۱۰۰، ۵۰)
    ov._drag_move(_Ev(600, 400))            # باید ۵۰۰،۳۵۰ شود
    assert ov.root.geo[-1] == "+500+350"
    assert ov._manual_pos == (500, 350)


def test_drag_clamped_to_desktop():
    ov = _make_overlay(x=100, y=100)
    ov._drag_start(_Ev(200, 150))
    ov._drag_move(_Ev(9000, 9000))
    assert ov.root.geo[-1] == f"+{1920 - 300 - 8}+{1080 - 120 - 8}"


def test_drag_without_press_is_ignored():
    ov = _make_overlay(x=100, y=100)
    ov._drag_move(_Ev(600, 400))           # press نشده — نباید حرکت کند
    assert ov.root.geo == []
    assert ov._manual_pos is None


def test_drag_release_clears_offset():
    ov = _make_overlay()
    ov._drag_start(_Ev(600, 400))
    ov._drag_end()
    ov._drag_move(_Ev(900, 500))            # بعد از رها شدن، حرکت معنا ندارد
    assert ov.root.geo == []
