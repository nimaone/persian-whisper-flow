"""قواعد نمایش دو تکه‌ی overlay — پیشوند قطعی (روشن) + پنجره‌ی جاری (کم‌رنگ)."""
from app.overlay import split_display


def test_plain_mode_has_no_separate_tail():
    """حالت معمولی: هیچ واژه‌ای قطعی‌نشده علامت نمی‌خورد."""
    assert split_display("الف ب پ", 0, 100) == ("الف ب پ", "", False)


def test_tail_is_split_off_at_word_boundary():
    assert split_display("الف ب پ ت", 2, 100) == ("الف ب", "پ ت", False)


def test_all_words_provisional():
    """پنجره‌ی جاری هنوز هیچ واژه‌ی قطعی نساخته — سر خالی می‌ماند."""
    assert split_display("الف ب پ", 9, 100) == ("", "الف ب پ", False)


def test_current_window_never_trimmed():
    """سقف ارتفاع فقط سرِ متن قطعی را می‌بُرد؛ پنجره‌ی جاری کامل می‌ماند."""
    words = [f"w{i}" for i in range(100)]
    head, tail, cut = split_display(" ".join(words), 5, 20)
    assert cut is True
    assert tail.split() == words[-5:]
    assert head.startswith("… ")
    assert len(head.split()) + len(tail.split()) <= 20


def test_budget_smaller_than_window_leaves_only_window():
    """سقف از خود پنجره‌ی جاری کوچک‌تر است — فقط همان نمایش داده می‌شود."""
    words = [f"w{i}" for i in range(50)]
    head, tail, cut = split_display(" ".join(words), 8, 6)
    assert cut is True
    assert head == ""
    assert tail.split() == words[-8:]


def test_short_text_is_not_marked():
    assert split_display("الف ب پ", 1, 3)[2] is False
