"""شبیه‌سازی: بافر دنباله‌دار main.py در برابر بافر کامل — خط زمانی رأی/رقیب.

LiveTranscriber.partial_result فرض می‌کند پایان بافر = اکنون (زمان مطلق).
main.py فقط ۱۰ ثانیه‌ی آخر بافر را می‌فرستد (get_tail_16k(window_sec))،
پس buf_end همیشه ≈۱۰ ثانیه است و t_start همیشه ۰ — خط زمانی مطلق از
بین می‌رود. این شبیه‌سازی با یک «مدل فیک» قطعی، تعداد واژه‌های قفل‌شده
را در هر دو حالت می‌سنجد.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.asr import AsrHypothesis, AsrWord, LiveTranscriber

FRAMES_PER_SEC = 12.5  # هر فریم ۸۰ms
WINDOW = 10.0
STEP = 0.8
TOTAL = 40.0

# گفتار: از ثانیه ۱ تا ۳۹، هر ۰٫۷ ثانیه یک واژه (طول ۰٫۴s)، اطمینان ۰٫۸
WORDS = []
t = 1.0
i = 0
while t < TOTAL - 1.0:
    WORDS.append((t, t + 0.4, f"واژه{i}"))
    t += 0.7
    i += 1


def words_in(t0: float, t1: float):
    out = []
    for w0, w1, text in WORDS:
        if w1 > t0 + 1e-9 and w0 < t1 - 1e-9:
            s = max(0.0, w0 - t0)
            e = min(t1 - t0, w1 - t0)
            out.append(AsrWord(text=text, confidence=0.80, margin=0.5,
                               start=int(s * FRAMES_PER_SEC),
                               end=int(e * FRAMES_PER_SEC) + 1))
    return out


class FakeEngine:
    """مدلی که پنجره‌ی دریافتی را همیشه درست دیکد می‌کند (بدون خطا)."""

    def transcribe_with_details(self, win, sample_rate=16000, previous_text=None):
        t0 = FakeEngine.window_t0[0]
        t1 = t0 + win.size / 16000.0
        ws = words_in(t0, t1)
        text = " ".join(w.text for w in ws)
        return AsrHypothesis(text=text, confidence=0.8, margin=0.5,
                             words=tuple(ws))


def run(pass_full_buffer: bool):
    eng = FakeEngine()
    live = LiveTranscriber(eng, window_sec=WINDOW, stable_live=True)
    spoken = 0.0
    displayed_words = 0
    FakeEngine.window_t0 = [0.0]
    while spoken < TOTAL:
        spoken += STEP
        if pass_full_buffer:
            buf = (np.ones(int(spoken * 16000), dtype=np.float32))
            FakeEngine.window_t0 = [max(0.0, spoken - WINDOW)]
        else:
            # همان کاری که main.py می‌کند: فقط ۱۰ ثانیه‌ی آخر
            tail = min(WINDOW, spoken)
            buf = np.ones(int(tail * 16000), dtype=np.float32)
            FakeEngine.window_t0 = [spoken - tail]
        live.partial(buf)
        displayed_words = len(live._displayed.text.split()) if live._displayed else 0
    return displayed_words


full = run(pass_full_buffer=True)
tail = run(pass_full_buffer=False)
print(f"کل واژه‌های گفته‌شده: {len(WORDS)}")
print(f"حالت بافر کامل (فرض طراحی): {full} واژه قفل شد")
print(f"حالت بافر دنباله‌دار (main.py فعلی): {tail} واژه قفل شد")
