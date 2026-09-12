"""ضبط صدا از میکروفون — ری‌سمپل افزایشی به 16kHz مونو float32.

طراحی: بلوک‌های خام در callback فقط در صف کوتاه `_raw_new` می‌روند؛
هر بار `get_buffer_16k`/`get_tail_16k` صدا زده شود فقط داده‌ی تازه
ری‌سمپل می‌شود و خروجی 16k انباشته می‌گردد — پس فراخوانی مکرر حین
ضبط (partial) هزینه‌ی O(n) روی کل بافر ندارد.
"""
from __future__ import annotations

import threading
import time
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd

TARGET_RATE = 16000
MAX_BUFFER_SEC = 300.0  # سقف بافر نگه‌داشته‌شده — ضبط طولانی‌تر از ابتدا حذف می‌شود


class Recorder:
    """ضبط استریم از دستگاه ورودی پیش‌فرض (یا مشخص) به بافر 16kHz."""

    def __init__(self, device: int | None = None, block_ms: int = 100,
                 max_sec: float = MAX_BUFFER_SEC):
        self.device = device
        self.block_ms = block_ms
        self.max_sec = max_sec
        self._raw_new: list[np.ndarray] = []   # بلوک‌های خامِ هنوز ری‌سمپل‌نشده
        self._out_parts: list[np.ndarray] = []  # بخش‌های 16k به ترتیب زمانی
        self._out_total = 0
        self._raw_total = 0
        self._last_chunk = np.zeros(0, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)  # باقیمانده‌ی خامِ ری‌سمپل
        self._rs_in = 0  # ری‌سمپلر: کل نمونه‌های ورودی دیده‌شده
        self._rs_out = 0  # ری‌سمپلر: کل نمونه‌های خروجی تولیدشده
        self._lock = threading.Lock()
        # ری‌سمپل state دارد (carry)؛ drainهای هم‌زمان partial/finish را ردیف می‌کند
        self._resample_lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._src_rate = TARGET_RATE

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # خطاهای جزئی مانند overflow نادیده گرفته می‌شوند
        chunk = indata[:, 0].copy()  # کانال اول
        with self._lock:
            self._raw_new.append(chunk)
            self._last_chunk = chunk
            self._raw_total += chunk.size

    def start(self):
        if self._stream is not None:
            return
        with self._lock:
            self._raw_new = []
            self._out_parts = []
            self._out_total = 0
            self._raw_total = 0
            self._last_chunk = np.zeros(0, dtype=np.float32)
        with self._resample_lock:
            self._carry = np.zeros(0, dtype=np.float32)
            self._rs_in = 0
            self._rs_out = 0
        dev = self.device
        sr = int(sd.query_devices(dev or sd.default.device[0], "input")["default_samplerate"])
        self._src_rate = sr
        self._stream = sd.InputStream(
            device=dev,
            channels=1,
            samplerate=sr,
            dtype="float32",
            blocksize=int(sr * self.block_ms / 1000),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _resample_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """ری‌سمپل یک تکه با carry و شبکه‌ی زمانی سراسری — فقط با
        _resample_lock نگه‌داشته صدا زده شود.

        شمارنده‌های _rs_in/_rs_out موقعیت سراسری را نگه می‌دارند تا
        خروجیِ افزایشی دقیقاً روی همان شبکه‌ی ری‌سمپلِ کل بافر بیفتد
        (بدون drift در فراخوانی‌های متوالی).
        """
        if self._src_rate == TARGET_RATE:
            self._rs_in += chunk.size
            self._rs_out += chunk.size
            return chunk.astype(np.float32, copy=False)
        self._rs_in += chunk.size
        data = np.concatenate([self._carry, chunk.astype(np.float32)])
        base_in = self._rs_in - data.size  # اندیس سراسری اولین نمونه‌ی data
        ratio = TARGET_RATE / self._src_rate
        # خروجی‌ها فقط تا جایی که همسایه‌ی راستشان موجود است تولید می‌شوند
        max_j = int(np.floor((self._rs_in - 1) * ratio - 0.5))
        n_out = max(0, max_j + 1 - self._rs_out)
        if n_out == 0:
            self._carry = data.copy()
            return np.zeros(0, dtype=np.float32)
        x_old = (base_in + np.arange(data.size, dtype=np.float64)) / self._src_rate
        x_new = (np.arange(self._rs_out, self._rs_out + n_out, dtype=np.float64) + 0.5) / TARGET_RATE
        out = np.interp(x_new, x_old, data.astype(np.float64))
        self._rs_out += n_out
        # نگه‌داشتن فقط نمونه‌هایی که برای خروجی‌های آینده لازم‌اند
        next_t = (self._rs_out + 0.5) / TARGET_RATE
        keep_global = int(np.floor(next_t * self._src_rate))
        drop = min(max(keep_global - base_in, 0), data.size)
        self._carry = data[drop:].copy()
        return out.astype(np.float32)

    def _drain(self):
        """ری‌سمپل داده‌ی تازه و افزودن به خروجی انباشته."""
        with self._lock:
            new = self._raw_new
            if new:
                self._raw_new = []
        if not new:
            return
        data = np.concatenate(new) if len(new) > 1 else new[0]
        with self._resample_lock:
            out = self._resample_chunk(data)
        if out.size:
            with self._lock:
                self._out_parts.append(out)
                self._out_total += out.size
                self._trim_locked()

    def _trim_locked(self):
        """حذف از ابتدای خروجی تا سقف max_sec — با _lock نگه‌داشته."""
        excess = self._out_total - int(self.max_sec * TARGET_RATE)
        while excess > 0 and self._out_parts:
            part = self._out_parts[0]
            if part.size > excess:
                self._out_parts[0] = part[excess:]
                self._out_total -= excess
                excess = 0
            else:
                excess -= part.size
                self._out_total -= part.size
                self._out_parts.pop(0)

    def get_buffer_16k(self) -> np.ndarray:
        """کل بافر ضبط‌شده تا این لحظه (16kHz mono float32)."""
        self._drain()
        with self._lock:
            parts = list(self._out_parts)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def get_tail_16k(self, window_sec: float) -> np.ndarray:
        """آخرین window_sec ثانیه‌ی بافر 16k — بدون کپی کل بافر (برای partial)."""
        self._drain()
        need = int(window_sec * TARGET_RATE)
        with self._lock:
            parts = self._out_parts
            total = 0
            idx = len(parts)
            while idx > 0 and total < need:
                total += parts[idx - 1].size
                idx -= 1
            tail = list(parts[idx:])
        if not tail:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(tail)
        return out[-need:] if out.size > need else out

    def recent_rms(self) -> float:
        """RMS آخرین بلوک ضبط‌شده (~100ms) — برای نوار سطح overlay."""
        with self._lock:
            last = self._last_chunk
        if last.size == 0:
            return 0.0
        return float(np.sqrt((last.astype(np.float64) ** 2).mean()))

    def duration_sec(self) -> float:
        with self._lock:
            total = self._raw_total
        return total / self._src_rate


def current_input_devices() -> list[dict]:
    """همه‌ی ورودی‌های خام با Host API آن‌ها — پایه‌ی فهرست و بازیابی."""
    return [{"index": i, "name": d["name"],
             "rate": int(d["default_samplerate"]),
             "api": sd.query_hostapis(d["hostapi"])["name"]}
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def device_label(d: dict) -> str:
    """برچسب نمایشی پایدار — بدون ایندکس خام (ناپایدار بین بوت‌ها)."""
    return f"{d['name']} — {d['api']}"


def resolve_pinned_device(pinned: int | None, key: str | None) -> int | None:
    """اعتبارسنجی/بازیابی دستگاه پین‌شده — ایندکس‌های PortAudio بین بوت‌ها
    و جابه‌جایی USB عوض می‌شوند؛ کلید پایدار (نام — API) مرجع است.

    None در خروجی یعنی دستگاه پین‌شده دیگر پیدا نیست → حالت خودکار.
    """
    if pinned is None and not key:
        return None
    devices = current_input_devices()
    if key:
        for d in devices:
            if d["index"] == pinned and device_label(d) == key:
                return pinned  # پین همچنان معتبر
        for d in devices:
            if device_label(d) == key:
                return d["index"]  # ایندکس جابه‌جا شده — با کلید بازیابی
        return None  # دستگاه پیدا نشد
    return pinned


def list_input_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"], "rate": int(d["default_samplerate"])})
    return out


# ترجیح Host API برای نماینده‌ی هر میکروفون فیزیکی — WASAPI استاندارد
# ویندوز است؛ WDM-KS پس‌انداز (روی برخی سیستم‌ها ناپایدار)، MME و
# DirectSound قدیمی‌اند (نویز پایه بالاتر).
HOSTAPI_PREFERENCE = ("Windows WASAPI", "Windows WDM-KS",
                      "Windows DirectSound", "MME")


def dedupe_input_devices(entries: list[dict]) -> list[dict]:
    """یک مدخل برای هر میکروفون فیزیکی از میان ورودی‌های همه‌ی Host APIها.

    ویندوز هر دستگاه را به ازای هر API یک بار فهرست می‌کند (۳ میکروفون
    فیزیکی → ۱۵+ مدخل). گروه‌بندی با شباهت نام نرمال‌شده (نام دستگاه‌ها
    بین APIها کمی فرق می‌کند) و نماینده‌ی هر گروه = API با اولویت بالاتر.
    نام‌های مستعار سیستم («Sound Mapper»، «Primary Sound Capture») حذف
    می‌شوند — همان دستگاه پیش‌فرض‌اند، نه میکروفون جدا.
    """
    def api_rank(api: str) -> int:
        return HOSTAPI_PREFERENCE.index(api) if api in HOSTAPI_PREFERENCE \
            else len(HOSTAPI_PREFERENCE)

    def name_key(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    kept: list[dict] = []
    for e in sorted(entries, key=lambda e: api_rank(e.get("api", ""))):
        nk = name_key(e["name"])
        if "soundmapper" in nk or "primarysoundcapture" in nk:
            continue
        group = None
        for k in kept:
            if SequenceMatcher(None, nk, k["_norm"]).ratio() >= 0.6:
                group = k
                break
        if group is None:
            kept.append({**e, "_norm": nk})
    return [{k: v for k, v in e.items() if not k.startswith("_")} for e in kept]


def probe_device_level(device: int, seconds: float = 0.35) -> tuple[float, bool]:
    """سطح RMS دستگاه ورودی با استریم واقعی — برای تشخیص میکروفون زنده.

    مقدار دوم «اعتبار» است: دستگاه‌هایی که اصلاً باز نمی‌شوند یا داده‌ی
    خراب می‌دهند (دامنه بریده/غیرواقعی — بعضی مسیرهای WDM-KS) رد می‌شوند
    تا detect_best_device روی دستگاه شکسته نگه نایستد.
    """
    try:
        info = sd.query_devices(device, "input")
        sr = int(info["default_samplerate"])
        vals: list[float] = []
        peak = 0.0
        with sd.InputStream(device=device, channels=1, samplerate=sr,
                            dtype="float32",
                            blocksize=int(sr * 0.05)) as st:
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                data, _overflow = st.read(int(sr * 0.05))
                peak = max(peak, float(np.abs(data).max()))
                vals.append(float(np.sqrt(
                    (data[:, 0].astype(np.float64) ** 2).mean())))
        if not vals:
            return 0.0, False
        mean_rms = float(np.mean(vals))
        # داده‌ی خراب: نمونه فراتر از ۱٫۰ (بریده) یا RMS غیرواقعی
        if peak > 1.0 or mean_rms > 0.5:
            return 0.0, False
        return mean_rms, True
    except Exception:
        return 0.0, False


def detect_best_device() -> int | None:
    """دستگاه ورودی با بالاترین سطح سیگنال (و نرخ >= 16000) را برمی‌گرداند.

    فقط دستگاه‌هایی که واقعاً استریم سالم می‌دهند کاندیدند؛ میکروفون‌های
    مجازی (ManyCam و امثالش) معمولاً سکوت مطلق می‌دهند و مسیرهای خراب
    رد می‌شوند. هیچ دستگاهی سیگنال نداشت → None یعنی پیش‌فرض سیستم.
    """
    best, best_rms = None, 0.0
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        if int(d["default_samplerate"]) < 16000:
            continue
        rms, valid = probe_device_level(i)
        if not valid:
            continue
        if rms > best_rms:
            best, best_rms = i, rms
    if best is not None and best_rms > 0.0002:
        return best
    # هیچ دستگاهی سیگنال نداشت — device None یعنی پیش‌فرض سیستم
    return None


def input_quality(vals: list[float]) -> tuple[str, str]:
    """برآورد کیفیت ورودی از نمونه‌های RMS تست صدا → (متن، سطح).

    سطح یکی از good/warn/bad/none — برای رنگ نشانگر تب میکروفون.
    کف نویز = چارک ده‌میانگین‌ها، اوج = چارک نودوپنجم.
    """
    if not vals:
        return "برای سنجش کیفیت، تست را شروع کن و چند ثانیه صحبت کن", "none"
    s = sorted(vals)
    floor = s[max(0, int(len(s) * 0.10))]
    peak = s[min(len(s) - 1, int(len(s) * 0.95))]
    if peak < 0.0008:
        return "سیگنالی نمی‌آید — دستگاه دیگری را امتحان کن", "bad"
    snr = 20.0 * float(np.log10(max(peak, 1e-9) / max(floor, 1e-9)))
    stats = f"نویز پایه {floor:.4f} • اوج صدا {peak:.4f} • SNR≈{snr:.0f}dB"
    if snr >= 20:
        return f"کیفیت ورودی: خوب — {stats}", "good"
    if snr >= 12:
        return f"کیفیت ورودی: متوسط — {stats}", "warn"
    return f"کیفیت ورودی: ضعیف — نویز تقریباً هم‌سطح صداست — {stats}", "bad"
