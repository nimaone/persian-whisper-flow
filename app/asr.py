"""موتور تشخیص گفتار — بارگذاری مدل Shenava-Koochik از طریق sherpa-onnx.

مدل از نوع NeMo CTC آفلاین است؛ برای نمایش زنده، هر بار روی آخرین
بخش بافر inference مجدد اجرا می‌شود و پیشوند پایدار قفل می‌شود.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sherpa_onnx


@dataclass(frozen=True)
class AsrWord:
    """یک واژه‌ی تشخیص‌داده‌شده با امتیاز اطمینان مدل."""

    text: str
    confidence: float
    margin: float


@dataclass(frozen=True)
class AsrHypothesis:
    """نتیجه‌ی ASR همراه با نشانه‌های اطمینان.

    `confidence` میانگین احتمال توکن برتر روی فریم‌های غیر blank است و
    `margin` میانگین فاصله‌ی احتمال توکن اول و دوم روی همان فریم‌ها.
    این مقادیر برای مقایسه‌ی فرضیه‌های متوالی روی پنجره‌های هم‌اندازه
    مناسب‌اند.
    """

    text: str
    confidence: float = 0.0
    margin: float = 0.0
    words: tuple[AsrWord, ...] = ()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def ctc_confidence(
    log_probs: np.ndarray,
    blank_id: int,
    length: int | None = None,
    probs: np.ndarray | None = None,
) -> tuple[float, float]:
    """اطمینان میانگین و فاصله‌ی top-1/top-2 روی فریم‌های غیر blank.

    اگر `probs` از قبل محاسبه شده باشد، برای جلوگیری از softmax تکراری
    استفاده می‌شود.
    """
    if length is not None:
        log_probs = log_probs[:length]
        if probs is not None:
            probs = probs[:length]
    if log_probs.size == 0:
        return 0.0, 0.0
    probs = _softmax(log_probs) if probs is None else probs
    ids = probs.argmax(axis=-1)
    mask = ids != blank_id
    if not mask.any():
        mask = np.ones(ids.shape, dtype=bool)
    # sort کامل روی ۱۰۲۵ توکن لازم نیست؛ فقط دو احتمال برتر کافی است.
    top2 = np.partition(probs, -2, axis=-1)[:, -2:]
    top1 = top2[:, 1]
    margin = top1 - top2[:, 0]
    return float(top1[mask].mean()), float(margin[mask].mean())


def _word_confidences(
    log_probs: np.ndarray,
    tokens: list[tuple[str, tuple[int, int]]],
    blank_id: int,
    probs: np.ndarray | None = None,
) -> tuple[AsrWord, ...]:
    """امتیاز هر واژه را از فریم‌های مربوط به همان واژه در می‌آورد."""
    if not tokens:
        return ()
    probs = _softmax(log_probs) if probs is None else probs
    ids = probs.argmax(axis=-1)
    # sort کامل روی ۱۰۲۵ توکن لازم نیست؛ فقط دو احتمال برتر کافی است.
    top2 = np.partition(probs, -2, axis=-1)[:, -2:]
    top1 = top2[:, 1]
    margin = top1 - top2[:, 0]
    out: list[AsrWord] = []
    for text, (start, end) in tokens:
        end = max(end, start + 1)
        sl = slice(start, min(end, len(probs)))
        if sl.stop <= sl.start:
            continue
        mask = ids[sl] != blank_id
        if not mask.any():
            mask = np.ones(sl.stop - sl.start, dtype=bool)
        out.append(AsrWord(
            text=text,
            confidence=float(top1[sl][mask].mean()),
            margin=float(margin[sl][mask].mean()),
        ))
    return tuple(out)


class _CtcHypothesisDecoder:
    """اجرای مستقیم ONNX برای گرفتن log-probs و فرضیه‌های CTC.

    sherpa-onnx 1.13.7 برای مدل NeMo CTC مقدار `ys_log_probs` را خالی
    برمی‌گرداند، پس برای امتیازدهی باید خروجی خام مدل را گرفت.
    """

    def __init__(self, model_dir: str | Path, num_threads: int = 4,
                 beam_width: int = 4):
        import onnxruntime as ort
        from pyctcdecode import Alphabet, BeamSearchDecoderCTC

        from app.hotword_asr import kaldi_fbank, load_labels

        self._kaldi_fbank = kaldi_fbank
        labels = load_labels(Path(model_dir) / "tokens.txt")
        self._blank_id = labels.index("")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._decoder = BeamSearchDecoderCTC(Alphabet(labels, is_bpe=True))
        self._beam_width = beam_width
        self._lock = threading.Lock()

    def decode(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        previous_text: str | None = None,
    ) -> AsrHypothesis:
        feat = self._kaldi_fbank(samples, sample_rate)
        with self._lock:
            log_probs, lengths = self._session.run(
                None,
                {
                    "audio_signal": feat.T[None].astype(np.float32),
                    "length": np.array([feat.shape[0]], dtype=np.int64),
                },
            )
        length = int(lengths[0])
        log_probs = log_probs[0][:length]
        beams = self._decoder.decode_beams(
            log_probs,
            beam_width=self._beam_width,
            beam_prune_logp=-5.0,
            token_min_logp=-5.0,
        )
        if not beams:
            return AsrHypothesis("")
        top = beams[0]
        tokens = top[2]
        text = top[0].strip()
        # اگر متن تغییری نکرده باشد، امتیازدهی کامل واژه‌ها بی‌فایده است.
        if previous_text is not None and text == previous_text:
            return AsrHypothesis(text)
        probs = _softmax(log_probs)
        confidence, margin = ctc_confidence(
            log_probs, self._blank_id, length, probs=probs
        )
        return AsrHypothesis(
            text=text,
            confidence=confidence,
            margin=margin,
            words=_word_confidences(
                log_probs, tokens, self._blank_id, probs=probs
            ),
        )

    def release(self):
        """session و دیکدر را برای GC آزاد می‌کند."""
        self._session = None
        self._decoder = None


class AsrEngine:
    """بارگذاری مدل و ترنسکرایپ — thread-safe."""

    def __init__(self, model_dir: str | Path = "model", num_threads: int = 4):
        model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._num_threads = num_threads
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(model_dir / "model.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            debug=False,
        )
        self._lock = threading.Lock()
        self._detail_decoder: _CtcHypothesisDecoder | None = None
        self._detail_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """ترنسکرایپ موج صوتی float32 [-1..1] → متن فارسی."""
        if samples.size == 0:
            return ""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            self._recognizer.decode_stream(stream)
            return stream.result.text.strip()

    def transcribe_with_details(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        previous_text: str | None = None,
    ) -> AsrHypothesis:
        """ترنسکرایپ همراه با امتیاز اطمینان برای مسیر زنده.

        متن از همان مدل خوانده می‌شود، اما برای گرفتن log-probs و
        فاصله‌ی فرضیه‌ها خروجی خام ONNX لازم است. اگر این مسیر خطا
        داد، به ترنسکرایپ معمولی برمی‌گردیم تا زنده‌نمایی قطع نشود.
        """
        if samples.size == 0:
            return AsrHypothesis("")
        try:
            with self._detail_lock:
                if self._detail_decoder is None:
                    self._detail_decoder = _CtcHypothesisDecoder(
                        self._model_dir,
                        num_threads=self._num_threads,
                        beam_width=2,
                    )
                return self._detail_decoder.decode(
                    samples, sample_rate, previous_text=previous_text
                )
        except Exception:
            try:
                return AsrHypothesis(self.transcribe(samples, sample_rate))
            except Exception:
                return AsrHypothesis("")

    def release_detail_decoder(self):
        """session دوم ONNX را آزاد می‌کند؛ مسیر اصلی sherpa حفظ می‌شود."""
        with self._detail_lock:
            decoder = self._detail_decoder
            self._detail_decoder = None
        if decoder is not None and hasattr(decoder, "release"):
            decoder.release()


DEFAULT_BEAM_WIDTH = 2
FINAL_CHUNK_SEC = 16.0


class DirectCtcAsrEngine:
    """موتور تک‌مدلی CTC — فقط یک session ONNX دارد.

    برای حالت متن پایدار استفاده می‌شود تا sherpa و session دوم ONNX
    هم‌زمان در حافظه نمانند. `transcribe` نهایی هم از همین دیکدر
    خوانده می‌شود.
    """

    def __init__(
        self,
        model_dir: str | Path = "model",
        num_threads: int = 4,
        beam_width: int = 2,
    ):
        self._decoder = _CtcHypothesisDecoder(
            model_dir,
            num_threads=num_threads,
            beam_width=beam_width,
        )
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._decoder is not None and getattr(
            self._decoder, "_session", None
        ) is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """ترنسکرایپ موج صوتی float32 [-1..1] → متن فارسی.

        برای فایل‌های طولانی، decode در chunkهای ۱۶ ثانیه‌ای اجرا می‌شود.
        این کار هم سرعت نهایی را بالا می‌برد و هم از افت کیفیت decode
        طولانی جلوگیری می‌کند.
        """
        if samples.size == 0:
            return ""
        chunk_samples = int(FINAL_CHUNK_SEC * 16000)
        with self._lock:
            if samples.size <= chunk_samples:
                return self._decoder.decode(samples, sample_rate).text
            parts: list[str] = []
            for start in range(0, samples.size, chunk_samples):
                text = self._decoder.decode(
                    samples[start:start + chunk_samples], sample_rate
                ).text
                if text:
                    parts.append(text)
            return " ".join(parts)

    def transcribe_with_details(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        previous_text: str | None = None,
    ) -> AsrHypothesis:
        if samples.size == 0:
            return AsrHypothesis("")
        with self._lock:
            return self._decoder.decode(
                samples, sample_rate, previous_text=previous_text
            )

    def release_detail_decoder(self):
        """در موتور تک‌مدلی session اصلی نباید آزاد شود."""
        return None


class SpeechGate:
    """جلوگیری از inference غیرضروری در سکوت ادامه‌دار.

    تا `hangover` ثانیه پس از آخرین صدای واقعی اجازه‌ی decode می‌دهد
    تا دنباله‌ی کلمات قطع نشود؛ بعد از آن تا صدای تازه، inference را رد می‌کند.
    """

    def __init__(self, threshold: float, hangover: float = 1.5):
        self.threshold = float(threshold)
        self.hangover = float(hangover)
        self._last_speech_at: float | None = None

    def reset(self):
        self._last_speech_at = None

    def should_decode(self, rms: float, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else float(now)
        if rms > self.threshold:
            self._last_speech_at = now
            return True
        if self._last_speech_at is None:
            return False
        return (now - self._last_speech_at) <= self.hangover


class LiveTranscriber:
    """ترنسکرایپ تدریجی روی بافر در حال رشد، با قفل پیشوند پایدار.

    چون مدل آفلاین است، متن «زنده» با اجرای مجدد inference روی
    آخرین حداکثر `window_sec` ثانیه‌ی بافر ساخته می‌شود. متنِ قطعی
    به‌سختی عوض می‌شود؛ یک فرضیه‌ی جدید فقط وقتی جایگزین می‌شود که
    یا ادامه‌ی متن قبلی باشد، یا امتیاز اطمینانش به‌اندازه‌ی کافی
    بالاتر باشد و `hysteresis` بار پایدار مانده باشد.
    """

    def __init__(
        self,
        engine: AsrEngine,
        window_sec: float = 10.0,
        stable_live: bool = False,
        margin: float = 0.02,
        hysteresis: int = 2,
        min_word_confidence: float = 0.25,
    ):
        self.engine = engine
        self.window_sec = window_sec
        self.stable_live = bool(stable_live)
        self.margin = float(margin)
        self.hysteresis = max(1, int(hysteresis))
        self.min_word_confidence = float(min_word_confidence)
        self._displayed: AsrHypothesis | None = None
        self._candidate: AsrHypothesis | None = None
        self._candidate_count = 0

    def configure(self, stable_live: bool):
        """حالت متن پایدار را تغییر می‌دهد و وضعیت قبلی را پاک می‌کند."""
        stable_live = bool(stable_live)
        if self.stable_live == stable_live:
            return
        self.stable_live = stable_live
        self.reset()
        # اگر مسیر سبک انتخاب شد، session دوم ONNX دیگر لازم نیست.
        if not stable_live and hasattr(self.engine, "release_detail_decoder"):
            self.engine.release_detail_decoder()

    def reset(self):
        """وضعیت پیشوند پایدار را برای ضبط جدید پاک می‌کند."""
        self._displayed = None
        self._candidate = None
        self._candidate_count = 0

    def _words(self, hyp: AsrHypothesis) -> list[AsrWord]:
        if hyp.words:
            return list(hyp.words)
        return [AsrWord(t, hyp.confidence, hyp.margin) for t in re.findall(r"\S+", hyp.text)]

    def _word(self, hyp: AsrHypothesis | None, index: int) -> AsrWord | None:
        if hyp is None:
            return None
        words = self._words(hyp)
        return words[index] if 0 <= index < len(words) else None

    @staticmethod
    def _shift_prefix_len(old: list[AsrWord], new: list[AsrWord]) -> int | None:
        """اگر پنجره‌ی صدا جلو رفته باشد، چند واژه از ابتدا افتاده است؟"""
        for shift in range(1, len(old) + 1):
            overlap = min(len(old) - shift, len(new))
            if overlap <= 0:
                continue
            if [w.text for w in new[:overlap]] == [w.text for w in old[shift:shift + overlap]]:
                return shift
        return None

    def _accepts(self, candidate: AsrHypothesis, current: AsrHypothesis) -> bool:
        """آیا فرضیه‌ی جدید می‌تواند جایگزین متن نمایش‌داده‌شده شود؟"""
        cur_words = self._words(current)
        new_words = self._words(candidate)

        # ادامه‌ی متن قبلی: چون پیشوند پایدار است، فقط دنباله اضافه می‌شود.
        if len(new_words) >= len(cur_words) and all(
            n.text == c.text for n, c in zip(new_words, cur_words)
        ):
            if len(new_words) == len(cur_words):
                return True
            tail = new_words[len(cur_words):]
            return min(w.confidence for w in tail) >= self.min_word_confidence

        # پنجره‌ی ۱۰ ثانیه‌ای جلو رفته و واژه‌های ابتدایی خارج شده‌اند؛
        # این جابه‌جایی طبیعی است و نباید متن را برای همیشه قفل کند.
        shift = self._shift_prefix_len(cur_words, new_words)
        if shift is not None:
            overlap = len(cur_words) - shift
            tail = new_words[overlap:]
            if tail and min(w.confidence for w in tail) < self.min_word_confidence:
                return False
            return candidate.confidence >= current.confidence - 0.10

        # با هم‌ترازسازی واژه‌ها فقط تغییرهای واقعی را می‌سنجم؛ حذف ابتدای
        # متن به‌خاطر حرکت پنجره و درج دنباله‌ی جدید خطای نوسان نیست.
        matcher = SequenceMatcher(
            None, [w.text for w in cur_words], [w.text for w in new_words],
            autojunk=False,
        )
        matched = sum(block.size for block in matcher.get_matching_blocks())
        overlap = matched / max(1, min(len(cur_words), len(new_words)))
        if overlap >= 0.65 and candidate.confidence >= current.confidence - 0.10:
            has_replace = any(tag == "replace" for tag, *_ in matcher.get_opcodes())
            if not has_replace or self._candidate_count >= self.hysteresis:
                return True

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                old_conf = [w.confidence for w in cur_words[i1:i2]]
                new_conf = [w.confidence for w in new_words[j1:j2]]
                if sum(new_conf) / len(new_conf) < sum(old_conf) / len(old_conf) + self.margin:
                    return False
            elif tag == "delete":
                # حذف از ابتدا معمولاً به‌خاطر حرکت پنجره است؛ در بقیه‌ی
                # موارد فقط اگر اطمینان کلی به‌قدر کافی نیفتد بپذیر.
                if i1 != 0 and candidate.confidence < current.confidence - 0.10:
                    return False
            elif tag == "insert":
                if any(w.confidence < self.min_word_confidence for w in new_words[j1:j2]):
                    return False

        return True

    def partial_result(self, buffer16k: np.ndarray) -> AsrHypothesis:
        """ترنسکرایپ حین ضبط روی پنجره‌ی انتهای بافر."""
        win = buffer16k[-int(self.window_sec * 16000):]
        if win.size < 1600:  # کمتر از 0.1s صدا
            return AsrHypothesis("")
        # مسیر سبک پیش‌فرض: فقط گری‌دی sherpa، بدون session دوم و امتیازدهی.
        if not self.stable_live:
            try:
                return AsrHypothesis(self.engine.transcribe(win))
            except Exception:
                return AsrHypothesis("")
        try:
            details = getattr(self.engine, "transcribe_with_details", None)
            if details is not None:
                current = self._displayed
                result = details(
                    win,
                    previous_text=current.text if current is not None else None,
                )
            else:
                result = AsrHypothesis(self.engine.transcribe(win))
        except Exception:
            result = AsrHypothesis(self._displayed.text if self._displayed else "")
            return result

        if not result.text.strip():
            return self._displayed or result

        current = self._displayed
        # متن تکراری نباید با hypothesis بدون confidence جایگزین شود.
        if current is not None and result.text == current.text:
            self._candidate = None
            self._candidate_count = 0
            return current

        if self._candidate is not None and self._candidate.text == result.text:
            self._candidate_count += 1
        else:
            self._candidate = result
            self._candidate_count = 1

        current = self._displayed
        accepted = current is None or self._accepts(result, current)
        if not accepted and self._candidate_count >= self.hysteresis and result.confidence > 0.0:
            accepted = True
        if accepted:
            self._displayed = result
            self._candidate_count = 0
            self._candidate = None
            return result
        return current

    def partial(self, buffer16k: np.ndarray) -> str:
        """ترنسکرایپِ حین ضبط روی پنجره‌ی انتهای بافر."""
        return self.partial_result(buffer16k).text

    def final(self, buffer16k: np.ndarray) -> str:
        """ترنسکرایپ نهایی روی کل بافر بعد از توقف ضبط."""
        if buffer16k.size == 0:
            return ""
        try:
            return self.engine.transcribe(buffer16k)
        except Exception as e:
            raise RuntimeError(f"ترنسکرایپ نهایی ناموفق: {e}") from e


def load_engine(model_dir: str | Path = "model", num_threads: int = 4) -> AsrEngine:
    t0 = time.perf_counter()
    eng = AsrEngine(model_dir, num_threads=num_threads)
    t1 = time.perf_counter()
    print(f"[asr] مدل در {t1 - t0:.2f} ثانیه بارگذاری شد")
    return eng
