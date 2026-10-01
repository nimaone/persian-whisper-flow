#!/usr/bin/env python
"""استخراج زیرنویس از فایل صوتی/ویدیویی با موتور دیکته‌یار (sherpa-onnx NeMo CTC).

اجرا از ریشه‌ی پروژه:
    .venv/Scripts/python.exe subtitle.py INPUT [INPUT ...] [--chunk 16] [--out DIR]

برای هر ورودی یک فایل .srt (زیرنویس زمان‌بندی‌شده) و .txt (متن خام) کنار
خروجی می‌سازد. فایل ورودی می‌تواند هر فرمتی باشد که ffmpeg می‌خواند.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from app.asr import load_engine

SAMPLE_RATE = 16000


def load_audio_16k(path: Path) -> tuple[bytes, float]:
    """دیکد کردن هر فرمت ورودی به WAV مونوی ۱۶kHz با ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), str(tmp_path)],
        check=True,
    )
    with wave.open(str(tmp_path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
    tmp_path.unlink()
    return data, nframes / SAMPLE_RATE


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_file(path: Path, engine, chunk_sec: float, out_dir: Path):
    print(f"[subtitle] در حال پردازش {path.name} ...", flush=True)
    raw, duration = load_audio_16k(path)
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    cues: list[tuple[float, float, str]] = []
    for i, start in enumerate(range(0, samples.size, chunk_samples)):
        part = samples[start:start + chunk_samples]
        if part.size < SAMPLE_RATE // 10:  # تکه‌ی بی‌اهمیت انتهایی
            break
        text = engine.transcribe(part, SAMPLE_RATE)
        t0 = start / SAMPLE_RATE
        t1 = min((start + chunk_samples) / SAMPLE_RATE, duration)
        if text:
            cues.append((t0, t1, text))
        done = min(t1, duration)
        print(f"\r[subtitle] {done / duration * 100:5.1f}٪ ({done:.0f}s)",
              end="", flush=True)
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / (path.stem + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for n, (t0, t1, text) in enumerate(cues, 1):
            f.write(f"{n}\n{fmt_ts(t0)} --> {fmt_ts(t1)}\n{text}\n\n")
    txt_path = out_dir / (path.stem + ".txt")
    txt_path.write_text(" ".join(text for _, _, text in cues),
                        encoding="utf-8")
    print(f"[subtitle] {srt_path.name} و {txt_path.name} ساخته شدند")


def main():
    ap = argparse.ArgumentParser(description="ساخت زیرنویس با موتور دیکته‌یار")
    ap.add_argument("inputs", nargs="+", help="فایل صوتی/ویدیویی ورودی")
    ap.add_argument("--chunk", type=float, default=16.0,
                    help="طول هر تکه‌ی ترنسکرایپ به ثانیه (پیش‌فرض ۱۶)")
    ap.add_argument("--out", default=None,
                    help="پوشه خروجی (پیش‌فرض: کنار فایل ورودی)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    engine = load_engine(root / "model", num_threads=args.threads)
    for inp in args.inputs:
        p = Path(inp)
        out_dir = Path(args.out) if args.out else p.parent
        transcribe_file(p, engine, args.chunk, out_dir)


if __name__ == "__main__":
    sys.exit(main())
