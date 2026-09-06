"""Spike: load Shenava-Koochik (NeMo CTC) via sherpa-onnx and transcribe a wav file.

Usage:
    python transcribe_cli.py <path-to-wav>
"""
import sys
import time
import wave

import numpy as np
import sherpa_onnx


def load_recognizer(model_path: str, tokens_path: str) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model_path,
        tokens=tokens_path,
        num_threads=4,
        debug=False,
    )


def read_wav_mono_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit wav, got {width*8}-bit")
    data = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != 16000:
        # simple linear resample
        n_out = int(len(data) * 16000 / rate)
        x_old = np.linspace(0, len(data), num=len(data), endpoint=False)
        x_new = np.linspace(0, len(data), num=n_out, endpoint=False)
        data = np.interp(x_new, x_old, data.astype(np.float64)).astype(np.int16)
    return data.astype(np.float32) / 32768.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"[asr] loading model...")
    t0 = time.perf_counter()
    rec = load_recognizer("model/model.onnx", "model/tokens.txt")
    t1 = time.perf_counter()
    print(f"[asr] model loaded in {t1-t0:.2f}s")

    samples = read_wav_mono_16k(wav_path)
    print(f"[wav] {len(samples)/16000:.2f}s @16kHz")

    t2 = time.perf_counter()
    stream = rec.create_stream()
    stream.accept_waveform(16000, samples)
    rec.decode_stream(stream)
    t3 = time.perf_counter()

    text = stream.result.text
    print(f"[infer] decode took {t3-t2:.2f}s")
    print(f"[result] {text!r}")


if __name__ == "__main__":
    main()
