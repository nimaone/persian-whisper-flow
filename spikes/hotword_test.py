"""Spike: reproduce sherpa-onnx greedy output with pyctcdecode, then A/B test hotword boosting.

Pipeline: wav -> kaldi-style 80-dim fbank (25ms/10ms, povey, preemph 0.97)
          -> model.onnx (NeMo CTC encoder) -> log_probs -> pyctcdecode beam search.

Usage:
    python hotword_test.py <wav> [<wav> ...]
"""
import sys
import wave

import numpy as np
import onnxruntime as ort
from pyctcdecode import Alphabet, BeamSearchDecoderCTC

MODEL = "model/model.onnx"
TOKENS = "model/tokens.txt"


def load_labels():
    toks = {}
    with open(TOKENS, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2:
                toks[int(parts[1])] = parts[0]
    labels = [toks[i] for i in range(len(toks))]
    # pyctcdecode treats the empty-string label as CTC blank
    labels = ["" if t == "<blk>" else t for t in labels]
    return labels


def kaldi_fbank(x: np.ndarray, sr: int = 16000, num_mel: int = 80) -> np.ndarray:
    """kaldi-native-fbank defaults: povey window, 25ms/10ms, preemph 0.97,
    remove dc, power spectrum, 80 mel bins 20Hz..Nyquist, ln with floor."""
    frame_len, frame_shift = 400, 160
    if sr != 16000:
        n_out = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x), n_out, endpoint=False),
                      np.arange(len(x)), x)
    x = x.astype(np.float64)
    n_frames = 1 + max(0, (len(x) - frame_len) // frame_shift)
    idx = np.arange(frame_len)[None, :] + frame_shift * np.arange(n_frames)[:, None]
    frames = x[idx]
    # preemphasize (prepend first sample), then remove dc, then window
    pre = np.concatenate([frames[:, :1], frames], axis=1)
    frames = pre[:, 1:] - 0.97 * pre[:, :-1]
    frames = frames - frames.mean(axis=1, keepdims=True)
    n = np.arange(frame_len)
    povey = np.power(0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_len - 1)), 0.85)
    frames = frames * povey
    fft = np.fft.rfft(frames, n=512)
    power = (fft.real**2 + fft.imag**2)[:, :257]  # kaldi keeps N/2+1 bins
    # kaldi mel filterbank (non-HTK mel scale: 1127 ln(1+f/700))
    def hz2mel(f):
        return 1127.0 * np.log1p(np.asarray(f) / 700.0)
    mel_pts = np.linspace(hz2mel(20.0), hz2mel(8000.0), num_mel + 2)
    hz_pts = 700.0 * np.expm1(mel_pts / 1127.0)
    bins = np.floor((512 + 1) * hz_pts / 16000.0).astype(int)
    fbank = np.zeros((num_mel, 257))
    for m in range(1, num_mel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c += 1
        if r == c:
            r += 1
        for k in range(l, c):
            fbank[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            fbank[m - 1, k] = (r - k) / (r - c)
    feat = power @ fbank.T
    return np.log(np.maximum(feat, 1.19e-7))


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, w.getframerate()


def main():
    labels = load_labels()
    decoder = BeamSearchDecoderCTC(Alphabet(labels, is_bpe=True))
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

    for path in sys.argv[1:]:
        x, sr = read_wav(path)
        feat = kaldi_fbank(x, sr)  # [T, 80]
        log_probs, _ = sess.run(
            None,
            {"audio_signal": feat.T[None].astype(np.float32),
             "length": np.array([feat.shape[0]], dtype=np.int64)},
        )
        logits = log_probs[0]
        plain = decoder.decode(logits, beam_width=100)
        boosted = decoder.decode(logits, beam_width=100,
                                 hotwords=["اپ"], hotword_weight=3.0)
        print(f"--- {path}")
        print(f"  pyctc no-hotword : {plain!r}")
        print(f"  pyctc +hotword اپ : {boosted!r}")


if __name__ == "__main__":
    main()
