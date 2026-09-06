"""CLI فاز ۱ — ضبط از میکروفون با Enter شروع/توقف، سپس ترنسکرایپ نهایی.

استفاده:  python spikes/record_cli.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.asr import load_engine  # noqa: E402
from app.recorder import Recorder  # noqa: E402


def main():
    print("[cli] بارگذاری مدل...")
    engine = load_engine(ROOT / "model")

    rec = Recorder()
    print("[cli] Enter را برای شروع ضبط بزن...")
    input()
    rec.start()
    print("[cli] ضبط شروع شد — برای توقف Enter بزن...")
    input()
    rec.stop()
    print("[cli] ضبط متوقف شد. در حال ترنسکرایب...")

    buf = rec.get_buffer_16k()
    if buf.size == 0:
        print("[cli] بافر خالی!")
        return
    print(f"[cli] {len(buf)/16000:.1f} ثانیه صدا ضبط شد")
    text = engine.transcribe(buf)
    print(f"[result] {text!r}")


if __name__ == "__main__":
    main()
