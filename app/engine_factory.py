"""ساخت موتور فعال بر اساس تنظیمات — مشترک بین بک‌اند CTk و فلت.

مدل فعال از `model_id` در کاتالوگ مشخص می‌شود و نوع موتور از `kind`:
  - ctc        → مسیر فعلی (sherpa NeMo CTC، هات‌وورد، متن پایدار)
  - transducer → `NemotronAsrEngine` (هات‌وورد/متن پایدار رویش معنا ندارد)

هر تغییر در تنظیماتی که نوع موتور را عوض می‌کند باید در `engine_key`
هر بک‌اند هم لحاظ شود تا rebuild درست رخ دهد.
"""
from __future__ import annotations

from typing import Callable

from app.asr import DirectCtcAsrEngine, load_engine
from app.config import model_dir
from app.models_catalog import ModelSpec
from app.models_catalog import get as get_model


def active_spec(cfg) -> ModelSpec:
    """مدل فعال — ناشناخته/خالی → مدل پیش‌فرض."""
    return get_model(cfg.get("model_id"))


def is_transducer(cfg) -> bool:
    return active_spec(cfg).kind == "transducer"


def engine_key(cfg, hotwords) -> tuple:
    """امضای تنظیماتی که نوع/تخصیص موتور را تعیین می‌کند."""
    return (
        bool(cfg.get("hotword_boost")),
        tuple(hotwords or ()),
        bool(cfg.get("stable_live")),
        str(cfg.get("model_id") or ""),
        str(cfg.get("model_dir") or ""),
    )


def make_engine(cfg, notify: Callable[[str], None] | None = None, hotwords=None):
    """موتور فعال را می‌سازد؛ در نبود وابستگی به موتور عادی برمی‌گردد."""
    spec = active_spec(cfg)
    nthreads = int(cfg.get("num_threads") or 4)
    md = spec.resolve_dir(model_dir(cfg.get("model_dir")))

    if spec.kind == "transducer":
        from app.asr_nemotron import NemotronAsrEngine

        return NemotronAsrEngine(md, num_threads=nthreads)

    hw = list(hotwords or [])
    if cfg.get("hotword_boost") and hw:
        try:
            from app.hotword_asr import load_hotword_engine

            return load_hotword_engine(
                model_dir=md, num_threads=nthreads, hotwords=hw)
        except Exception as e:
            if notify:
                notify(f"حالت واژه‌های حساس فعال نشد؛ موتور عادی: {str(e)[:60]}")
    # اگر متن پایدار خواسته شده، به‌جای sherpa + session دوم،
    # یک session ONNX برای هم transcribe و هم امتیازدهی لود می‌کنیم.
    if bool(cfg.get("stable_live")):
        try:
            return DirectCtcAsrEngine(
                model_dir=md, num_threads=nthreads, beam_width=2)
        except Exception as e:
            if notify:
                notify(f"موتور متن پایدار فعال نشد؛ موتور عادی: {str(e)[:60]}")
    return load_engine(model_dir=md, num_threads=nthreads)
