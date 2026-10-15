"""کاتالوگ مدل‌های تشخیص گفتار — یک منبع حقیقت برای دانلود/انتخاب/موتور.

هر مدل با یک `ModelSpec` توصیف می‌شود: نوع معماری (CTC یا transducer)،
پوشه‌ی نسبی زیر پوشه‌ی مدل، فایل‌ها با کف سایز و آدرس‌های دانلود، و
متادیتای نمایشی (حجم/سرعت/دقت/پشتیبانی هات‌وورد).

مدل پیش‌فرض همان CTC فعلی اپ است (`shenava-koochik`) و در ریشه‌ی پوشه‌ی
مدل می‌نشیند تا با نصب‌های قبلی سازگار بماند؛ مدل‌های دیگر در زیرپوشه‌ی
خودشان می‌آیند.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# مخزن گیت‌هاب پروژه — assetهای مدل از Release همین تگ دانلود می‌شوند
GITHUB_REPO = "nimaone/persian-whisper-flow"
GITHUB_RELEASE_TAG = "v1.1.0"

# منبع اصلی (HuggingFace) — fallback دو مدل
HF_SHENAVA = "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx/resolve/main"
HF_NEMOTRON = "https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx/resolve/main"

DEFAULT_MODEL_ID = "shenava-koochik"


@dataclass(frozen=True)
class ModelFile:
    """یک فایل مدل با کف سایز (نگهبان دانلود ناقص) و آدرس‌های دانلود."""

    name: str
    min_size: int
    hf_url: str
    # اگر True باشد، اول همان نام از Release گیت‌هاب هم امتحان می‌شود
    on_github: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """توصیف کامل یک مدل — برای دانلود، انتخاب در UI و ساخت موتور."""

    id: str
    name: str            # نام نمایشی کوتاه
    kind: str            # "ctc" | "transducer"
    subdir: str          # پوشه‌ی نسبی زیر پوشه‌ی مدل ("" = ریشه، سازگاری با قبل)
    approx_size: str     # حجم تقریبی برای نمایش
    speed: str
    accuracy: str
    hotwords: bool       # پشتیبانی از هات‌وورد (فقط CTC)
    recommended: bool
    description: str
    files: tuple[ModelFile, ...]
    feature_dim: int = 80

    def dir(self, model_dir: Path) -> Path:
        """پوشه‌ی واقعی این مدل زیر پوشه‌ی مدل فعال."""
        return Path(model_dir) / self.subdir if self.subdir else Path(model_dir)

    def resolve_dir(self, model_dir: Path) -> Path:
        """پوشه‌ی فایل‌های این مدل با تحملِ انتخاب مستقیم کاربر.

        اگر خودِ پوشه‌ی پایه فایل‌های این مدل را داشته باشد (کاربر همان
        پوشه‌ی حاوی فایل‌ها را به‌عنوان «پوشه مدل» انتخاب کرده)، همان
        به‌کار می‌رود؛ وگرنه زیرپوشه‌ی این مدل. مدل پیش‌فرض (subdir خالی)
        همیشه خودِ پایه است.
        """
        base = Path(model_dir)
        d = self.dir(base)
        if d != base and base.is_dir() and all(
                (base / f.name).exists() for f in self.files):
            return base
        return d


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="shenava-koochik",
        name="شنوا کوچیک v1.0",
        kind="ctc",
        subdir="",
        approx_size="~۴۴۰MB",
        speed="سریع",
        accuracy="پایه",
        hotwords=True,
        recommended=True,
        description=("مدل پیش‌فرض: سبک و سریع، سازگار با واژه‌های حساس. "
                     "برای اکثر سیستم‌ها و CPUهای قدیمی مناسب است."),
        files=(
            ModelFile("model.onnx", 100_000_000,
                      f"{HF_SHENAVA}/model.onnx", on_github=True),
            ModelFile("tokens.txt", 10_000,
                      f"{HF_SHENAVA}/tokens.txt", on_github=True),
        ),
    ),
    ModelSpec(
        id="nemotron-farsi",
        name="نیموترون (فارسی)",
        kind="transducer",
        subdir="nemotron",
        approx_size="~۶۶۷MB",
        speed="~۴ برابر کندتر",
        accuracy="دقیق‌تر",
        hotwords=False,
        recommended=False,
        description=("فاین‌تیون ۶۰۰M نیموترون؛ دقت بهتر روی گفتار محاوره‌ای "
                     "(~۵ امتیاز WER). به واژه‌های حساس پشتیبانی ندارد."),
        files=(
            ModelFile("encoder.int8.onnx", 500_000_000,
                      f"{HF_NEMOTRON}/encoder.int8.onnx"),
            ModelFile("decoder.int8.onnx", 5_000_000,
                      f"{HF_NEMOTRON}/decoder.int8.onnx"),
            ModelFile("joiner.int8.onnx", 1_000_000,
                      f"{HF_NEMOTRON}/joiner.int8.onnx"),
            ModelFile("tokens.txt", 10_000,
                      f"{HF_NEMOTRON}/tokens.txt"),
        ),
        feature_dim=128,
    ),
)

_BY_ID = {m.id: m for m in MODELS}


def all_models() -> tuple[ModelSpec, ...]:
    return MODELS


def get(model_id: str | None) -> ModelSpec:
    """مدل با شناسه — ناشناخته/خالی → مدل پیش‌فرض (اپ هرگز بی‌مدل نمی‌ماند)."""
    return _BY_ID.get(str(model_id or "").strip(), _BY_ID[DEFAULT_MODEL_ID])


def default_model() -> ModelSpec:
    return _BY_ID[DEFAULT_MODEL_ID]


def github_url(spec: ModelSpec, file: ModelFile) -> str:
    return (f"https://github.com/{GITHUB_REPO}/releases/download/"
            f"{GITHUB_RELEASE_TAG}/{file.name}")
