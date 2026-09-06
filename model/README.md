# شنوا کوچیک v1.0 — Shenava Koochik v1.0

اطلاعات کامل مدل ASR که این اپ استفاده می‌کند. منبع: [HuggingFace](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) و [GitHub پروژه](https://github.com/Reza2kn/shenava-1).

## مشخصات کلی

| | |
|---|---|
| **نام مدل** | Shenava Koochik v1.0 (شنوا کوچیک) |
| **سازنده** | Reza2kn — پروژه Shenava-1 / VisualEars |
| **تعداد پارامتر** | ۱۱۴ میلیون (چک‌پوینت FP32 NeMo ≈ ۴۶۰MB) |
| **معماری** | FastConformer Hybrid RNNT/CTC — سر CTC در خروجی‌های دیپلوی استفاده می‌شود |
| **انکودر** | `d_model=512`، ۱۷ لایه، subsampling ۸ برابر (هر قدم انکودر ≈ ۸۰ms صوت) |
| **ورودی** | گفتار فارسی مونو ۱۶kHz |
| **توکنایزر** | `ve_tok_v4` — SentencePiece BPE با ۱۰۲۴ توکن + CTC blank |
| **مدل پایه (fine-tune از)** | [nvidia/stt_fa_fastconformer_hybrid_large](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) (~۱۱۵M) |
| **مجوز** | Apache-2.0 |

### تطابق با فایل ONNX این اپ

متادیتای `model.onnx` موجود در همین پوشه دقیقاً همین را تأیید می‌کند:

- `vocab_size: 1025` ← ۱۰۲۴ توکن BPE + blank
- `subsampling_factor: 8`
- `feat_dim: 80` (فیچر mel ۸۰ بُعدی)
- `model_type: EncDecCTCModel`
- ورودی `audio_signal [N, 80, T]` ← خروجی `log_probs [N, T_out, 1025]`

## استریمینگ (cache-aware)

مدل با contextهای توجه زیر استریم می‌شود:

- `[70, 13]` (پیش‌فرض دیکته، دقیق‌ترین)
- `[70, 6]`
- `[70, 1]`
- `[70, 0]` (کم‌ترین تأخیر)

## دیتاست‌های آموزش

| دیتاست | حجم | نقش |
|---|---|---|
| [visualears-persian-asr-16k](https://huggingface.co/datasets/Reza2kn/visualears-persian-asr-16k) | ۳٬۹۲۷٬۷۳۳ ردیف ≈ **۷٬۷۷۱ ساعت** صوت (۵۱۲GB، ۱۵۸ shard Parquet، ۱۶kHz) | پیکره اصلی آموزش |
| [persian-asr-relabeled-gemini](https://huggingface.co/datasets/Reza2kn/persian-asr-relabeled-gemini) | ۴۵۴٬۶۷۵ ردیف ≈ ۶۹۰ ساعت | بازبرچسب‌گذاری با Gemini |
| [visualears-115m-pref350-curriculum-relabels](https://huggingface.co/datasets/Reza2kn/visualears-115m-pref350-curriculum-relabels) | ۲۳۱٬۹۵۹ ردیف ≈ ۴۱۲ ساعت | زیرمجموعه فیلترشده مخصوص Koochik |
| [visualears-al-114m-audio](https://huggingface.co/datasets/Reza2kn/visualears-al-114m-audio) + [hardword-sentences](https://huggingface.co/datasets/Reza2kn/visualears-hardword-sentences) | ۶٬۹۶۹ asset | یادگیری فعال و جملات سخت مخصوص Koochik |

دو دیتاست دیگرِ ذکرشده در مدل‌کارت — [visualears-golden-6669](https://huggingface.co/datasets/Reza2kn/visualears-golden-6669) (۶٬۶۶۹ رکورد) و [fleurs-fa-benchmark](https://huggingface.co/datasets/Reza2kn/fleurs-fa-benchmark) — **فقط برای ارزیابی هستند و جزو آموزش نیستند.**

## کیفیت منتشرشده

دیکد با context `[70,13]` و نرمال‌سازی ITN/رقم فارسی:

| بنچمارک | WER | CER |
|---|---:|---:|
| visualears-golden-6669 | **۷٫۴۹٪** | ۲٫۳۰٪ |
| FLEURS-fa | **۱۰٫۶۴٪** | ۳٫۷۹٪ |

## نکته درباره اعداد

مدل‌کارت صراحتاً می‌گوید: «اعداد به فرم تلفظی خروجی می‌آیند؛ برای رقم فارسی باید ITN اعمال کنید.»
پس محدودیت README اصلی اپ («بیست و سه ← ۲۳ تبدیل نمی‌شود») خاصیت عمدی داده‌های آموزش این مدل است، نه باگ اپ — خود مدل ITN ندارد و مؤلف آن را به مصرف‌کننده واگذار کرده است.

## تنظیمات دقیق آموزش

هایپرپارامترهای دقیق (learning rate، batch size، epochs) در مدل‌کارت نیامده و در [مقاله SLT](https://openreview.net/forum?id=QTa6ax9PU3) پروژه است. مدل پایه NVIDIA با اسکریپت استاندارد [`speech_to_text_finetune.py`](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_finetune.py) NeMo آموزش دیده.

## خانواده مدل (Shenava-1)

| مدل | پارامتر | کاربرد |
|---|---:|---|
| **Koochik 114M** (این اپ) | ۱۱۴M | مدل پرچم‌دار — دیکته و زیرنویس |
| [Rizeh 32M](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | ۳۲M | نسخه سبک‌تر |
| [Rizeh-Pizeh 6.9M](https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0) | ۶٫۹M | نسخه فوق‌سبک — حتی روی ESP32-S3 اجرا می‌شود |
| [ShenavaSanj 0.2B](https://huggingface.co/Reza2kn/ShenavaSanj-v1.0) | ۰٫۲B | مدل اهمیت واژه برای ارزیابی Semantic WER |

## لینک‌ها

- مدل اصلی: https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0
- نسخه sherpa-onnx (همان که این اپ استفاده می‌کند): https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx
- مخزن مرکزی پروژه: https://github.com/Reza2kn/shenava-1
- مقاله SLT: https://openreview.net/forum?id=QTa6ax9PU3
- اپ رسمی فارسی: https://shenava.app
