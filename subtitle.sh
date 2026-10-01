#!/usr/bin/env bash
# استخراج زیرنویس با موتور دیکته‌یار — پوشش ساده روی subtitle.py
# مصرف: ./subtitle.sh FILE [FILE...] [--chunk 16] [--out DIR]
set -euo pipefail
cd "$(dirname "$0")"
exec ".venv/Scripts/python.exe" subtitle.py "$@"
