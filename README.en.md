<div align="center">

<img src="assets/logo.png" width="128" alt="DikteYar logo">

[فارسی](README.md) | **English**

</div>

# DikteYar (دیکته‌یار)

Farsi (Persian) speech-to-text dictation, fully offline and local — triggered by a global hotkey from any app. A Wispr Flow–like experience, without a single byte of audio ever leaving your machine.

> 💚 **Special thanks to [Reza2kn](https://github.com/Reza2kn)**, creator of the [Shenava Koochik v1.0](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) model that powers this app. This project simply would not exist without their open, high-quality model. If you find this app useful, please also star the model and the [Shenava-1 project](https://github.com/Reza2kn/shenava-1).
>
> 🙏 **And thanks to [mallahyari (mehdi-hf)](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi)**, creator of the Persian fine-tune of [Nemotron Streaming](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx) (based on [NVIDIA nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)), now available as the "more accurate" model.

<div align="center">
<a href="docs/demo.mp4"><img src="docs/demo-poster.jpg" width="640" alt="DikteYar video demo"></a>
</div>

🎬 **30-second video demo (with audio)** — dictating into Notepad: the live overlay next to the cursor, the spoken «newline» command, and «بیست و سه» (twenty-three) turned into ۲۳ on insert. (File: [docs/demo.mp4](docs/demo.mp4) — poster: [docs/demo-poster.jpg](docs/demo-poster.jpg))

https://github.com/user-attachments/assets/b8c6a8cd-9f77-40a1-adb9-305340b48e58

🎬 **80-second agent-driven demo (with audio)** — a settings tour, a fix request dictated straight into ZCode, and the agent carrying it out: the Save/Cancel/Reset buttons become equal-width and align with the cards above. (File: [docs/demo-agent.mp4](docs/demo-agent.mp4) — poster: [docs/demo-agent-poster.jpg](docs/demo-agent-poster.jpg))

https://github.com/user-attachments/assets/070fce67-3225-4949-b35c-9d734daae93b

- **Models**: two selectable models — **Shenava Koochik v1.0** (FastConformer NeMo CTC, 114M, light/fast, default) and **Nemotron Streaming Persian** (RNN-T, 600M, more accurate) — both via sherpa-onnx, fully offline
- **Language**: Python 3.12 — CustomTkinter UI (no heavyweight GUI dependencies)
- **Platform**: Windows 10/11 **64-bit (x64)** — **this app is Windows-only** and does not run on Windows ARM (core modules are cross-platform, but the UI, text insertion and installer are built for Windows x64)
- **Version**: 1.2.1 — Windows installer (`DikteYar-Setup-1.2.1-x64.exe`) available under [Releases](https://github.com/nimaone/persian-whisper-flow/releases/latest)

## Features

- Changeable global hotkey + live text overlay window next to the mouse cursor
- Choose the ASR model from Settings (light Shenava / more accurate Nemotron), download it, and change the model folder without reinstalling
- Voice punctuation commands (period, comma, quotes, newline, delete last word)
- Auto-stop after silence (configurable: off/3/5/10 seconds)
- Short start/stop recording beeps
- Minimize to Windows system tray (Hidden Icons) + run at Windows startup
- Automatic active-microphone detection + live spectral mic test
- Text insertion via clipboard (with previous-content restore) or direct Unicode typing
- RTL Persian UI, dark Fluent theme, no white flash on launch

## Screenshots

| | |
|---|---|
| <img src="docs/screenshots/main.png" width="300" alt="Main window"> | <img src="docs/screenshots/overlay.png" width="420" alt="Live overlay"> |
| **Main window** — status and record toggle | **Live overlay** — text appears as you speak |
| <img src="docs/screenshots/settings_general.png" width="300" alt="General settings"> | <img src="docs/screenshots/settings_mic.png" width="300" alt="Microphone settings"> |
| **Settings — General** | **Settings — Microphone** (live spectrum test) |
| <img src="docs/screenshots/settings_insert.png" width="300" alt="Insert settings"> | <img src="docs/screenshots/settings_advanced.png" width="300" alt="Advanced settings"> |
| **Settings — Text insertion** | **Settings — Advanced** (hotwords) |
| <img src="docs/screenshots/settings_help.png" width="300" alt="Help settings"> | |
| **Settings — Help** | |

## Install & Run

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python download_model.py   # fetch the default model (see below)
.venv\Scripts\python app\main.py
```

Two UIs are available — same capture/transcribe logic, different shell:

```bash
.venv\Scripts\python app\main.py          # current UI (CustomTkinter)
.venv\Scripts\python -m flet_ui.run       # new UI (Flet) — full app
.venv\Scripts\python -m flet_ui.run demo  # UI only, no model/microphone
```

> The new Flet UI follows the same lifecycle: no model → first-run dialog to choose/download a model → main app; minimize → tray, close → full exit.

### Getting a model

The ASR models are not stored in this repository. Pick one and fetch it with the CLI:

```bash
.venv\Scripts\python download_model.py                 # default model (Shenava Koochik)
.venv\Scripts\python download_model.py nemotron-farsi  # more accurate model (Nemotron)
.venv\Scripts\python download_model.py --list          # list available models
```

| Model | Architecture | Size | Speed | Accuracy | Hotwords |
|---|---|---|---:|---|---|
| **Shenava Koochik v1.0** (default) | FastConformer NeMo CTC, 114M | ~440 MB | fast | baseline | ✓ |
| **Nemotron Streaming Persian** | RNN-T, 600M | ~667 MB | ~4× slower | more accurate | ✗ |

- Shenava files land in `model/` (`model.onnx` + `tokens.txt`); Nemotron files land in `model/nemotron/` (`encoder/decoder/joiner.int8.onnx` + `tokens.txt`).
- Automatic download first tries this repo's GitHub Releases, then falls back to HuggingFace.
- **Manual** — grab the files from HuggingFace: [Shenava](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) · [Nemotron](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx), and place them in the folders above.
- Switch the active model anytime in Settings → General → the "Speech recognition model" card (the engine rebuilds on Save).
- Nemotron is a transducer: hotword boosting and the "stable live text" mode do not apply to it.

> Note for users in Iran: huggingface.co may be unreachable without a VPN/proxy; the GitHub Releases path (or a proxy) helps there.

## Usage

1. Run the app → tray icon appears (after ~3 s model load: "Ready")
2. In any app (Notepad, Telegram, browser…) press **Ctrl+Shift+Space** and speak
3. The live overlay next to your cursor shows the text being recognized
4. Press the hotkey again → text is inserted at the cursor
5. **Minimize** the main window → app stays in the tray (Hidden Icons); **double-click the tray icon** or use its "Show window" menu to bring it back

## Voice commands

| Say | Result |
|---|---|
| «نقطه» (dot) | . |
| «ویرگول» (comma) | ، |
| «علامت سوال» (question mark) | ؟ |
| «گیومه باز» / «گیومه بسته» | « / » |
| «نقطه ویرگول» (semicolon) | ؛ |
| «خط جدید» (new line) | Enter |
| «حذف آخرین کلمه» (delete last word) | Ctrl+Backspace |

Note: single words that also occur in normal speech (like «سوال») are never treated as commands, so regular sentences stay intact — say the full phrase «علامت سوال» for a question mark.

## Settings

Tray menu → "تنظیمات" (Settings). Five tabs (changes apply on "ذخیره"/Save; "بازنشانی"/Reset restores defaults) — the customtkinter build has the same model-folder card:

- **General (عمومی)**: hotkey capture, start/stop beeps, live overlay + its font size, run at startup, **speech recognition model** (pick light Shenava or more accurate Nemotron + "Download selected model"), **model folder** (use "مرور…"/Browse to point at a folder containing the model files; "پیش‌فرض"/Default is the folder next to the app — an empty chosen folder gets the download, and the change reloads the model on save)
- **Microphone (میکروفون)**: input device ("خودکار"/Auto = strongest-signal mic probed at recording start), live spectral test, auto-stop on silence
- **Text insertion (درج متن)**: insertion method (clipboard recommended / direct typing), clipboard restore, enable/disable voice commands, Persian ITN
- **Advanced (پیشرفته)**: model inference threads (applied after restart), engine about
- **Help (راهنما)**: app version, quick guide, voice commands, troubleshooting

## Project layout

```
model/              active ASR model (Shenava: model.onnx + tokens.txt / Nemotron: model/nemotron/)
app/
  main.py           entry point — tray + hotkey + UI loop + silence auto-stop
  asr.py            sherpa-onnx CTC engine + live/final transcription
  asr_nemotron.py   Nemotron engine / streaming session (RNN-T, sherpa-onnx)
  models_catalog.py model catalog (id/kind/files/download URLs/display metadata)
  engine_factory.py build the active engine based on the selected model (CTC or transducer)
  model_download.py download files for the selected model + partial-download guard
  first_run.py      first-run dialog: choose/download a model (CTk build)
  recorder.py       sounddevice capture + incremental 16kHz resample + mic detection
  overlay.py        frameless rounded live overlay (never steals focus)
  paster.py         insert at cursor: clipboard (backup/restore) or direct typing
  voice_commands.py Persian voice commands → punctuation/keys
  config.py         JSON settings in %APPDATA%\WhisperFlowFarsi + registry autostart
  theme.py          dark Fluent palette + icon loading
  win32.py          DWM dark/rounded windows, flashless show, min/max dialogs
  control_window.py main window (minimize → tray)
  settings_ui.py    five-tab settings + model picker + hotkey capture + mic test
  fonts.py          per-process Vazirmatn font registration (AddFontResourceEx)
flet_ui/            new Flet UI — entry point: run.py (`python -m flet_ui.run`)
download_model.py   model download CLI (selected model or --list)
assets/             logo (logo.png / logo.ico) + Vazirmatn font
spikes/             CLI test tools
```

## Technical notes

- The model is fully offline; the "live" view re-runs inference on the trailing 10-second window every 800 ms (CPU inference latency ≈ 0.04× audio length).
- **Incremental resampling**: each audio block is processed once; the partial only reads the trailing 10-second window — CPU cost is independent of recording length. Buffer is capped at 5 minutes.
- The hotkey is registered with `suppress`: the chosen combo is consumed by this app only.
- Mic auto-detection: all input devices are probed with a 0.35 s signal test and the strongest is picked.
- Clipboard insertion restores the previous clipboard content afterwards (~600 ms wait for slow apps like Electron).
- Windows get dark title bars and rounded corners via DWM and appear with alpha-fade (no white flash).
- Settings live in `%APPDATA%\WhisperFlowFarsi\settings.json`. Keys include `hotkey`, `paste_method`, `voice_commands`, …, `stable_live`, `model_id` (active model id — default `shenava-koochik`, alternative `nemotron-farsi`), and `model_dir` (custom model folder — empty/`null` means the default next to the app).
- **Persian ITN**: spoken numbers («بیست و سه», «سی‌وپنج») are converted to digits (۲۳, ۳۵) on insertion — toggleable in Settings. Small standalone numbers («یک», «هشت») intentionally stay as words.

## Limitations

- Insertion does not work into run-as-administrator apps (Windows limitation).
- Non-text clipboard content (image/file) cannot be restored — the dictation text stays on the clipboard.
- Quality is better on dictated sentences than free conversation.
- The recording buffer is capped at 5 minutes.

## Troubleshooting

- **Nothing is recognized** → run the mic "Test" in Settings; if no signal, try another device or keep "Auto".
- **Text is not inserted** → target app is run-as-administrator; run this app as admin too, or switch to direct typing.
- **Partial insertion** → slow target app; if clipboard mode fails, try direct typing.
- **Hotkey conflicts** → pick a new combo in Settings (e.g. Ctrl+Alt+D).

## Building the Windows exe (branch: feature/exe-installer)

```bash
.venv\Scripts\pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller dikteyar.spec --noconfirm
```

Output: `dist/DikteYar/` — a folder you can zip as a portable build. Notes:

- **onedir, not onefile** — instant startup; onefile extracts ~150 MB to temp on every launch and misbehaves with keyboard hooks and antivirus.
- **The model is not bundled** (light installer) — on first run a dialog asks you to pick a model (**Shenava Koochik** ~440 MB default/fast or **Nemotron** ~667 MB more accurate) then auto-downloads it, or lets you manually select a folder containing the files; the download goes to the folder chosen in settings. Both the model and the model folder can be changed later from the General settings tab.
- `version_info.txt` provides the Windows version resource (name/version/icon in file Properties).
- For a full local test, copy `model/` next to `DikteYar.exe` to skip the first-run dialog.

### Installer (Inno Setup)

```bash
winget install JRSoftware.InnoSetup
"C:\Users\<you>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" installer\dikteyar.iss
```

Output: `installer/Output/DikteYar-Setup-1.2.1-x64.exe` (~36 MB). Highlights: **per-user** install into `%LOCALAPPDATA%\Programs\DikteYar` (no admin required), optional desktop shortcut, run-at-startup checkbox, and the model is excluded from the payload so the installer stays light even if you copied it next to the exe for testing.

## License & Credits

- This app's code ("DikteYar"): **Apache-2.0** — see [LICENSE](LICENSE) (same family as the model's license)
- The "Shenava Koochik v1.0" model by Reza2kn: **Apache-2.0** — see [model/LICENSE](model/LICENSE)
  · [sherpa-onnx model repo](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) · [Shenava-1 project](https://github.com/Reza2kn/shenava-1)
- The "Nemotron Streaming Persian" model (sherpa-onnx int8) by mallahyari (mehdi-hf): **OpenMDW 1.1** — fine-tuned from [mehdi-hf/nemotron-asr-streaming-farsi](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi) on the [NVIDIA nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) architecture
  · [ONNX repo](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx) · [full details](nemotron/hf_upload/README.md)
- Full model details (architecture, training data, metrics): [model/README.md](model/README.md)
- Third-party attributions: [NOTICE](NOTICE)
