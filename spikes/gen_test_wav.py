"""Generate a Persian test wav using Windows SAPI TTS (if a fa-IR voice exists)."""
import sys
import os

try:
    import win32com.client
except ImportError:
    print("pywin32 not installed")
    sys.exit(1)

out_dir = os.path.join(os.path.dirname(__file__), "test_audio")
os.makedirs(out_dir, exist_ok=True)


def list_voices():
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    voices = sp.GetVoices()
    names = []
    for i in range(voices.Count):
        names.append(voices.Item(i).GetDescription())
    return names


def synth_to_wav(text: str, wav_path: str, voice_substr: str = ""):
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    if voice_substr:
        voices = sp.GetVoices()
        for i in range(voices.Count):
            desc = voices.Item(i).GetDescription()
            if voice_substr.lower() in desc.lower():
                sp.Voice = voices.Item(i)
                break
    fmt = win32com.client.Dispatch("SAPI.SpFileStreamFormat")
    # 16kHz 16-bit mono
    fmt.Type = 16  # SAFT16kHz16BitMono? enum values vary; try 16
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Format = fmt
    stream.Open(wav_path, 3, False)  # SSFMCreateForWrite = 3
    sp.AudioOutputStream = stream
    sp.Speak(text, 3)  # SVSFDefault | SVSFlagsAsync? use 0 sync
    stream.Close()


if __name__ == "__main__":
    print("voices:")
    for n in list_voices():
        print("  -", n)
