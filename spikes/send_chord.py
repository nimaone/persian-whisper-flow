import ctypes, time
user32 = ctypes.windll.user32
INPUT_KEYBOARD, KEYEVENTF_KEYUP = 1, 2
VK = {"ctrl": 0x11, "alt": 0x12, "shift": 0x10}
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_ubyte * 32)]
    _anonymous_ = ("u",); _fields_ = [("type", ctypes.c_ulong), ("u", _U)]
def key(vk, up=False):
    i = _INPUT(); i.type = INPUT_KEYBOARD
    i.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, None)
    user32.SendInput(1, ctypes.byref(i), ctypes.sizeof(_INPUT))
def chord(key_vk, mods):
    for m in mods: key(VK[m]); time.sleep(0.06)
    time.sleep(0.1); key(key_vk); time.sleep(0.05); key(key_vk, up=True)
    for m in reversed(mods): key(VK[m], up=True); time.sleep(0.06)
