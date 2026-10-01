"""لانچر مستقل پنجره تنظیمات CTk — بدون مدل/میکروفون/تری."""
import os
import tkinter as tk

from app.settings_ui import open_settings

root = tk.Tk()
root.withdraw()
open_settings(root, app=None)

def _watch():
    # وقتی پنجره تنظیمات بسته شد، پروسه تمام شود
    try:
        for w in root.winfo_children():
            if w.winfo_exists():
                root.after(500, _watch)
                return
    except Exception:
        pass
    os._exit(0)

root.after(1000, _watch)
root.mainloop()
