"""Explorer's right-click "Unlock with FaceGate" for locked files and folders.

Registered per user (HKCU, no administrator rights) while at least one item is locked, removed
when none is. On Windows 11 it sits under "Show more options". The command starts FaceGate (or
hands the request to the running copy), which asks for your face and then opens the item.
"""

from __future__ import annotations

import winreg
from pathlib import Path

from .system import app_command, current_exe, is_compiled

VERB = "FaceGate.Unlock"
KEYS = (rf"Software\Classes\*\shell\{VERB}", rf"Software\Classes\Directory\shell\{VERB}")


def command() -> tuple[str, str]:
    """(command line with %1, icon) for the verb."""
    if is_compiled():
        exe, args = app_command(["open"])
        icon = exe
    else:
        # Not "-m facegate": Explorer starts verbs in the item's folder, where the package is not
        # importable. The launcher script puts the source tree on sys.path itself.
        root = Path(__file__).resolve().parents[2]
        exe, args = app_command([])[0], [str(root / "launch_facegate.py"), "open"]
        icon = str(root / "assets" / "facegate.ico")
    return " ".join(f'"{a}"' for a in (exe, *args)) + ' "%1"', icon


def is_registered() -> bool:
    try:
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEYS[0] + r"\command"))
        return True
    except OSError:
        return False


def register() -> None:
    cmd, icon = command()
    for key in KEYS:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as k:
            winreg.SetValueEx(k, "MUIVerb", 0, winreg.REG_SZ, "Unlock with FaceGate")
            winreg.SetValueEx(k, "Icon", 0, winreg.REG_SZ, icon)
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key + r"\command") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, cmd)


def unregister() -> None:
    for key in KEYS:
        for sub in (key + r"\command", key):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            except FileNotFoundError:
                pass


def set_registered(on: bool) -> None:
    """Idempotent; failures are not fatal (the window and the tray still unlock items)."""
    try:
        if on:
            register()
        elif is_registered():
            unregister()
    except OSError:
        pass
