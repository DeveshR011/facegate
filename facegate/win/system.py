"""Elevation, password verification, LSA secrets and process helpers."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path

SECRET_NAME = "L$FaceGateUnlock"        # "L$" = local secret, readable by SYSTEM / admins only


# ---- elevation -----------------------------------------------------------

def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False


def python_gui_exe() -> str:
    """pythonw.exe next to the running interpreter (no console window)."""
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


class _ShellExecuteInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
        ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE),
    ]


def run_elevated(args: list[str], wait: bool = True, exe: str | None = None) -> int | None:
    """Start `exe args…` through a UAC prompt.

    Returns None when the process was not started (the user declined the
    prompt), otherwise its exit code - or 0 straight away when wait=False.
    """
    import subprocess

    SEE_MASK_NOCLOSEPROCESS = 0x40
    info = _ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = exe or python_gui_exe()
    info.lpParameters = subprocess.list2cmdline(args)
    info.nShow = 1
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        return None
    kernel32 = ctypes.windll.kernel32
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    if not info.hProcess:
        return 0
    if not wait:
        kernel32.CloseHandle(info.hProcess)
        return 0
    kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
    kernel32.CloseHandle(info.hProcess)
    return int(code.value)


# ---- account / password --------------------------------------------------

def current_account() -> tuple[str, str]:
    """(domain, user) of the interactive user, e.g. ('LAPTOP-AB12', 'rawat')."""
    import win32api
    import win32con

    sam = win32api.GetUserNameEx(win32con.NameSamCompatible)
    domain, _, user = sam.partition("\\")
    return domain, user


def current_sid() -> str:
    import win32api
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), 0x0008)  # TOKEN_QUERY
    sid, _ = win32security.GetTokenInformation(token, win32security.TokenUser)
    return win32security.ConvertSidToStringSid(sid)


def check_credentials(domain: str, user: str, password: str) -> bool:
    import pywintypes
    import win32con
    import win32security

    try:
        handle = win32security.LogonUser(
            user, domain, password, win32con.LOGON32_LOGON_NETWORK, win32con.LOGON32_PROVIDER_DEFAULT
        )
    except pywintypes.error:
        return False
    handle.Close()
    return True


def verify_password(password: str, microsoft_account: str = "") -> tuple[str, str] | None:
    """Check the password for the signed-in user.

    Returns the (domain, user) pair that validated, or None. Windows accounts
    linked to a Microsoft account validate either under the local name or as
    MicrosoftAccount\\<email>.
    """
    if not password:
        return None
    candidates = [current_account()]
    if microsoft_account:
        candidates.insert(0, ("MicrosoftAccount", microsoft_account))
    for domain, user in candidates:
        if check_credentials(domain, user, password):
            return domain, user
    return None


# ---- LSA private data (holds the Windows credential for lock-screen unlock) ----

def _policy():
    import win32security

    return win32security.LsaOpenPolicy(None, win32security.POLICY_ALL_ACCESS)


def store_unlock_secret(domain: str, user: str, password: str, sid: str, local_user: str) -> None:
    """Requires elevation. The secret never touches the file system in the clear."""
    import win32security

    payload = json.dumps({"domain": domain, "user": user, "password": password, "sid": sid,
                          "local_user": local_user})
    policy = _policy()
    try:
        win32security.LsaStorePrivateData(policy, SECRET_NAME, payload)
    finally:
        win32security.LsaClose(policy)


def load_unlock_secret() -> dict | None:
    import pywintypes
    import win32security

    try:
        policy = _policy()
    except pywintypes.error:
        return None
    try:
        raw = win32security.LsaRetrievePrivateData(policy, SECRET_NAME)
    except pywintypes.error:
        return None
    finally:
        win32security.LsaClose(policy)
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-16-le")
    try:
        return json.loads(raw)
    except ValueError:
        return None


def delete_unlock_secret() -> None:
    import pywintypes
    import win32security

    policy = _policy()
    try:
        win32security.LsaStorePrivateData(policy, SECRET_NAME, None)
    except pywintypes.error:
        pass
    finally:
        win32security.LsaClose(policy)


# ---- executables ---------------------------------------------------------

def norm_exe(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def exe_display_name(path: str) -> str:
    """FileDescription from the version resource, falling back to the file name."""
    try:
        import win32api

        langs = win32api.GetFileVersionInfo(path, r"\VarFileInfo\Translation")
        for lang, codepage in langs:
            key = rf"\StringFileInfo\{lang:04x}{codepage:04x}\FileDescription"
            name = win32api.GetFileVersionInfo(path, key)
            if name and name.strip():
                return name.strip()
    except Exception:
        pass
    return Path(path).stem


def running_apps() -> list[tuple[str, str]]:
    """(name, exe path) for every process that owns a visible top-level window."""
    import psutil
    import win32gui
    import win32process

    pids: set[int] = set()

    def visit(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            pids.add(win32process.GetWindowThreadProcessId(hwnd)[1])
        return True

    win32gui.EnumWindows(visit, None)
    seen: dict[str, str] = {}
    for pid in pids:
        try:
            exe = psutil.Process(pid).exe()
        except (psutil.Error, OSError):
            continue
        if exe and norm_exe(exe) not in seen:
            seen[norm_exe(exe)] = exe
    return sorted(((exe_display_name(p), p) for p in seen.values()), key=lambda t: t[0].lower())
