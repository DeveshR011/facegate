"""DPAPI wrappers (ctypes only, so the core has no pywin32 dependency)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

CRYPTPROTECT_UI_FORBIDDEN = 0x1
CRYPTPROTECT_LOCAL_MACHINE = 0x4

# Binds ciphertexts to this application; not a secret.
_ENTROPY = b"FaceGate/templates/v1"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _call(fn, data: bytes, flags: int) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    src, ent, out = _blob(data), _blob(_ENTROPY), _Blob()
    ok = getattr(crypt32, fn)(
        ctypes.byref(src), None, ctypes.byref(ent), None, None, flags, ctypes.byref(out)
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def protect(data: bytes, machine_scope: bool = True) -> bytes:
    """Encrypt with DPAPI.

    machine_scope=True lets any account on *this* PC decrypt (needed so the
    SYSTEM lock-screen service can read templates enrolled by the user) while
    keeping the file useless if it is copied to another machine.
    """
    flags = CRYPTPROTECT_UI_FORBIDDEN | (CRYPTPROTECT_LOCAL_MACHINE if machine_scope else 0)
    return _call("CryptProtectData", data, flags)


def unprotect(data: bytes) -> bytes:
    return _call("CryptUnprotectData", data, CRYPTPROTECT_UI_FORBIDDEN)
