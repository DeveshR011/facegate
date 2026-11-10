"""Paths and persisted settings.

Everything lives under one data directory so the lock-screen service (which
runs as SYSTEM) and the desktop app (which runs as the user) see the same
models, templates and settings.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import APP_NAME


def data_dir() -> Path:
    """Root for models, templates and settings. Override with FACEGATE_HOME."""
    override = os.environ.get("FACEGATE_HOME")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / APP_NAME
    else:
        root = Path.home() / ".facegate"
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ProtectedApp:
    path: str
    name: str = ""
    enabled: bool = True


@dataclass
class Settings:
    # --- camera ---
    camera_index: int = 0

    # --- matching (calibrated on LFW, see tools/eval_lfw.py and README) ---
    match_threshold: float = 0.40      # cosine similarity that unlocks on its own
    occlusion_floor: float = 0.30      # a covered face may score this low on the whole face…
    mask_threshold: float = 0.40       # …if the upper-face view scores at least this
    occlusion_unlock: bool = True      # allow that second path at all

    # --- liveness ---
    require_liveness: bool = True
    liveness_threshold: float = 0.55

    # --- session behaviour ---
    frames_required: int = 3           # agreeing frames needed before unlocking
    timeout_s: float = 8.0
    max_failed_attempts: int = 5       # then the password is required
    lockout_s: int = 300
    adaptive_learning: bool = True
    tta_flip: bool = True              # average the embedding with its mirror image

    # --- app guard ---
    guard_enabled: bool = True
    grace_seconds: int = 20            # re-open without a scan within this window
    protected_apps: list[ProtectedApp] = field(default_factory=list)

    # --- windows unlock ---
    windows_unlock: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, raw: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in known}
        apps = [
            ProtectedApp(**{k: v for k, v in a.items() if k in ("path", "name", "enabled")})
            for a in clean.pop("protected_apps", [])
            if isinstance(a, dict) and a.get("path")
        ]
        s = cls(**clean)
        s.protected_apps = apps
        return s


_lock = threading.Lock()


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    p = settings_path()
    with _lock:
        if not p.exists():
            return Settings()
        try:
            return Settings.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            # A corrupt settings file must never stop the unlock path from starting.
            return Settings()


def save_settings(settings: Settings) -> None:
    p = settings_path()
    tmp = p.with_suffix(".tmp")
    with _lock:
        tmp.write_text(settings.to_json(), encoding="utf-8")
        os.replace(tmp, p)
