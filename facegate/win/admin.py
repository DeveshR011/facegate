"""Operations that need administrator rights.

Run through a UAC prompt as:  pythonw -m facegate admin <action> [--user DOMAIN\\name --sid S-1-…]

    autostart-on / autostart-off     start FaceGate (elevated) at sign-in, without a UAC prompt
    unlock-install / unlock-uninstall   lock-screen credential provider + background service

Nothing here is executed implicitly; each action is started by an explicit
click in the settings window.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import winreg
from pathlib import Path
from xml.sax.saxutils import escape

from ..config import data_dir
from .system import app_command, delete_unlock_secret, elevation_safe, is_admin, is_compiled

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # repo root, or the unpacked Nuitka bundle
TASK_APP = r"FaceGate\App"
TASK_SERVICE = r"FaceGate\UnlockService"

PROVIDER_CLSID = "{7B3C9F2E-5A41-4D8B-9E6C-2F1A8D4B6C73}"
PROVIDER_NAME = "FaceGate Credential Provider"
PROVIDER_DLL = "FaceGateCP.dll"
_CP_KEY = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Authentication\Credential Providers\{PROVIDER_CLSID}"
_CLSID_KEY = rf"SOFTWARE\Classes\CLSID\{PROVIDER_CLSID}"
_APP_KEY = r"SOFTWARE\FaceGate"

EXIT_OK, EXIT_FAILED, EXIT_NOT_ADMIN, EXIT_NOT_BUILT, EXIT_NOT_INSTALLED = 0, 1, 2, 3, 4


def _unsafe_for_elevation(exe: str) -> bool:
    """A compiled FaceGate that runs elevated must live where only administrators can write."""
    if not elevation_safe():
        print(f"Refusing to register an elevated task for {exe}: it, or the folder its code is "
              "unpacked to, is writable by ordinary programs. Install the folder build to "
              "Program Files first (Install-FaceGate.ps1).")
        return True
    return False


def built_provider_dll() -> Path | None:
    for candidate in (
        PROJECT_ROOT / "native" / PROVIDER_DLL,                  # bundled in the exe
        PROJECT_ROOT / "credential_provider" / "build" / "Release" / PROVIDER_DLL,
        PROJECT_ROOT / "credential_provider" / "build" / PROVIDER_DLL,
    ):
        if candidate.exists():
            return candidate
    return None


def installed_provider_dll() -> Path:
    return Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "FaceGate" / PROVIDER_DLL


# ---- scheduled tasks -----------------------------------------------------

def _task_xml(description: str, trigger: str, principal: str, command: tuple[str, list[str]],
              restart: bool) -> str:
    exe, args = command
    workdir = str(Path(exe).parent) if is_compiled() else str(PROJECT_ROOT)
    restart_xml = (
        "<RestartOnFailure><Interval>PT1M</Interval><Count>5</Count></RestartOnFailure>"
        if restart else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>{escape(description)}</Description></RegistrationInfo>
  <Triggers>{trigger}</Triggers>
  <Principals><Principal id="Author">{principal}</Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
    <!-- Tasks default to 7 (below normal CPU, I/O and memory priority), meant for batch jobs.
         FaceGate has to freeze a protected app the moment it starts, and it is idle otherwise. -->
    <Priority>4</Priority>
    {restart_xml}
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(exe)}</Command>
      <Arguments>{escape(subprocess.list2cmdline(args))}</Arguments>
      <WorkingDirectory>{escape(workdir)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW)


def _create_task(name: str, xml: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-16") as f:
        f.write(xml)
        path = f.name
    try:
        r = _schtasks("/Create", "/TN", name, "/XML", path, "/F")
        if r.returncode != 0:
            raise RuntimeError(f"schtasks failed: {r.stderr.strip() or r.stdout.strip()}")
    finally:
        os.unlink(path)


def task_exists(name: str) -> bool:
    return _schtasks("/Query", "/TN", name).returncode == 0


def autostart_on(user: str) -> int:
    command = app_command(["--tray"])
    if _unsafe_for_elevation(command[0]):
        return EXIT_NOT_INSTALLED
    trigger = f"<LogonTrigger><Enabled>true</Enabled><UserId>{escape(user)}</UserId></LogonTrigger>"
    principal = (f"<UserId>{escape(user)}</UserId><LogonType>InteractiveToken</LogonType>"
                 "<RunLevel>HighestAvailable</RunLevel>")
    _create_task(TASK_APP, _task_xml("FaceGate app lock (starts in the tray)", trigger, principal,
                                     command, restart=False))
    return EXIT_OK


def autostart_off() -> None:
    _schtasks("/Delete", "/TN", TASK_APP, "/F")


# ---- lock-screen unlock --------------------------------------------------

def _set_reg(root, key: str, name: str | None, value: str) -> None:
    with winreg.CreateKeyEx(root, key, 0, winreg.KEY_WRITE | winreg.KEY_WOW64_64KEY) as k:
        winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)


def _delete_tree(root, key: str) -> None:
    try:
        with winreg.OpenKey(root, key, 0, winreg.KEY_ALL_ACCESS | winreg.KEY_WOW64_64KEY) as k:
            while True:
                try:
                    _delete_tree(root, key + "\\" + winreg.EnumKey(k, 0))
                except OSError:
                    break
        winreg.DeleteKeyEx(root, key, winreg.KEY_WOW64_64KEY)
    except OSError:
        pass


def _lock_down_data_dir(user: str) -> None:
    """Only SYSTEM, administrators and the owner may touch templates and settings:
    whoever can write the template file can enroll a face for the lock screen."""
    subprocess.run(
        ["icacls", str(data_dir()), "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F",
         "*S-1-5-32-544:(OI)(CI)F", f"{user}:(OI)(CI)M", "/T", "/C", "/Q"],
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )


def unlock_install(user: str, sid: str) -> int:
    src = built_provider_dll()
    if src is None:
        print("The credential provider has not been built. Run credential_provider\\build.ps1 first.")
        return EXIT_NOT_BUILT
    command = app_command(["service"])
    if _unsafe_for_elevation(command[0]):
        return EXIT_NOT_INSTALLED
    dst = installed_provider_dll()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    _set_reg(winreg.HKEY_LOCAL_MACHINE, _APP_KEY, "UserSid", sid)
    _set_reg(winreg.HKEY_LOCAL_MACHINE, _CLSID_KEY, None, PROVIDER_NAME)
    _set_reg(winreg.HKEY_LOCAL_MACHINE, _CLSID_KEY + r"\InprocServer32", None, str(dst))
    _set_reg(winreg.HKEY_LOCAL_MACHINE, _CLSID_KEY + r"\InprocServer32", "ThreadingModel", "Apartment")
    _set_reg(winreg.HKEY_LOCAL_MACHINE, _CP_KEY, None, PROVIDER_NAME)

    _lock_down_data_dir(user)
    trigger = "<BootTrigger><Enabled>true</Enabled></BootTrigger>"
    principal = "<UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel>"
    _create_task(TASK_SERVICE, _task_xml("FaceGate lock-screen recognition service", trigger,
                                         principal, command, restart=True))
    _schtasks("/Run", "/TN", TASK_SERVICE)
    return EXIT_OK


def unlock_uninstall() -> int:
    _schtasks("/End", "/TN", TASK_SERVICE)
    _schtasks("/Delete", "/TN", TASK_SERVICE, "/F")
    _delete_tree(winreg.HKEY_LOCAL_MACHINE, _CP_KEY)
    _delete_tree(winreg.HKEY_LOCAL_MACHINE, _CLSID_KEY)
    _delete_tree(winreg.HKEY_LOCAL_MACHINE, _APP_KEY)
    delete_unlock_secret()
    try:
        installed_provider_dll().unlink(missing_ok=True)
    except OSError:
        pass        # LogonUI may still have it mapped; it is inert once unregistered
    return EXIT_OK


def unlock_installed() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CP_KEY, 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
            return True
    except OSError:
        return False


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="facegate admin")
    ap.add_argument("action", choices=["autostart-on", "autostart-off", "unlock-install",
                                       "unlock-uninstall"])
    ap.add_argument("--user", default="")
    ap.add_argument("--sid", default="")
    args = ap.parse_args(argv)
    if not is_admin():
        return EXIT_NOT_ADMIN
    try:
        if args.action == "autostart-on":
            return autostart_on(args.user)
        elif args.action == "autostart-off":
            autostart_off()
        elif args.action == "unlock-install":
            return unlock_install(args.user, args.sid)
        elif args.action == "unlock-uninstall":
            return unlock_uninstall()
        return EXIT_OK
    except Exception:
        log.exception("admin action %s failed", args.action)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
