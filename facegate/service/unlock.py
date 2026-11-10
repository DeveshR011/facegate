r"""Lock-screen recognition service.

Runs as SYSTEM (scheduled task, at boot). The credential provider DLL inside
LogonUI connects to \\.\pipe\FaceGateUnlock and sends:

    AUTH        -> scan; reply  OK\t<domain>\t<user>\t<password>   or   FAIL\t<reason>
    REJECTED    -> Windows refused the credential we returned; stop offering it

Messages are UTF-16LE. The pipe's DACL admits SYSTEM only, so the credential
can only ever be handed to the logon UI.

Face unlock is refused until the user has signed in with their password once
since boot - the same rule Face ID applies after a restart.
"""

from __future__ import annotations

import logging
import time

import pywintypes
import win32file
import win32pipe
import win32security
import win32ts

from ..config import data_dir, load_settings
from ..core.auth import AuthSession, AuthState
from ..core.camera import Camera, CameraError
from ..core.limiter import AttemptLimiter
from ..core.store import TemplateStore
from ..win.system import load_unlock_secret

log = logging.getLogger("facegate.service")

PIPE_NAME = r"\\.\pipe\FaceGateUnlock"
REJECTED_FLAG = "unlock_rejected.flag"
BUF = 8192


def _security_attributes() -> win32security.SECURITY_ATTRIBUTES:
    sd = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
        "D:P(A;;GA;;;SY)", win32security.SDDL_REVISION_1
    )
    sa = win32security.SECURITY_ATTRIBUTES()
    sa.SECURITY_DESCRIPTOR = sd
    sa.bInheritHandle = False
    return sa


def user_has_session(local_user: str) -> bool:
    """True when the user already has a (locked) desktop session."""
    try:
        for s in win32ts.WTSEnumerateSessions(win32ts.WTS_CURRENT_SERVER_HANDLE):
            name = win32ts.WTSQuerySessionInformation(
                win32ts.WTS_CURRENT_SERVER_HANDLE, s["SessionId"], win32ts.WTSUserName
            )
            if name and name.lower() == local_user.lower():
                return True
    except pywintypes.error:
        log.exception("session enumeration failed")
    return False


class UnlockService:
    def __init__(self) -> None:
        from ..core.engine import FaceEngine

        self.engine = FaceEngine(allow_download=False)
        self.store = TemplateStore()
        s = load_settings()
        self.limiter = AttemptLimiter(data_dir() / "attempts_service.json",
                                      s.max_failed_attempts, s.lockout_s, clock=time.time)
        self.flag = data_dir() / REJECTED_FLAG

    def _client_alive(self, pipe) -> bool:
        try:
            win32pipe.PeekNamedPipe(pipe, 0)
            return True
        except pywintypes.error:
            return False

    def authenticate(self, pipe) -> str:
        settings = load_settings()
        if not settings.windows_unlock:
            return "FAIL\tdisabled"
        secret = load_unlock_secret()
        if not secret:
            return "FAIL\tnot configured"
        if self.flag.exists():
            return "FAIL\tPassword changed - update it in FaceGate"
        if not user_has_session(secret.get("local_user") or secret["user"]):
            return "FAIL\tPassword required after restart"
        if self.limiter.locked:
            return "FAIL\tToo many attempts - use your password"

        self.store.refresh()
        try:
            with Camera(settings.camera_index) as cam:
                cam.warm_up()
                session = AuthSession(self.engine, self.store, settings)
                while True:
                    if not self._client_alive(pipe):
                        return "FAIL\tcancelled"
                    frame = cam.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    update = session.feed(frame)
                    if update.done:
                        break
        except CameraError as exc:
            log.warning("camera: %s", exc)
            return "FAIL\tCamera unavailable"

        if update.ok:
            self.limiter.record_success()
            log.info("match (%s view, score %.3f, liveness %.2f)", update.view, update.score,
                     update.liveness)
            return "OK\t{domain}\t{user}\t{password}".format(**secret)
        if update.state in (AuthState.FAILED, AuthState.SPOOF):
            self.limiter.record_failure()
        log.info("no match: %s", update.state.value)
        return f"FAIL\t{update.message}"

    def handle(self, pipe) -> None:
        _, data = win32file.ReadFile(pipe, BUF)
        command = data.decode("utf-16-le", errors="replace").strip("\x00\r\n ")
        if command == "AUTH":
            reply = self.authenticate(pipe)
        elif command == "REJECTED":
            self.flag.write_text(str(time.time()))
            log.warning("Windows rejected the stored credential; face unlock paused")
            reply = "OK"
        else:
            reply = "FAIL\tbad request"
        win32file.WriteFile(pipe, reply.encode("utf-16-le"))

    def serve_forever(self) -> None:
        sa = _security_attributes()
        first = True
        log.info("listening on %s (%s)", PIPE_NAME, self.engine.provider)
        while True:
            flags = win32pipe.PIPE_ACCESS_DUPLEX
            if first:
                # Refuse to run if someone else already owns the pipe name.
                flags |= win32file.FILE_FLAG_FIRST_PIPE_INSTANCE
            pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME, flags,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE
                | win32pipe.PIPE_WAIT | win32pipe.PIPE_REJECT_REMOTE_CLIENTS,
                1, BUF, BUF, 0, sa,
            )
            first = False
            try:
                win32pipe.ConnectNamedPipe(pipe, None)
                self.handle(pipe)
                win32file.FlushFileBuffers(pipe)
            except pywintypes.error as exc:
                log.info("client dropped: %s", exc)
            except Exception:
                log.exception("request failed")
            finally:
                try:
                    win32pipe.DisconnectNamedPipe(pipe)
                except pywintypes.error:
                    pass
                win32file.CloseHandle(pipe)


def main() -> int:
    log_dir = data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(log_dir / "service.log", maxBytes=512_000, backupCount=2,
                                  encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        UnlockService().serve_forever()
    except Exception:
        log.exception("service stopped")
        return 1
    return 0
