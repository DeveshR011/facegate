"""Write assets/facegate.ico from the in-app icon (PNG-compressed entries, Windows Vista+).

    python tools/make_icon.py
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from facegate.ui.app import app_icon  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    app = QGuiApplication([])
    icon = app_icon()
    pngs = []
    for size in SIZES:
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        icon.pixmap(size, size).save(buf, "PNG")
        pngs.append(bytes(data))

    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries = b""
    for size, png in zip(SIZES, pngs):
        dim = 0 if size >= 256 else size            # 0 means 256 in the ICO directory
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
    out = ROOT / "assets" / "facegate.ico"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(header + entries + b"".join(pngs))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, sizes {SIZES})")
    del app


if __name__ == "__main__":
    main()
