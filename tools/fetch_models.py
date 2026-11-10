"""Download the model weights and print their SHA-256 digests.

    python tools/fetch_models.py            # download + verify against pinned hashes
    python tools/fetch_models.py --print    # download and print digests (for re-pinning)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facegate.core import models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="skip verification, print digests")
    args = ap.parse_args()

    if args.print:
        models._verify = lambda path, spec: None  # type: ignore[assignment]

    last = {"pct": -1}

    def progress(label: str, done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct // 10 != last["pct"] // 10:
            print(f"  {label}: {pct}%", flush=True)
        last["pct"] = pct

    paths = models.ensure_models(progress)
    for name, path in paths.items():
        print(f"{name:14s} {path.name:22s} {path.stat().st_size:>10d}  {models.sha256_of(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
