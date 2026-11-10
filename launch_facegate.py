"""Entry point for the Nuitka build (the package's own __main__ cannot be compiled as a script)."""

from facegate.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
