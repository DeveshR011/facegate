"""Nuitka user plugin for FaceGate - loaded by tools/build_exe.ps1 with --user-plugin.

  bundle  the C++ engine, its ONNX Runtime / DirectML runtime, the prepared models and the
          lock-screen DLL, under native\\ next to FaceGate.exe (where facegate/core/native.py looks)
  shrink  keep out modules FaceGate can never import on Windows
  speed   ship large constant tables as bytecode instead of compiling them to C

DLLs cannot be vetoed from a plugin, so the two DLL-level trims (OpenCV's ffmpeg plugin and unused
Qt plugins) stay as --noinclude-* flags in tools/build_exe.ps1.
"""

from __future__ import annotations

import os
from pathlib import Path

from nuitka.plugins.PluginBase import NuitkaPluginBase

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine" / "build" / "Release"
MODELS = ROOT / "engine" / "models"
NATIVE = {                                        # source -> path inside the distribution
    ENGINE / "facegate_engine.dll": "facegate_engine.dll",
    ENGINE / "onnxruntime.dll": "onnxruntime.dll",
    ENGINE / "onnxruntime_providers_shared.dll": "onnxruntime_providers_shared.dll",
    ENGINE / "DirectML.dll": "DirectML.dll",
    ROOT / "credential_provider" / "build" / "Release" / "FaceGateCP.dll": "FaceGateCP.dll",
}

# psutil's backends for other systems. Never imported on Windows, yet followed statically:
# psutil._pslinux alone was 4 MB of generated C.
OTHER_OS = ("psutil._pslinux", "psutil._psosx", "psutil._psbsd", "psutil._pssunos",
            "psutil._psaix", "psutil._psposix")

# pywin32's constant tables: winerror is 9 MB of generated C and win32con 4.6 MB - the two largest
# files in the build - for lookups that run at C speed either way.
AS_BYTECODE = ("winerror", "win32con")


class FaceGatePlugin(NuitkaPluginBase):
    plugin_name = "facegate"
    plugin_desc = "Bundles FaceGate's native engine and models; trims what Windows never loads."

    @staticmethod
    def isAlwaysEnabled():
        return True

    def onCompilationStartChecks(self):
        missing = [str(p) for p in (*NATIVE, MODELS / "manifest.json") if not p.exists()]
        if missing:
            self.sysexit("Missing build inputs - see the prerequisites in tools/build_exe.ps1:\n  "
                         + "\n  ".join(missing))

    def considerDataFiles(self, module):
        if module.getFullName() != "facegate":
            return
        for source, name in NATIVE.items():
            yield self.makeIncludedDataFile(str(source), os.path.join("native", name),
                                            "FaceGate C++ engine and runtime")
        for model in sorted(MODELS.iterdir()):
            if model.is_file():
                yield self.makeIncludedDataFile(str(model), os.path.join("native", "models", model.name),
                                                "FaceGate model")

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if module_name in OTHER_OS:
            return False, "FaceGate runs on Windows only"
        return None

    def decideCompilation(self, module_name):
        return "bytecode" if module_name in AS_BYTECODE else None
