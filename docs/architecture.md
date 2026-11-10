# FaceGate architecture

## Shape of the system

```mermaid
flowchart LR
    subgraph exe["FaceGate.exe  (Nuitka-compiled Python + Qt)"]
        UI["ui/<br/>overlay, enrollment, settings, themes"]
        Guard["guard/<br/>process watcher + trust policy"]
        Locks["guard/locks.py<br/>file & folder lock"]
        Auth["core/auth.py<br/>state machines, decision rule"]
        Store["core/store.py<br/>DPAPI-sealed templates"]
        Native["core/native.py<br/>ctypes binding"]
    end
    subgraph cpp["facegate_engine.dll  (C++17, C ABI)"]
        Det["SCRFD detect<br/>+ resize / decode / NMS"]
        Align["align 112x112"]
        Emb["ArcFace embed<br/>full + mask view, mirror TTA"]
        Live["MiniFASNet x2 liveness"]
    end
    ORT["onnxruntime.dll<br/>DirectML EP"] --> GPU[("RTX 4050")]
    UI --> Auth
    Guard --> UI
    Locks --> UI
    Guard -.->|command lines of new processes| Locks
    Locks --> ACL[("NTFS: deny rule<br/>for INTERACTIVE")]
    Verb["Explorer: Unlock with FaceGate"] -.->|local socket| UI
    Auth --> Store
    Auth --> Native --> cpp --> ORT
    SVC["facegate service (SYSTEM)<br/>named pipe"] --> Auth
    CP["FaceGateCP.dll<br/>LogonUI credential provider"] -. pipe .-> SVC
```

**Boundary.** Everything that is *compute* (pixels in, numbers out) is in the C++ engine. Everything that is *policy* (when to unlock, how many frames, lockout, which app) is Python. The boundary is one flat C ABI (`engine/include/facegate_engine.h`) of six functions, so the engine has no dependency on Python, Qt or OpenCV and can be reused from any host.

The Python `FaceEngine` (ONNX Runtime for Python) is kept for three jobs: the reference the C++ engine is tested against, the engine inside the Linux container, and the fallback when the DLL or its models are missing.

## Decisions

### 1. C++ engine over ONNX Runtime + DirectML, not CUDA or TensorRT
DirectML runs on any Windows GPU with no extra install. CUDA needs cuDNN and cuBLAS (about 1 GB) and TensorRT more; neither is practical to ship inside an exe. Measured, the engine already finishes a frame in about a quarter of the camera's frame interval, so a faster backend would not be felt.

### 2. Static shapes and FP16, prepared offline (`tools/prepare_models.py`)
| Change | Evidence (RTX 4050, DirectML) |
|---|---|
| Recognizer FP16, fixed batch 4 | 10.7 → 4.0 ms; embeddings stay at cosine ≥ 0.99995 of FP32 |
| Detector fixed at 384×640 | 16:9 frames no longer waste 40% on padding: 6.3 → 3.2 ms (with FP16) |
| Dynamic batch removed | DirectML rejects the recognizer's dynamic batch outright |
| Liveness stays FP32, moved to the GPU | 37 ms on the CPU with default threading → 1.4 ms |
FP16 was judged by the end-to-end LFW evaluation, not only by cosine: accuracy is unchanged (99.28% vs 99.26% clean).

### 3. Keep ResNet-50; do not adopt ResNet-100
`glintr100` (Glint360K, 261 MB) was evaluated on the same 7,081 photos, FP16 on the GPU. At FAR 1e-5 it is slightly better on masked probes (95.75% vs 94.38%); at the 1e-6 operating point, which is what matters for unlocking, it is worse (clean 99.39% vs 99.59%, masked 81.0% vs 88.8%) and costs twice the compute. LFW is saturated, so this is not a verdict on harder datasets - only that this benchmark gives no reason to pay for it.

### 4. Explicit GPU choice
The laptop has an Intel iGPU and the RTX 4050. DirectML's default happened to pick the NVIDIA part, but a graphics-settings change or a different boot can reorder adapters, and the iGPU is 10× slower. The engine requests the high-performance adapter and reports which one it got; `tools/bench_engine.py` shows NVIDIA utilisation while it runs.

### 5. ONNX Runtime's idle threads must not spin
Left at defaults, idle worker threads busy-wait on a 16-core hybrid CPU and steal time from the caller (an unrelated OpenCV call ran 4× slower). The engine disables spinning and gives GPU sessions a single CPU thread. Measure engines in separate processes - one engine's leftover thread pools slow the next.

### 6. Thresholds are calibrated, not guessed
`tools/eval_lfw.py` embeds every photo three ways (clean, synthetic mask, synthetic cap) through the real pipeline and searches thresholds for a stated false-accept budget: at 1 in 1,000,000 the whole-face threshold is 0.356. A second, "mask" view (lower face painted out, on both enrolled and probe side) is a second opinion for near-misses. A matching "cap" view was built and removed because it only added false accepts.

### 7. Two build products, because of privilege
Code that runs elevated (the SYSTEM lock-screen service, the elevated autostart task, "Restart as Administrator") must not load from a folder ordinary programs can write to, or those programs could swap it and gain administrator rights.

| Product | Unpacks to | Elevated features |
|---|---|---|
| Folder build, installed to Program Files | in place (admin-only folder) | yes |
| Single-file `FaceGate-portable.exe` | per-user cache | refused |

The compiled app enforces this (`elevation_safe()`), and elevated processes ignore the `FACEGATE_HOME` / `FACEGATE_NATIVE_DIR` overrides, because they would inherit them from the user's environment.

### 8. File & folder lock is an NTFS deny rule, not encryption and not process interception
The lock must stop *every* program, including ones FaceGate never sees start (Explorer is already running; a script reads the file directly). Watching processes alone cannot do that, so the operating system enforces the lock:

```
icacls <item> /deny *S-1-5-4:(OI)(CI)(RD,X,WD,AD)      lock   (no (OI)(CI) for a file)
icacls <item> /remove:d *S-1-5-4                       unlock
```

| Alternative | Why not |
|---|---|
| Encrypt the files (AES or EFS) | Rewrites the user's data; a crash mid-way, or a lost key, loses it. Large folders take minutes. Kept as a future, opt-in vault. |
| Hide or rename | Anyone can unhide; programs holding the old path break. |
| Process interception only | Explorer and already-running apps are never seen starting. |
| Deny *Everyone* or the user's own SID | Would also block the SYSTEM service, backup and the indexer, and the user-SID rule does not cover other accounts. |

*INTERACTIVE* is in the token of every program a signed-in person starts, elevated or not, on any account, and in no service's token. The rule is added by the item's owner, so no administrator rights are needed, and it is inherited by everything inside a folder and removed with it. Measured on this laptop:
- **Timing:** 5,000 files lock in about 350 ms, and unlock in the same.
- **Clean removal:** unlock leaves no rule behind.
- **Rename and move:** renaming or moving an item keeps its lock.
- **Delete:** a non-empty locked folder cannot be deleted. A locked file can be, because delete permission comes from the parent folder.

The process watcher remains, but only as the *trigger*. While any item is locked it reads the command line of each new process. If one names a locked path, the watcher suspends it, challenges, unlocks the item and resumes it. A failed challenge resumes it too, because Windows already refuses access. `icacls.exe` is never held: FaceGate's own unlock would otherwise deadlock on itself. Items lock again after `relock_minutes`, on "Lock Files Now", and when FaceGate quits. On start-up every configured item is locked again, which also recovers from a crash.

`can_lock` refuses whatever would break Windows or FaceGate if it became unreadable:
- system, program and app-data folders, and anything that contains them;
- FaceGate's own folders;
- drive roots and network paths;
- the shell folders themselves (Desktop, Documents and so on);
- overlapping items.

### 9. Build: Nuitka with MSVC, clcache and a user plugin
Nuitka translates every Python module to C and compiles it. With `--msvc` it uses its bundled **clcache** automatically, so a rebuild that changes a few Python modules recompiles only their C files. Real ccache applies only to Nuitka's MinGW mode, and switching would change the C runtime that every native wheel (Qt, OpenCV, numpy) was built with, so MSVC stays. `tools/build_exe.ps1` writes the full log to `dist/nuitka-build.log` and prints the cache line.

The cache alone did not make rebuilds fast. With all 204 C files served from the cache, a rebuild still took 4.3 minutes, and timestamps on the log showed why: three of those minutes were the linker. Under link-time optimisation, MSVC defers code generation to the link, so it is redone on every build and never cached. The build therefore uses `--lto=no`:

| Build time | LTO | `--lto=no` |
|---|---|---|
| First build | 4.5 min | 3.9 min |
| Unchanged rebuild | 4.3 min | 1.5 min |

The link now takes about a second. Size is unchanged (318 MB), and the compiled Python's speed is irrelevant here because inference runs in the C++ engine. What remains is about a minute of Nuitka's own analysis of the Python code, which is not cached.

`packaging/nuitka/facegate_plugin.py` replaces a list of `--include-data-files` flags. It does three things:

| Job | How | Effect |
|---|---|---|
| Bundle the engine, runtime, models and lock-screen DLL | `considerDataFiles` on the `facegate` package; `onCompilationStartChecks` stops the build if any input is missing | One place lists what ships; a missing model fails in seconds, not after the compile |
| Keep out other operating systems' code | `onModuleEncounter` vetoes psutil's Linux, macOS and BSD backends | `psutil._pslinux` alone was 4 MB of generated C |
| Faster, smaller C | `decideCompilation` returns `bytecode` for `winerror` and `win32con` | 9 MB and 4.6 MB of generated C (the two largest files) for constant lookups |

DLLs cannot be vetoed from a plugin, so two trims stay as flags:
- **`--noinclude-dlls=cv2/opencv_videoio_ffmpeg*.dll`** (29 MB). It decodes video files only; webcams use DirectShow or Media Foundation, which are compiled into `cv2.pyd`. `selftest` reports the camera backends, so a regression is visible.
- **`--noinclude-qt-plugins=tls,imageformats,iconengines,styles`.** The app draws its own icons, uses the built-in Fusion style and loads no image files.

## Failure behaviour
- DLL, runtime or models missing → the Python engine is used (source installs), or the app reports it.
- GPU session cannot be created → that model falls back to the CPU.
- Hostile or malformed coordinates, undersized buffers and mismatched model outputs are rejected with an error code, never followed.
- Anything uncertain at the lock screen yields *no credential*; the password and PIN providers are untouched.

## Verification
| What | How |
|---|---|
| C++ engine equals the Python reference | `tests/test_native.py`: detection, alignment, both embedding views, liveness, template interchangeability, concurrency, hostile input |
| The two engines are interchangeable | a template enrolled by one engine is matched by the other (cosine > 0.98 for the same photo) |
| Accuracy | `tools/eval_lfw.py`: 17,118 same-person and 50,456,802 different-person comparisons |
| Speed | `tools/bench_engine.py`: each engine in its own process, per-stage split, GPU utilisation |
| Lock-screen component | `credential_provider/test/cp_test.cpp` plays LogonUI against a fake service |
| File & folder lock | `tests/test_locks.py`: real deny rules on temporary files; a real `cmd` reader is frozen, challenged, then allowed or refused; relock on time-out and on quit |
| Packaged build | `FaceGate.exe selftest`: GPU, liveness, every bundled dependency, camera backends |
