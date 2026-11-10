# FaceGate

Face unlock for Windows: lock individual apps behind your face, and (experimentally) unlock the PC itself from the lock screen. Recognition runs on the GPU in a C++ engine with deep networks, not landmark geometry. Ships as a single Windows `.exe`.

- **App lock** – pick any `.exe`; it is frozen the instant it starts and released only after you are recognised.
- **Windows unlock** – a credential provider adds FaceGate as a sign-in option on the lock screen.
- **Works with a cap, glasses or a mask** – measured below, not just claimed.
- **Anti-spoofing** – two networks reject photos and screens.
- **Animated overlay in your colours** – seven palettes (some flowing gradients) or any colour you pick; brackets shimmer while scanning, close into a ring with a tick on success, shake on failure.
- **Fast** – about 8 ms of compute per camera frame, on the discrete GPU.
- **Private** – only 512-number embeddings are stored, sealed with Windows DPAPI. No images, no network calls after setup.

## Read this first: what a webcam can and cannot do

Apple's Face ID projects 30,000 infrared dots and reads a depth map. This laptop (Acer Predator Helios Neo 16) has one ordinary RGB webcam and no infrared or depth sensor, so **no software can give it Face ID's resistance to spoofing**.

| | Face ID | FaceGate |
|---|---|---|
| Telling two people apart | about 1 in 1,000,000 (Apple's figure) | about 1 in 1,150,000 on the LFW benchmark |
| Rejecting a photo or video | depth sensing | a learned classifier on a 2-D image; good against casual attempts, not against a determined attacker |
| Works in the dark | yes (infrared) | no; needs some light on your face |

Treat it as a convenience lock that keeps family, colleagues and passers-by out. Do not rely on it against someone prepared to build a spoof of your face. Your Windows password or PIN always remains available.

## Measured accuracy

`tools/eval_lfw.py` runs the real pipeline (the C++ engine, FP16, on the GPU) over 7,105 photos of 3,180 people from [LFW](http://vis-www.cs.umass.edu/lfw/) and compares every photo with every other: 17,118 same-person and 50,456,802 different-person comparisons. Each photo is also tested with a synthetic mask and a synthetic cap painted on. The thresholds are not hand-picked: the tool searches for the best setting at a stated false-accept budget (here 1 in 1,000,000).

| Setting | Clear face accepted | Masked face accepted | Capped face accepted | Stranger accepted |
|---|---|---|---|---|
| Default (mask unlock on) | 99.58% | 90.10% | 99.22% | 1 in ~1,150,000 (clear), 1 in ~2,300,000 (masked) |
| Mask unlock off | 99.58% | 88.34% | 99.22% | 1 in ~1,150,000 |

How to read these numbers:

- The "accepted" figures compare against **one** enrolled photo taken years apart in different places. FaceGate enrolls about eleven samples across head poses on the camera you unlock with, so real acceptance should be higher. This has not been measured.
- LFW contains mislabeled people. 68 "different-person" pairs score above 0.50, which genuine strangers essentially never do; they are excluded from the stranger column. Including them, the stranger rate is 1 in ~450,000.
- A gallery of several samples gives a stranger several chances to match, so the per-attempt false-accept rate is higher than the per-comparison figure (roughly up to the gallery size times).
- Masks and caps are painted polygons. Real ones vary more.
- **The anti-spoofing networks have not been benchmarked here** (no spoof dataset was used). On this laptop a live face scored 0.87–0.95 against a 0.55 threshold.
- FP16 and the C++ pipeline cost no accuracy: the Python FP32 reference measured 99.26% / 92.71% / 98.59% at the previous thresholds, the C++ pipeline 99.28% / 92.77% / 98.50%.

## Speed

Per camera frame, 1280×720, on this laptop (RTX 4050, DirectML) - `tools/bench_engine.py`, each engine in its own process:

| Stage | Python engine | C++ engine |
|---|---|---|
| Detect (SCRFD) | 15.8 ms | 4.0 ms |
| Embed (full + mask view, with mirror images) | 9.7 ms | 3.6 ms |
| Liveness (two networks) | 33.4 ms | 1.4 ms |
| **One authentication frame** (with liveness) | **59.8 ms** | **9.8 ms** |

The camera delivers a frame every 33 ms, so the engine now keeps up with it several times over. What remains is the camera itself: about 0.75 s for the first frame, which no software can shorten, so expect roughly one second from trigger to unlock.

What the research found (details and evidence in [docs/architecture.md](docs/architecture.md)):

- DirectML does land on the RTX 4050 here, not the Intel iGPU (which is 10× slower), but the engine now requests it explicitly and reports which GPU it got.
- Liveness was the biggest cost: running on the CPU with default threading it took 37 ms. On the GPU it takes about 1.4 ms.
- FP16 recognizer: 10.7 → 4.0 ms with embeddings at cosine ≥ 0.99995 of FP32. A 384×640 detector input (the shape of a 16:9 frame) halves detection time.
- ONNX Runtime's idle threads busy-wait by default and slow everything else down on a 16-core hybrid CPU; the engine turns that off.
- A bigger recognizer (ResNet-100) was tested and rejected: worse at the 1-in-a-million operating point and twice the compute.

## Get it

### Build the exe

Requires Windows 10/11, Python 3.12, Visual Studio 2022 Build Tools (C++) and CMake.

```powershell
git clone https://github.com/DeveshR011/facegate.git
cd facegate
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-build.txt
```

```powershell
engine\build.ps1                          # C++ engine + signed ONNX Runtime / DirectML runtime
.venv\Scripts\python tools\prepare_models.py     # downloads the models, writes static-shape FP16 versions
credential_provider\build.ps1             # lock-screen component
tools\build_exe.ps1                       # Nuitka: dist\FaceGate\FaceGate.exe
```

The first Nuitka build takes a while; later ones reuse its cache. `tools\build_exe.ps1 -OneFile` produces a single `dist\FaceGate-portable.exe` instead.

| | Folder build `dist\FaceGate\` | Single file `FaceGate-portable.exe` |
|---|---|---|
| Start-up | fastest | unpacks once into a per-user cache, then fast |
| App lock, enrollment, themes | yes | yes |
| Elevated features (lock-screen unlock, start with Windows, lock admin apps) | yes, after `Install-FaceGate.ps1` | **no** (see Security notes) |

`Install-FaceGate.ps1` (in the folder) copies the build to Program Files and adds a Start Menu shortcut.

### Or run from source

```powershell
.venv\Scripts\pythonw -m facegate
```

The C++ engine is used automatically when built; otherwise the Python engine is (it downloads about 290 MB of models on first run, verified against pinned SHA-256 values).

`FaceGate.exe selftest --image face.jpg` loads the engine, prints which GPU it is on, and checks that every bundled dependency is present - no camera needed.

## Use

1. **Set Up Face ID** – look at the camera and follow the prompts to turn your head.
2. **Add Alternate Appearance** – repeat wearing a cap, glasses or a mask if you use them often. This helps more than any setting.
3. **Add App…** – choose an `.exe` (or pick a running app) and leave its switch on.
4. Launch that app: the overlay appears, and the app continues only after a match.
5. **Appearance** – pick the animation colours; **Preview** plays scanning, success and failure.

Closing the window keeps FaceGate guarding from the system tray.

| Setting | Effect |
|---|---|
| Unlock with a mask | Accepts a near miss when the eyes-and-brows view independently agrees. |
| Require a live face | Anti-spoofing check. Turning it off needs your face or password. |
| Adaptive learning | Adds confident live matches to the gallery so it follows a beard or new glasses. |
| Start with Windows | Scheduled task that starts FaceGate elevated at sign-in without a UAC prompt. |
| Appearance | Seven palettes (Classic, Ocean, Aurora, Sunset, Neon, Rainbow, Mono) or a custom colour. |

After five failed scans Face ID pauses for five minutes and only the Windows password is accepted. Removing an app, switching off its lock, disabling liveness and quitting all require your face or password first.

### Administrator rights

Windows does not let a normal process freeze one that runs as administrator. Click **Restart as Administrator** (or enable **Start with Windows**) so FaceGate can lock elevated apps too.

## Windows unlock (experimental)

> **Not yet tested on a real lock screen.** The DLL compiles warning-free and passes a harness that plays LogonUI's role against a fake service (seven scenarios, including the packed credential and cancellation). It has never been loaded by the real LogonUI, and the SYSTEM service has not been run in session 0. Try it only when you can afford to troubleshoot.

How it works:

```
lock screen (LogonUI)                       session 0
┌──────────────────────┐   named pipe    ┌───────────────────────────┐
│ FaceGateCP.dll       │ ──── AUTH ────▶ │ facegate service (SYSTEM) │
│ credential provider  │ ◀── OK + cred ─ │ camera → match → liveness │
└──────────────────────┘                 └───────────────────────────┘
```

Windows will not unlock without a real credential, so you type your account password once. It is stored as an LSA private secret (readable by SYSTEM and administrators only), never in a file, and handed to the lock screen only after a match.

Set-up: install the folder build (`Install-FaceGate.ps1`), start it from the Start Menu, then **Restart as Administrator** → switch on **Unlock Windows with your face** → enter your password. Lock with `Win+L` to try it.

Safeguards:

- After a restart you must sign in with your password once before face unlock is offered.
- One automatic scan per lock-screen appearance; more only when you click **Try again**.
- If Windows rejects the stored password (you changed it), face unlock pauses itself rather than retrying.
- The pipe admits SYSTEM only, and the provider connects at identification level so a fake pipe cannot impersonate LogonUI.

If something goes wrong: choose **Sign-in options** and use your password or PIN. Windows does not load third-party credential providers in Safe Mode, so you can always boot there and switch the feature off, or delete this registry key:

```
HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Authentication\Credential Providers\{7B3C9F2E-5A41-4D8B-9E6C-2F1A8D4B6C73}
```

## How recognition works

| Stage | Model | Purpose |
|---|---|---|
| Detect | SCRFD-10G (FP16, 384×640) | Face box and five landmarks |
| Align | similarity transform | Canonical 112×112 crop |
| Embed | ArcFace ResNet-50, WebFace600K (FP16, batch 4) | 512-d identity vector, averaged with its mirror image |
| Liveness | MiniFASNetV2 + V1SE | Probability the face is live rather than a print or screen |

All four run in the C++ engine (`engine/`, a C ABI over ONNX Runtime with the DirectML provider). A scan succeeds when three frames match and their average liveness clears the threshold.

**Masks.** A recogniser embeds whatever is in the crop, mask included. FaceGate also embeds a second view of every sample with the lower face painted out, on both the enrolled and the live side, so that view compares only eyes, brows and forehead. It never decides alone: a whole-face score between 0.32 and 0.36 is accepted only if this view scores at least 0.50 too. A matching "cap" view was built and removed: the plain embedding already handles caps (99.2% against 99.6% uncovered) and the extra view only added false accepts.

## Docker

The camera overlay, app lock and Windows unlock are native Windows code and cannot run in a Linux container. The image packages the Python reference pipeline as an HTTP API.

> **The images have not been built or run on this machine.** CI builds the CPU image on every push; `Dockerfile.gpu` is untested.

```bash
docker compose up --build
```

```bash
docker compose --profile gpu up --build
```

```bash
curl -F "images=@me1.jpg" -F "images=@me2.jpg" http://127.0.0.1:8000/enroll
curl -F "image=@probe.jpg" http://127.0.0.1:8000/verify
```

| Endpoint | |
|---|---|
| `GET /health` | Engine status |
| `POST /enroll` | One or more `images`, optional `label` |
| `POST /verify` | One `image` → match, score, liveness |
| `DELETE /enrollment` | Forget everything |

Ports bind to localhost. Set `FACEGATE_API_KEY` to require an `X-API-Key` header.

## Security notes

- **Elevated code must live in an administrator-only folder.** The compiled app refuses to register elevated tasks, or restart itself elevated, unless it both runs from and unpacks to Program Files. This is why the single-file build cannot do elevated features (it unpacks into a per-user cache any of your programs can write to), and why elevated processes ignore the `FACEGATE_HOME` and `FACEGATE_NATIVE_DIR` overrides.
- **Running from source is different.** The lock-screen service and elevated autostart then run your source tree and its `.venv`; anyone who can edit those files can run code as SYSTEM. On a single-user PC where you are the administrator that grants nothing new; on a shared PC use the installed folder build.
- **App lock is a deterrent, not a sandbox.** It matches executables by path, so a copy of the `.exe` elsewhere is not locked. An administrator can end FaceGate from Task Manager.
- **Templates use machine-scope DPAPI** so the SYSTEM service can read what you enrolled. Any local account can decrypt them; the file is useless on another PC. Enabling Windows unlock restricts the data folder to SYSTEM, administrators and you.
- Embeddings are not images, but research has shown approximate faces can be reconstructed from them. Treat `templates.bin` as sensitive.
- ONNX Runtime and DirectML are downloaded from NuGet and the build refuses to continue unless both DLLs carry a valid Microsoft signature.

## Development

```powershell
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests
.venv\Scripts\python tools\eval_lfw.py          # accuracy + threshold calibration
.venv\Scripts\python tools\bench_engine.py      # speed, Python vs C++
credential_provider\build\Release\cp_test.exe credential_provider\build\Release\FaceGateCP_test.dll
```

95 tests. The state machines run against a fake engine; the app-guard tests freeze, release and kill real `ping.exe` processes; the native tests compare the C++ engine with the Python reference on real photographs (detection, alignment, both embedding views, liveness, template interchangeability, concurrent callers, hostile input) and are skipped when the engine, models or LFW are absent.

```
facegate/
  core/       engine interface, C++ binding, Python reference engine, matching, template store
  guard/      process watcher and trust policy for app lock
  ui/         glyph animation, palettes, overlay, enrollment, settings window
  service/    lock-screen pipe service, HTTP API
  win/        DPAPI, elevation, LSA secret, scheduled tasks, installer
engine/                C++17 inference engine (C ABI), build + SDK fetch scripts
credential_provider/   C++ lock-screen provider and its test harness
packaging/             Install-FaceGate.ps1
tools/                 model preparation, LFW calibration, benchmark, Nuitka build
docs/                  architecture and decision record
```

## Uninstall

Switch off **Unlock Windows with your face** and **Start with Windows** in the app and quit from the tray. For an installed build run `Install-FaceGate.ps1 -Uninstall` (add `-RemoveData` to delete `C:\ProgramData\FaceGate` too); otherwise delete that folder and this one.

## Models and licences

Code: MIT. The model weights keep their own terms:

- SCRFD and ArcFace from [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`) – **non-commercial research use only**.
- MiniFASNet from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), ONNX export by [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) – Apache-2.0.

**A built `FaceGate.exe` contains these weights.** Keep it for personal and research use; do not redistribute it commercially. The repository itself contains no weights.

"Face ID" is a trademark of Apple Inc. FaceGate is not affiliated with Apple.
