# FaceGate

Face unlock for Windows: lock individual apps behind your face, and (experimentally) unlock the PC itself from the lock screen. Recognition runs locally on the GPU with deep networks, not landmark geometry.

- **App lock** – pick any `.exe`; it is frozen the instant it starts and released only after you are recognised.
- **Windows unlock** – a credential provider adds FaceGate as a sign-in option on the lock screen.
- **Works with a cap, glasses or a mask** – measured below, not just claimed.
- **Anti-spoofing** – two networks reject photos and screens.
- **Animated overlay** – brackets shimmer while scanning, close into a ring with a tick on success, shake on failure.
- **Private** – only 512-number embeddings are stored, sealed with Windows DPAPI. No images, no network calls after the model download.

## Read this first: what a webcam can and cannot do

Apple's Face ID projects 30,000 infrared dots and reads a depth map. This laptop (Acer Predator Helios Neo 16) has one ordinary RGB webcam and no infrared or depth sensor, so **no software can give it Face ID's resistance to spoofing**.

| | Face ID | FaceGate |
|---|---|---|
| Telling two people apart | about 1 in 1,000,000 (Apple's figure) | about 1 in 500,000 on the LFW benchmark, 1 in 5,000,000 with mask unlock off |
| Rejecting a photo or video | depth sensing | a learned classifier on a 2-D image; good against casual attempts, not against a determined attacker |
| Works in the dark | yes (infrared) | no; needs some light on your face |

Treat it as a convenience lock that keeps family, colleagues and passers-by out. Do not rely on it against someone prepared to build a spoof of your face. Your Windows password or PIN always remains available.

## Measured accuracy

`tools/eval_lfw.py` runs the real pipeline over 7,081 photos of 3,177 people from [LFW](http://vis-www.cs.umass.edu/lfw/) and compares every photo with every other: 17,004 same-person and 50,116,476 different-person comparisons. Each photo is also tested with a synthetic mask and a synthetic cap painted on.

| Setting | Clear face accepted | Masked face accepted | Capped face accepted | Stranger accepted |
|---|---|---|---|---|
| Default (mask unlock on) | 99.26% | 92.71% | 98.59% | 1 in ~480,000 |
| Mask unlock off | 99.22% | 80.20% | 98.49% | 1 in ~5,000,000 |

How to read these numbers:

- The "accepted" figures compare against **one** enrolled photo taken years apart in different places. FaceGate enrolls about eleven samples across head poses on the same camera you unlock with, so real acceptance should be higher. This has not been measured.
- LFW contains mislabeled people. 68 "different-person" pairs score above 0.50, which genuine strangers essentially never do; they are excluded from the stranger column. Including them, the default rate is 1 in ~295,000.
- A stored gallery of several samples gives a stranger several chances to match, which raises the false-accept rate somewhat above the single-comparison figure.
- Masks and caps are painted polygons. Real ones vary more.
- **The anti-spoofing networks have not been benchmarked here** (no spoof dataset was used). On this laptop a live face scored 0.87–0.95 against a 0.55 threshold.

Speed on the RTX 4050 (DirectML): about 20–27 ms per frame for detection, embedding and liveness. The camera takes about 0.75 s to deliver its first frame, so expect roughly one second from trigger to unlock.

## Install

Requires Windows 10/11 and Python 3.12 (the version this was built and tested on).

```powershell
git clone https://github.com/DeveshR011/facegate.git
cd facegate
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pythonw -m facegate
```

The first launch downloads about 290 MB, keeps the 195 MB of model weights it needs in `C:\ProgramData\FaceGate\models`, and verifies each file against a pinned SHA-256.

## Use

1. **Set Up Face ID** – look at the camera and follow the prompts to turn your head.
2. **Add Alternate Appearance** – repeat wearing a cap, glasses or a mask if you use them often. This helps more than any setting.
3. **Add App…** – choose an `.exe` (or pick a running app) and leave its switch on.
4. Launch that app: the overlay appears, and the app continues only after a match.

Closing the window keeps FaceGate guarding from the system tray.

| Setting | Effect |
|---|---|
| Unlock with a mask | Accepts a near miss when the eyes-and-brows view independently agrees. About ten times more permissive. |
| Require a live face | Anti-spoofing check. Turning it off needs your face or password. |
| Adaptive learning | Adds confident live matches to the gallery so it follows a beard or new glasses. |
| Start with Windows | Scheduled task that starts FaceGate elevated at sign-in without a UAC prompt. |

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

Set-up:

```powershell
powershell -ExecutionPolicy Bypass -File credential_provider\build.ps1
```

Then in FaceGate: **Restart as Administrator** → switch on **Unlock Windows with your face** → enter your password. Lock with `Win+L` to try it.

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
| Detect | SCRFD-10G | Face box and five landmarks |
| Align | similarity transform | Canonical 112×112 crop |
| Embed | ArcFace ResNet-50 (WebFace600K) | 512-d identity vector, averaged with its mirror image |
| Liveness | MiniFASNetV2 + V1SE | Probability the face is live rather than a print or screen |

A scan succeeds when three frames match and their average liveness clears the threshold.

**Masks.** A recogniser embeds whatever is in the crop, mask included. FaceGate also embeds a second view of every sample with the lower face painted out, on both the enrolled and the live side, so that view compares only eyes, brows and forehead. It never decides alone: a whole-face score between 0.30 and 0.40 is accepted only if this view scores at least 0.40 too.

Two things the benchmark changed:

- A matching "cap" view was built and removed. The plain embedding already handles caps (98.5% against 99.2% uncovered) and the extra view only added false accepts.
- The mask view helps less than expected. Simply lowering the whole-face threshold reaches similar masked acceptance; the second view's benefit is doing so with fewer false accepts on uncovered faces.

## Docker

The camera overlay, app lock and Windows unlock are native Windows code and cannot run in a Linux container. The image packages the same recognition pipeline as an HTTP API.

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

- **App lock is a deterrent, not a sandbox.** It matches executables by path, so a copy of the `.exe` elsewhere is not locked. An administrator can end FaceGate from Task Manager.
- **The lock-screen service runs your source tree as SYSTEM.** Anyone who can edit the files in this folder or its `.venv` can run code as SYSTEM. On a single-user PC where you are the administrator that grants nothing new; on a shared PC, move the project somewhere only administrators can write.
- **Templates use machine-scope DPAPI** so the SYSTEM service can read what you enrolled. Any local account can decrypt them; the file is useless on another PC. Enabling Windows unlock restricts the data folder to SYSTEM, administrators and you.
- Embeddings are not images, but research has shown approximate faces can be reconstructed from them. Treat `templates.bin` as sensitive.

## Development

```powershell
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests
.venv\Scripts\python tools\eval_lfw.py
credential_provider\build\Release\cp_test.exe credential_provider\build\Release\FaceGateCP_test.dll
```

73 tests. The state machines run against a fake engine; the app-guard tests freeze, release and kill real `ping.exe` processes; the integration tests use the real models and are skipped when the weights or LFW are absent.

```
facegate/
  core/       detector, alignment, embedding, liveness, matching, template store
  guard/      process watcher and trust policy for app lock
  ui/         glyph animation, overlay, enrollment, settings window
  service/    lock-screen pipe service, HTTP API
  win/        DPAPI, elevation, LSA secret, scheduled tasks, installer
credential_provider/   C++ lock-screen provider and its test harness
tools/                 model fetcher, LFW calibration
```

## Uninstall

Switch off **Unlock Windows with your face** and **Start with Windows** in the app, quit from the tray, then delete `C:\ProgramData\FaceGate` and this folder.

## Models and licences

Code: MIT. The weights are downloaded, not redistributed, and keep their own terms:

- SCRFD and ArcFace from [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`) – **non-commercial research use only**.
- MiniFASNet from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), ONNX export by [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) – Apache-2.0.

"Face ID" is a trademark of Apple Inc. FaceGate is not affiliated with Apple.
