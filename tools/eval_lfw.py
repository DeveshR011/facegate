"""Calibrate thresholds and measure accuracy on LFW (Labeled Faces in the Wild).

Every image is run through the real pipeline (detect -> align -> embed) three
times: as-is, with a synthetic mask painted on, and with a synthetic cap.
All-vs-all comparison then gives ~50 million impostor pairs, enough to read
false-accept rates down to about 1e-6.

    python tools/eval_lfw.py                 # download LFW, embed, report
    python tools/eval_lfw.py --max-per-id 6  # cap images per identity (default)

Reading the numbers
* The gallery side is always ONE clean image. Real enrollment stores about
  eleven samples across head poses, so true-accept rates here are a lower bound.
* LFW contains mislabeled identities (the same person filed under two names).
  They show up as "impostor" pairs with similarity above 0.5, which a correct
  impostor pair essentially never reaches. Rates are reported both raw and
  with those pairs excluded.
* Masks and caps are synthetic (painted polygons). Real ones vary more.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facegate.config import Settings  # noqa: E402
from facegate.core.engine import FaceEngine  # noqa: E402
from facegate.core.occlusion import VIEWS, synth_cap, synth_mask  # noqa: E402

LFW_URL = "https://ndownloader.figshare.com/files/5976015"   # lfw-funneled.tgz
CACHE = ROOT / ".cache"
CONDITIONS = ("clean", "masked", "capped")
LABEL_NOISE = 0.50      # clean full-face similarity above which an "impostor" pair is a mislabel


def fetch_lfw() -> Path:
    out = CACHE / "lfw_funneled"
    if out.exists():
        return out
    CACHE.mkdir(exist_ok=True)
    tgz = CACHE / "lfw-funneled.tgz"
    if not tgz.exists():
        print("downloading LFW (~230 MB)…", flush=True)
        req = urllib.request.Request(LFW_URL, headers={"User-Agent": "FaceGate-eval"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tgz, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    print("extracting…", flush=True)
    with tarfile.open(tgz) as tf:
        tf.extractall(CACHE, filter="data")
    tgz.unlink()
    return out


def select_images(root: Path, max_per_id: int, singletons: int) -> tuple[list[Path], np.ndarray]:
    paths, labels = [], []
    people = sorted(p for p in root.iterdir() if p.is_dir())
    single_used = 0
    for label, person in enumerate(people):
        imgs = sorted(person.glob("*.jpg"))
        if len(imgs) == 1:
            if single_used >= singletons:
                continue
            single_used += 1
        for img in imgs[:max_per_id]:
            paths.append(img)
            labels.append(label)
    return paths, np.array(labels)


def embed_all(paths: list[Path], engine: FaceEngine) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(7)
    feats = {c: {v: [] for v in VIEWS} for c in CONDITIONS}
    keep = []
    t0 = time.time()
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        faces = engine.detect(img)
        if not faces:
            continue
        # LFW centres the labelled person; ignore bystanders.
        centre = np.array(img.shape[1::-1]) / 2
        face = min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))
        crop = engine.align(img, face)
        variants = {"clean": crop, "masked": synth_mask(crop, rng), "capped": synth_cap(crop, rng)}
        for cond in CONDITIONS:
            emb = engine.embed_views(variants[cond], VIEWS, flip=True)
            for v in VIEWS:
                feats[cond][v].append(emb[v])
        keep.append(i)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(paths)}  ({rate:.0f} img/s)", flush=True)
    out = {c: {v: np.stack(a) for v, a in d.items()} for c, d in feats.items()}
    return out, np.array(keep)


def far_threshold(impostor: np.ndarray, far: float) -> float:
    """Smallest threshold whose false-accept rate is <= far."""
    k = int(np.floor(far * impostor.size))
    if k <= 0:
        return float(impostor.max()) + 1e-4
    return float(np.partition(impostor, impostor.size - k - 1)[impostor.size - k - 1]) + 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-id", type=int, default=6)
    ap.add_argument("--singletons", type=int, default=1500)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    root = fetch_lfw()
    paths, labels = select_images(root, args.max_per_id, args.singletons)
    cache = CACHE / f"lfw_emb_v2_{args.max_per_id}_{args.singletons}.npz"

    if cache.exists() and not args.no_cache:
        z = np.load(cache)
        feats = {c: {v: z[f"{c}_{v}"] for v in VIEWS} for c in CONDITIONS}
        labels = z["labels"]
    else:
        engine = FaceEngine()
        print(f"embedding {len(paths)} images on {engine.provider}…", flush=True)
        feats, keep = embed_all(paths, engine)
        labels = labels[keep]
        np.savez(cache, labels=labels,
                 **{f"{c}_{v}": a for c, d in feats.items() for v, a in d.items()})

    n = len(labels)
    same = labels[:, None] == labels[None, :]
    genuine = same & ~np.eye(n, dtype=bool)
    impostor = ~same

    gallery = feats["clean"]
    S = {(c, v): gallery[v] @ feats[c][v].T for c in CONDITIONS for v in VIEWS}

    noise = impostor & (S[("clean", "full")] >= LABEL_NOISE)
    clean_imp = impostor & ~noise
    print(f"\n{n} images, {len(np.unique(labels))} identities")
    print(f"{genuine.sum():,} genuine comparisons, {impostor.sum():,} impostor comparisons")
    print(f"{noise.sum()} impostor pairs score >= {LABEL_NOISE} on clean full faces "
          f"(LFW label noise; excluded from the 'adjusted' columns)\n")

    print("Whole-face view alone")
    print(f"{'thr':>5s} | {'clean TAR':>9s} {'masked TAR':>10s} {'capped TAR':>10s} | "
          f"{'FAR raw':>8s} {'FAR adjusted':>12s}")
    for t in (0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45):
        tar = [100 * (S[(c, "full")][genuine] >= t).mean() for c in CONDITIONS]
        raw = (S[("clean", "full")][impostor] >= t).mean()
        adj = (S[("clean", "full")][clean_imp] >= t).mean()
        print(f"{t:5.2f} | {tar[0]:8.2f}% {tar[1]:9.2f}% {tar[2]:9.2f}% | {raw:8.1e} {adj:12.1e}")

    for far in (1e-4, 1e-5):
        t = far_threshold(S[("clean", "full")][clean_imp], far)
        print(f"  adjusted FAR {far:.0e} is reached at threshold {t:.3f}")

    def report(s: Settings, title: str) -> None:
        print(f"\n{title}")
        print(f"  match_threshold={s.match_threshold}  occlusion_floor={s.occlusion_floor}  "
              f"mask_threshold={s.mask_threshold}  occlusion_unlock={s.occlusion_unlock}")
        print(f"  {'probe':7s} {'TAR':>8s} {'FAR raw':>9s} {'FAR adjusted':>13s} {'false accepts (adj.)':>21s}")
        for cond in CONDITIONS:
            full = S[(cond, "full")]
            accept = full >= s.match_threshold
            if s.occlusion_unlock:
                accept |= (full >= s.occlusion_floor) & (S[(cond, "mask")] >= s.mask_threshold)
            fa = int(accept[clean_imp].sum())
            print(f"  {cond:7s} {100 * accept[genuine].mean():7.2f}% {accept[impostor].mean():9.1e} "
                  f"{fa / clean_imp.sum():13.1e} {fa:>21d}")

    defaults = Settings()
    report(defaults, "Shipped decision rule")
    strict = Settings()
    strict.occlusion_unlock = False
    report(strict, "With 'Unlock with a mask' switched off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
