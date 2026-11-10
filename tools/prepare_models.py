"""Turn the downloaded research models into the static-shape, FP16 files the C++ engine loads.

    python tools/prepare_models.py                  # -> engine/models/
    python tools/prepare_models.py --out DIR

Why each change (measured on an RTX 4050 laptop GPU, DirectML):

  static shapes   DirectML compiles one GPU graph per shape and rejects the dynamic batch
                  of the ArcFace model outright. Fixing the shapes also lets it plan memory.
  detector        input 384x640: a 16:9 camera frame letterboxes to 640x360, so the usual
                  640x640 square wastes 40% of the compute on padding. 6.3 -> 3.2 ms.
  recognizer      FP16 with batch 4 (full + mask view, each with its mirror image):
                  10.7 -> 4.0 ms; embeddings stay at cosine >= 0.99995 of FP32.
  liveness        kept FP32 (tiny, and its accuracy is unmeasured here, so not perturbed);
                  batch fixed to 1.

Inputs and outputs stay FP32 (keep_io_types), so the engine needs no half-float handling.
Requires the dev extras:  pip install onnx onnxconverter-common
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
from onnxconverter_common import float16  # noqa: E402

from facegate.core import models as registry  # noqa: E402

warnings.filterwarnings("ignore", message=r"the float32 number .* will be truncated")

DET_SIZE = (384, 640)      # (height, width)
REC_BATCH = 4


def _fix_input(model: onnx.ModelProto, dims: dict[int, int], pin_symbolic_outputs: bool = False) -> None:
    shape = model.graph.input[0].type.tensor_type.shape
    for axis, value in dims.items():
        shape.dim[axis].dim_value = value          # replaces any symbolic dim_param
    for out in model.graph.output:
        if pin_symbolic_outputs:                   # older models: checker insists on an output shape
            for d in out.type.tensor_type.shape.dim:
                if d.HasField("dim_param"):
                    d.dim_value = dims[0]
        else:                                      # otherwise let the runtime re-infer it
            out.type.tensor_type.ClearField("shape")
    del model.graph.value_info[:]


def _save(model: onnx.ModelProto, path: Path) -> dict:
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": registry.sha256_of(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "engine" / "models")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = registry.ensure_models()
    manifest = {"detector_size_hw": DET_SIZE, "recognizer_batch": REC_BATCH, "files": {}}

    m = onnx.load(src["detector"])
    _fix_input(m, {0: 1, 2: DET_SIZE[0], 3: DET_SIZE[1]})
    m = float16.convert_float_to_float16(m, keep_io_types=True)
    manifest["files"]["detector"] = _save(m, args.out / "det_10g_384x640_fp16.onnx")

    m = onnx.load(src["recognizer"])
    _fix_input(m, {0: REC_BATCH})
    m = float16.convert_float_to_float16(m, keep_io_types=True)
    manifest["files"]["recognizer"] = _save(m, args.out / "w600k_r50_b4_fp16.onnx")

    for key, name in (("liveness_v2", "MiniFASNetV2"), ("liveness_v1se", "MiniFASNetV1SE")):
        m = onnx.load(src[key])
        _fix_input(m, {0: 1}, pin_symbolic_outputs=True)
        manifest["files"][key] = _save(m, args.out / f"{name}_b1.onnx")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for key, info in manifest["files"].items():
        print(f"{key:14s} {info['file']:28s} {info['bytes'] / 1e6:7.2f} MB  {info['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
