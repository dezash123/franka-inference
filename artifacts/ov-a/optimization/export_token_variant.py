"""Export the Pi0.5 DROID OpenVINO W8A8 IR for a different language-token count.

Mirrors run_framework_baselines.openvino_baseline (BF16 trace, static shapes,
first KV update as clone) and run_w8a8_baselines.compress_openvino
(NNCF INT8_SYM weight compression), but on CPU and for the inputs under
inputs/tokens-<T>/. Output: inputs/tokens-<T>/openvino-dynamic-w8a8/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PI05_BASELINE_ROOT", "/workflow")
import run_framework_baselines as base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--inputs", default="/workflow/inputs")
ap.add_argument("--threads", type=int, default=os.cpu_count())
args = ap.parse_args()
T = args.tokens
folder = Path(args.inputs) / f"tokens-{T}"
out = folder / "openvino-dynamic-w8a8"
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)

import openvino as ov  # noqa: E402
import nncf  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers.cache_utils import DynamicLayer  # noqa: E402

original_update = DynamicLayer.update


def ov_cache_update(self, key_states, value_states, *a, **k):
    if not self.is_initialized:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys, self.values = key_states.clone(), value_states.clone()
        self.is_initialized = True
        return self.keys, self.values
    return original_update(self, key_states, value_states, *a, **k)


DynamicLayer.update = ov_cache_update

values = load_file(str(folder / "zero.safetensors"), device="cpu")
keys = [f"input.{n}" for n in range(1236, 1245)]
inputs = tuple(values[k].contiguous() for k in keys)
assert inputs[6].shape == (1, T) and inputs[7].shape == (1, T), [x.shape for x in inputs]

wrapper = base.load_model("cpu")          # BF16 parameters, as the baseline export
started = time.perf_counter()
print("TRACE_START", flush=True)
with torch.inference_mode():
    traced = torch.jit.trace(wrapper, inputs, check_trace=False, strict=False)
    ir = ov.convert_model(traced, input=[list(x.shape) for x in inputs])
trace_seconds = time.perf_counter() - started
print("CONVERTED", trace_seconds, flush=True)
fp_xml = out / f"pi05_droid_tokens{T}_fp.xml"
ov.serialize(ir, str(fp_xml))
del traced, wrapper
gc.collect()

started = time.perf_counter()
model = ov.Core().read_model(str(fp_xml))
compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)
xml = out / "pi05_droid_dynamic_w8a8.xml"
ov.serialize(compressed, str(xml))
compress_seconds = time.perf_counter() - started
print("COMPRESSED", compress_seconds, flush=True)
fp_xml.unlink(); fp_xml.with_suffix(".bin").unlink()

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

(out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in (xml, xml.with_suffix(".bin"))))
(out / "export.json").write_text(json.dumps({
    "tokens": T, "trace_dtype": "bfloat16", "trace_device": "cpu",
    "trace_and_convert_seconds": trace_seconds, "compress_seconds": compress_seconds,
    "openvino": ov.__version__, "nncf": nncf.__version__, "torch": torch.__version__,
    "weight_compression": "INT8_SYM", "input_shapes": [list(x.shape) for x in inputs],
}, indent=2) + "\n")
print("DONE", xml, flush=True)
