"""Export the Pi0.5 DROID OpenVINO W8A8 IR for a camera-count / token-count
variant. Same recipe as export_token_variant.py (BF16 CPU trace, static
shapes, first KV update as clone, NNCF INT8_SYM), with C images and T tokens.
Writes inputs/cams<C>-tokens<T>/{zero,random,validation-*}.safetensors keyed
by the exported input names, and the IR under openvino-dynamic-w8a8/.
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
ap.add_argument("--cams", type=int, required=True)
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--inputs", default="/workflow/inputs")
ap.add_argument("--threads", type=int, default=os.cpu_count())
args = ap.parse_args()
C, T = args.cams, args.tokens
src = Path(args.inputs)
folder = src / f"cams{C}-tokens{T}"
out = folder / "openvino-dynamic-w8a8"
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)

import openvino as ov  # noqa: E402
import nncf  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402
from transformers.cache_utils import DynamicLayer  # noqa: E402

# EXPORT FIX (W8A8Line, 2026-09-22; carried into the mux exports by OvMux
# 2026-09-23): `sample_actions` computes the prefix position ids as
# `torch.cumsum(prefix_pad_masks, dim=1) - 1` on a BOOLEAN mask.
# torch.jit.trace records the op with the boolean element type and the
# OpenVINO conversion keeps it, so the running sum is accumulated in 8 bits
# and WRAPS EVERY 256 TOKENS.  Casting the mask to int32 before the cumsum is
# numerically a no-op in torch (cumsum of bool already returns int64) and
# makes the conversion emit an i32 CumSum.
_original_cumsum = torch.cumsum
_original_tensor_cumsum = torch.Tensor.cumsum


def _cumsum_int(value, *args, **kwargs):
    if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
        value = value.to(torch.int32)
    return _original_cumsum(value, *args, **kwargs)


def _tensor_cumsum_int(self, *args, **kwargs):
    if self.dtype == torch.bool:
        return _original_tensor_cumsum(self.to(torch.int32), *args, **kwargs)
    return _original_tensor_cumsum(self, *args, **kwargs)


torch.cumsum = _cumsum_int
torch.Tensor.cumsum = _tensor_cumsum_int

original_update = DynamicLayer.update


def ov_cache_update(self, key_states, value_states, *a, **k):
    if not self.is_initialized:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys, self.values = key_states.clone(), value_states.clone()
        self.is_initialized = True
        return self.keys, self.values
    return original_update(self, key_states, value_states, *a, **k)


DynamicLayer.update = ov_cache_update


class Wrapper(torch.nn.Module):
    def __init__(self, inner, cams):
        super().__init__()
        self.inner = inner
        self.cams = cams

    def forward(self, *args):
        images = list(args[: self.cams])
        masks = list(args[self.cams: 2 * self.cams])
        tokens, token_mask, noise = args[2 * self.cams:]
        return self.inner.sample_actions(images, masks, tokens, token_mask, noise=noise)


def variant_values(values):
    """Keep cameras 0..C-1 and T tokens of a 3-camera, 200-token sample.

    T above the 200 tokens the stored samples carry (the 256-token window)
    is reached by padding with token id 0 and mask False, which is what a
    short prompt looks like in the served window anyway.
    """
    out = []
    for i in range(C):
        out.append(values[f"input.{1236 + i}"])
    for i in range(C):
        out.append(values[f"input.{1239 + i}"])
    for key, pad in (("input.1242", 0), ("input.1243", False)):
        value = values[key]
        if T <= value.shape[1]:
            out.append(np.ascontiguousarray(value[:, :T]))
        else:
            grown = np.full((value.shape[0], T), pad, dtype=value.dtype)
            grown[:, : value.shape[1]] = value
            out.append(grown)
    out.append(values["input.1244"])
    return out


zero = variant_values(load_file(str(src / "zero.safetensors")))
inputs = tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in zero)
wrapper = Wrapper(base.load_model("cpu").inner, C)
started = time.perf_counter()
print("TRACE_START", [tuple(x.shape) for x in inputs], flush=True)
with torch.inference_mode():
    traced = torch.jit.trace(wrapper, inputs, check_trace=False, strict=False)
    ir = ov.convert_model(traced, input=[list(x.shape) for x in inputs])
trace_seconds = time.perf_counter() - started
names = [port.get_any_name() for port in ir.inputs]
print("CONVERTED", trace_seconds, names, flush=True)
fp_xml = out / f"pi05_droid_cams{C}_tokens{T}_fp.xml"
ov.serialize(ir, str(fp_xml))
del traced, wrapper
gc.collect()

# Samples keyed by the exported names, in the exported order.
for sample in ("zero", "random"):
    vals = variant_values(load_file(str(src / f"{sample}.safetensors")))
    save_file(dict(zip(names, vals)), str(folder / f"{sample}.safetensors"))
rnd = variant_values(load_file(str(src / "random.safetensors")))
for case in range(6):
    rng = np.random.default_rng(80520 + case)
    vals = [v.copy() for v in rnd]
    for image in range(C):
        vals[image] = rng.uniform(-1, 1, vals[image].shape).astype(np.float32)
        vals[C + image][:] = case % 4 != image
    vals[2 * C] = rng.integers(0, 256000, (1, T), dtype=np.int64)
    vals[2 * C + 1][:] = np.arange(T) < min(T, 24 + 29 * case)
    vals[2 * C + 2] = rng.normal(0, 0.25 + case * 0.35, (1, 15, 32)).astype(np.float32)
    save_file(dict(zip(names, vals)), str(folder / f"validation-{case}.safetensors"))

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
    "cams": C, "tokens": T, "input_names": names, "trace_dtype": "bfloat16", "trace_device": "cpu",
    "trace_and_convert_seconds": trace_seconds, "compress_seconds": compress_seconds,
    "openvino": ov.__version__, "nncf": nncf.__version__, "torch": torch.__version__,
    "weight_compression": "INT8_SYM", "input_shapes": [list(x.shape) for x in inputs],
}, indent=2) + "\n")
print("DONE", xml, flush=True)
