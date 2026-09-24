"""Generate float eager references for every validation sample on CPU.

Loads the trained lerobot/pi05_droid checkpoint exactly as the framework
baseline does, but on CPU in the requested dtype, and saves one output per
sample. FP32 is the accuracy reference; a BF16 run is used to relate the CPU
path to the archived XPU BF16 references.
"""
from __future__ import annotations

import argparse
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
ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
ap.add_argument("--samples", default="zero,random,validation-0,validation-1,validation-2,validation-3,validation-4,validation-5")
ap.add_argument("--output", default="/workflow/reference")
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--cams", type=int, default=3, help="camera inputs per sample (variant exports use fewer)")
ap.add_argument("--input-names", help="JSON list of sample keys in call order (variant exports)")
args = ap.parse_args()

torch.set_num_threads(args.threads)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
tag = "torch-eager-fp32" if args.dtype == "float32" else "torch-eager-cpu-bf16"

model = base.load_model("cpu")
if args.input_names:
    from safetensors.torch import load_file as load_torch
    names = json.loads(Path(args.input_names).read_text())["input_names"]
    inner = model.inner

    class Variant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, *a):
            c = args.cams
            return inner.sample_actions(list(a[:c]), list(a[c:2 * c]), a[2 * c], a[2 * c + 1], noise=a[2 * c + 2])
    model = Variant()

    def load_variant(sample):
        values = load_torch(str(Path(os.environ["PI05_BASELINE_ROOT"]) / f"{sample}.safetensors"), device="cpu")
        return tuple(values[k].contiguous() for k in names)
    base.load_inputs = load_variant
if args.dtype == "float32":
    model.inner.float()
    config = model.inner.config
    # LeRobot casts activations to config.dtype inside the policy; keep the
    # whole computation in FP32 when an FP32 reference is requested.
    if hasattr(config, "dtype"):
        config.dtype = "float32"
dtype_seen = {p.dtype for p in model.parameters()}
print("PARAM_DTYPES", sorted(str(d) for d in dtype_seen), flush=True)
manifest = {"dtype": args.dtype, "device": "cpu", "threads": args.threads, "samples": {}}
with torch.inference_mode():
    for sample in args.samples.split(","):
        inputs = base.load_inputs(sample)
        started = time.perf_counter()
        value = model(*inputs).float().cpu().numpy()
        seconds = time.perf_counter() - started
        assert value.shape == (1, 15, 32) and np.isfinite(value).all(), (sample, value.shape)
        np.save(out / f"{tag}-{sample}.npy", value)
        manifest["samples"][sample] = {"seconds": seconds, "abs_mean": float(np.abs(value).mean())}
        print("SAMPLE", sample, f"{seconds:.1f}s", flush=True)
(out / f"{tag}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("DONE", tag, flush=True)
