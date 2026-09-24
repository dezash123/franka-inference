"""Derive language-token-count variants of the input samples.

Keeps images, camera masks, and noise; truncates token ids and the token mask
to T positions. Writes <sample>.safetensors under inputs/tokens-<T>/ and the
validation cases regenerated with the same seeds as
generate_validation_cases.py (mask lengths capped at T).
"""
import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--inputs", default="inputs")
args = ap.parse_args()
T = args.tokens
src = Path(args.inputs)
dst = src / f"tokens-{T}"
dst.mkdir(exist_ok=True)


def truncate(values):
    out = {k: v.copy() for k, v in values.items()}
    out["input.1242"] = np.ascontiguousarray(values["input.1242"][:, :T])
    out["input.1243"] = np.ascontiguousarray(values["input.1243"][:, :T])
    return out


for sample in ("zero", "random"):
    save_file(truncate(load_file(str(src / f"{sample}.safetensors"))), str(dst / f"{sample}.safetensors"))
base = truncate(load_file(str(src / "random.safetensors")))
for case in range(6):
    rng = np.random.default_rng(80520 + case)
    values = {k: v.copy() for k, v in base.items()}
    for image in range(3):
        values[f"input.{1236+image}"] = rng.uniform(-1, 1, base[f"input.{1236+image}"].shape).astype(np.float32)
        values[f"input.{1239+image}"][:] = case % 4 != image
    values["input.1242"] = rng.integers(0, 256000, (1, T), dtype=np.int64)
    values["input.1243"][:] = np.arange(T) < min(T, 24 + 29 * case)
    values["input.1244"] = rng.normal(0, 0.25 + case * 0.35, (1, 15, 32)).astype(np.float32)
    save_file(values, str(dst / f"validation-{case}.safetensors"))
print("WROTE", dst, "tokens", T)
