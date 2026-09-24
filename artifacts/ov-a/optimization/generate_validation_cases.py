"""Create bounded synthetic functional tests, never an accuracy dataset."""
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file

root = Path("/workflow")
base = load_file(str(root / "random.safetensors"))
for case in range(6):
    rng = np.random.default_rng(80520+case)
    values = {k: v.copy() for k, v in base.items()}
    for image in range(3):
        values[f"input.{1236+image}"] = rng.uniform(-1, 1, base[f"input.{1236+image}"].shape).astype(np.float32)
        values[f"input.{1239+image}"][:] = case % 4 != image
    values["input.1242"] = rng.integers(0, 256000, (1, 200), dtype=np.int64)
    values["input.1243"][:] = np.arange(200) < (24 + 29*case)
    values["input.1244"] = rng.normal(0, 0.25 + case*0.35, (1, 15, 32)).astype(np.float32)
    save_file(values, str(root / f"validation-{case}.safetensors"))
print("WROTE_VALIDATION_CASES", 6)
