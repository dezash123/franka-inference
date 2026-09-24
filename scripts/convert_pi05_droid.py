#!/usr/bin/env python3
"""Run the official OpenPI Orbax/JAX -> PyTorch PI05 conversion on CPU."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

CHECKPOINT_DIR = Path("/home/spring/robo_run/checkpoints/pi05_droid")
OUTPUT_DIR = Path("/home/spring/robo_run/checkpoints/pi05_droid_pytorch")
OPENPI_DIR = Path("/home/spring/robo_run/openpi")
RESULT_JSON = Path("/home/spring/robo_run/results/convert_pi05_droid.json")
RESULT_MD = Path("/home/spring/robo_run/results/convert_pi05_droid.md")
CONFIG_NAME = "pi05_droid"
PRECISION = "bfloat16"


def package_versions() -> dict[str, str]:
    names = ["python", "jax", "jaxlib", "orbax.checkpoint", "flax", "numpy", "torch", "safetensors", "tyro", "transformers", "openpi"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            if name == "python":
                versions[name] = sys.version.split()[0]
                continue
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if version is None:
                version = importlib.metadata.version({"orbax.checkpoint": "orbax-checkpoint"}.get(name, name))
            versions[name] = str(version)
        except Exception as exc:  # metadata must remain useful even on blocked conversion
            try:
                versions[name] = importlib.metadata.version({"orbax.checkpoint": "orbax-checkpoint"}.get(name, name))
            except importlib.metadata.PackageNotFoundError:
                versions[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


def files_under(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        {"path": str(item.relative_to(path)), "bytes": item.stat().st_size}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def write_results(status: str, started: float, error: str | None, peak_rss_kib: int) -> None:
    output_files = files_under(OUTPUT_DIR)
    norm_stats = OUTPUT_DIR / "assets" / "droid" / "norm_stats.json"
    data = {
        "conversion_status": status,
        "jax_inference": False,
        "inference_performed": False,
        "input_checkpoint": str(CHECKPOINT_DIR),
        "output_checkpoint": str(OUTPUT_DIR),
        "config_name": CONFIG_NAME,
        "precision": PRECISION,
        "converter": str(OPENPI_DIR / "examples" / "convert_jax_model_to_pytorch.py"),
        "converter_command": [sys.executable, str(OPENPI_DIR / "examples" / "convert_jax_model_to_pytorch.py"), "--checkpoint_dir", str(CHECKPOINT_DIR), "--config_name", CONFIG_NAME, "--output_path", str(OUTPUT_DIR), "--precision", PRECISION],
        "git_commit": subprocess.run(["git", "-C", str(OPENPI_DIR), "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_kib": peak_rss_kib,
        "package_versions": package_versions(),
        "output_files": output_files,
        "output_bytes": sum(int(item["bytes"]) for item in output_files),
        "norm_stats": {"path": str(norm_stats), "present": norm_stats.is_file(), "bytes": norm_stats.stat().st_size if norm_stats.is_file() else 0},
        "pytorch_inference_ready": status == "success" and (OUTPUT_DIR / "model.safetensors").is_file() and norm_stats.is_file(),
        "error": error,
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# PI05 DROID JAX-to-PyTorch conversion",
        "",
        f"- Status: `{status}`",
        f"- JAX inference: `{data['jax_inference']}` (conversion only)",
        f"- Input: `{CHECKPOINT_DIR}`",
        f"- Output: `{OUTPUT_DIR}`",
        f"- Config: `{CONFIG_NAME}`; precision: `{PRECISION}`",
        f"- Git commit: `{data['git_commit']}`",
        f"- Elapsed seconds: `{data['elapsed_seconds']}`",
        f"- Peak RSS: `{peak_rss_kib} KiB`",
        f"- Output bytes: `{data['output_bytes']}`; files: `{len(output_files)}`",
        f"- DROID norm stats: `{data['norm_stats']['path']}` present=`{data['norm_stats']['present']}`",
        f"- PyTorch-inference-ready: `{data['pytorch_inference_ready']}`",
        "",
        "## Converter command",
        "",
        "```text",
        " ".join(data["converter_command"]),
        "```",
        "",
        "## Package versions",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in data["package_versions"].items())
    lines += ["", "## Output files", ""]
    lines.extend(f"- `{item['path']}` ({item['bytes']} bytes)" for item in output_files)
    if error:
        lines += ["", "## Error", "", "```text", error, "```"]
    RESULT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    started = time.monotonic()
    error: str | None = None
    status = "blocked"
    peak_rss_kib = 0
    command = [sys.executable, str(OPENPI_DIR / "examples" / "convert_jax_model_to_pytorch.py"), "--checkpoint_dir", str(CHECKPOINT_DIR), "--config_name", CONFIG_NAME, "--output_path", str(OUTPUT_DIR), "--precision", PRECISION]
    env = os.environ.copy()
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"})
    try:
        if not (CHECKPOINT_DIR / "params").is_dir() or list((CHECKPOINT_DIR / "params").glob("*.part")):
            raise RuntimeError("input checkpoint is incomplete or contains .part files")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=OPENPI_DIR, env=env, check=False)
        peak_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if completed.returncode != 0:
            raise RuntimeError(f"official converter exited with status {completed.returncode}")
        # The official converter expects assets beside the checkpoint directory.
        # Downloaded openpi-assets checkpoints carry assets inside that directory.
        source_assets = CHECKPOINT_DIR / "assets"
        output_assets = OUTPUT_DIR / "assets"
        if source_assets.is_dir() and not output_assets.exists():
            shutil.copytree(source_assets, output_assets)
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        peak_rss_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    write_results(status, started, error, peak_rss_kib)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
