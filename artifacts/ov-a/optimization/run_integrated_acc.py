"""Run full-model ablations sequentially, with auditable tuning and cache keys."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=("reuse", "gemm", "attention", "combined"), required=True)
ap.add_argument("--profile", type=int, default=0)
ap.add_argument("--iterations", type=int, default=30)
ap.add_argument("--validation-repeats", type=int, default=1)
ap.add_argument("--tag", default="round2")
ap.add_argument("--result-suffix", default="",
                help="Distinct repeat artifacts using the same compiled-model cache")
ap.add_argument("--plugins", default="/workflow/custom-plugin-integrated/plugins.xml")
ap.add_argument("--gateup-folder", help="Experimental native gate/up host.cl and gateup.h directory")
ap.add_argument("--trace-region", action="store_true")
overrides_group = ap.add_mutually_exclusive_group()
overrides_group.add_argument("--tuning-env", default="{}", help="Explicit PI05_ overrides recorded in result.json")
overrides_group.add_argument("--tuning-file", help="JSON file with explicit PI05_ overrides")
ap.add_argument("--allow-roundoff", action="store_true", help="Use the existing numerical gate instead of bitwise equality")
ap.add_argument("--float-gate", help="Envelope JSON for the accuracy-versus-FP32 gate (replaces the bitwise requirement)")
ap.add_argument("--dump-sources", help="Native generated kernel source directory for diagnostics")
ap.add_argument("--root", default="/workflow", help="Sample/reference root passed to profile_openvino.py")
ap.add_argument("--model", help="IR path passed to profile_openvino.py (default: <root>/model/pi05_droid_dynamic_w8a8.xml)")
args = ap.parse_args()
# W8A8Line bisect: which of profile_openvino.py's graph rewrites are on.
_XFORM = os.environ.get("LANE_XFORM", "fuseqkv,batchvision,alignmlp,fusepad")
# Which saved reference the bitwise gate is taken against.  The corrected
# line ships gate/reference-corrected -> backend name "custom-corrected";
# the retired defective-graph references were "custom-per-token-fusedpad".
_BITREF = os.environ.get("LANE_BITWISE_REF", "custom-per-token-fusedpad")

env = dict(os.environ)
for key in list(env):
    if key.startswith("PI05_"):
        del env[key]
env.update(
    PI05_DQ_REUSE="1",
    PI05_GEMM_TABLE="16384,968,2048,-1,64,32|wg 4x8",
    PI05_SDPA_TABLE="256,968,968|16,32,32,32,8,2,8,2",
)
if args.variant in ("gemm", "combined"):
    env["PI05_GEMM_TABLE"] = (
        "16384,968,2048,0,64,32|wg 4x8;"
        "16384,968,2048,1,32,64|wg 2x4;"
        "2048,968,16384,-1,32,32|wg 4x8"
    )
if args.variant in ("attention", "combined"):
    env["PI05_SDPA_TABLE"] = (
        "256,968,968|32,32,32,32,8,1,8,1;"
        "72,256,256|16,32,32,32,4,2,4,2"
    )
if args.gateup_folder:
    env["PI05_GATEUP_SOURCE"] = str(Path(args.gateup_folder)/"host.cl")
    env["PI05_GATEUP_HEADER"] = str(Path(args.gateup_folder)/"gateup.h")
overrides = json.loads(Path(args.tuning_file).read_text() if args.tuning_file else args.tuning_env)
assert isinstance(overrides, dict) and all(k.startswith("PI05_") and isinstance(v, str) for k, v in overrides.items())
env.update(overrides)
# Shape tables are written for the 968-token prefix; retarget them when a
# tuning file selects another prefix length (768 vision + language tokens).
if "PI05_PREFIX_TOKENS" in overrides and overrides["PI05_PREFIX_TOKENS"] != "968":
    prefix = overrides["PI05_PREFIX_TOKENS"]
    for key in ("PI05_GEMM_TABLE", "PI05_SDPA_TABLE"):
        if key in env and key not in overrides:
            env[key] = env[key].replace("968", prefix)
label = f"integrated-{args.variant}-{args.tag}-profile{args.profile}"
if args.trace_region:
    label += "-trace"
if args.result_suffix:
    assert "/" not in args.result_suffix and "\\" not in args.result_suffix
    label += "-" + args.result_suffix
root = Path(args.root)
out = root / "results" / label
config = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615"}
if args.dump_sources:
    env["OV_GPU_DUMP_SOURCES_PATH"] = args.dump_sources
cmd = [
    sys.executable, str(Path(__file__).with_name("profile_openvino.py")),
    "--output", str(out), "--profile", str(args.profile),
    "--root", str(root), "--model", args.model or str(root / "model/pi05_droid_dynamic_w8a8.xml"),
    "--iterations", str(args.iterations),
    "--validation-repeats", str(args.validation_repeats),
    "--plugins", args.plugins,
    "--cache-tag", f"integrated-{args.variant}-{args.tag}",
    *([] if "fuseqkv" not in _XFORM else ["--fuse-qkv", "vision,language,expert"]),
    *([] if "batchvision" not in _XFORM else ["--batch-vision"]),
    *([] if "alignmlp" not in _XFORM else ["--align-vision-mlp"]),
    *([] if "fusepad" not in _XFORM else ["--fuse-vision-padding"]),
    "--config", json.dumps(config),
    "--validation-samples", os.environ["LANE_SAMPLES"],
]
if args.float_gate:
    cmd += ["--require-float-gate", args.float_gate]
elif args.variant in ("reuse", "gemm") and not args.allow_roundoff:
    cmd += ["--require-bitwise", _BITREF]
if args.gateup_folder:
    cmd += ["--require-gateup-count", "107" if "PI05_EXPERT_GATEUP_SOURCE" in env else "17"]
if "PI05_RMS_QUANTIZE_SOURCE" in env:
    cmd += ["--require-rms-quantize-count", "35"]
if "PI05_MVN_QUANTIZE_SOURCE" in env:
    cmd += ["--require-mvn-quantize-count", "54"]
if "PI05_ADALN_QUANTIZE_SOURCE" in env:
    cmd += ["--require-adaln-quantize-count", "185"]
if "PI05_LARGE_QUANTIZE_SOURCE" in env:
    cmd += ["--require-large-quantize-count", "17"]
if "PI05_VISION_QUANTIZE_SOURCE" in env:
    cmd += ["--require-vision-quantize-count", "27"]
if "PI05_TRANSPOSE_QUANTIZE_SOURCE" in env:
    cmd += ["--require-transpose-quantize-count", "134"]
if "PI05_QKV_SPLIT_SOURCE" in env:
    scope = env.get("PI05_QKV_SPLIT_SCOPE", "vision,prefix,expert")
    count = sum(n for family,n in (("vision",27),("prefix",17),("expert",90)) if family in scope)
    if "PI05_QKV_ROPE_SOURCE" in env:
        count -= 107
    cmd += ["--require-qkv-split-count", str(count)]
if "PI05_PACKED_ROPE_SOURCE" in env:
    scope = env.get("PI05_PACKED_ROPE_SCOPE", "prefix,expert")
    count = sum(n for family,n in (("prefix",34),("expert",180)) if family in scope)
    if "PI05_QKV_ROPE_SOURCE" in env:
        count = 0
    cmd += ["--require-packed-rope-count", str(count)]
if "PI05_QKV_ROPE_SOURCE" in env:
    cmd += ["--require-qkv-rope-count", "107"]
if "PI05_QKV_ROPE_KV_SOURCE" in env:
    cmd += ["--require-qkv-rope-kv-count", "90"]
if args.trace_region:
    cmd += ["--trace-region"]
if os.environ.get("LANE_NO_REQUIRE"):
    # Bisect mode: the --require-*-count gates assert the fusion counts of the
    # SELECTED configuration.  When a graph rewrite is deliberately disabled the
    # counts legitimately change, so drop the gates and keep the kernels.
    filtered, skip = [], 0
    for token in cmd:
        if skip:
            skip -= 1
            continue
        if token.startswith("--require-") and token.endswith("-count"):
            skip = 1
            continue
        filtered.append(token)
    cmd = filtered
log_path = root / f"{label}.log"
with log_path.open("w") as log:
    code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
if code:
    raise RuntimeError(log_path.read_text()[-5000:])
result = json.loads((out / "result.json").read_text())
if not args.float_gate and not args.allow_roundoff:
    for sample, checks in result["sanity_checks"].items():
        check = checks.get(_BITREF)
        if check is None:
            continue   # acc_A real-frame samples have no saved per-token reference
        assert check["cosine"] > 0.99999 and check["rmse"] < 0.005, (sample, check)
print(json.dumps({"label": label, "median_ms": result["timing"]["median_ms"],
                  "checks": result["sanity_checks"]}), flush=True)
