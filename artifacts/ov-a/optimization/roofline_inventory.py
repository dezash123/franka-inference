"""Freeze source-operation rooflines independently of candidate latency.

Run on dstack. Compiled precision and measured ceilings are separate evidence.
The aggregate compute bound is the primary, stricter target. The phase bound
adds mandatory expert-weight reads, allowing a generous 18 MiB cache reuse.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("probe")
    ap.add_argument("output")
    args = ap.parse_args()
    inventory = json.loads(Path(args.inventory).read_text())
    probe = json.loads(Path(args.probe).read_text())
    by_phase = defaultdict(lambda: {"int8_ops": 0, "fp16_ops": 0, "weight_ranges": set()})
    for r in inventory["records"]:
        if not r["dynamic"]:
            continue
        phase = r["phase"]
        p = by_phase[phase]
        kind = "int8_ops" if r.get("compressed_i8_weight") else "fp16_ops"
        p[kind] += r["useful_matrix_ops"]
        p["weight_ranges"].update(tuple(v) for v in r.get("weight_ranges", []))
    int8_ops = sum(p["int8_ops"] for p in by_phase.values())
    fp16_ops = sum(p["fp16_ops"] for p in by_phase.values())
    peak_i8, peak_f16, bandwidth = 233.472, 116.736, 456
    aggregate_ms = int8_ops/(peak_i8*1e9) + fp16_ops/(peak_f16*1e9)
    phase_ms = 0
    for name, p in by_phase.items():
        unique_weights = sum(n for _, n in p.pop("weight_ranges"))
        p["unique_matmul_weights_bytes"] = unique_weights
        p["ideal_matrix_ms"] = p["int8_ops"]/(peak_i8*1e9) + p["fp16_ops"]/(peak_f16*1e9)
        # Vision camera batching reads one copy; prefix weights one copy.
        # Expert weights are revisited across five sequential flow steps.
        compulsory = unique_weights if name != "action_expert" else max(0, 5*unique_weights - 4*18*1024**2)
        p["lower_bound_weight_read_bytes"] = compulsory
        p["ideal_memory_ms"] = compulsory/(bandwidth*1e6)
        p["phase_lower_bound_ms"] = max(p["ideal_matrix_ms"], p["ideal_memory_ms"])
        phase_ms += p["phase_lower_bound_ms"]
    result = {
        "primary": {
            "definition": "Aggregate dense useful matrix compute bound; no padding, sparse TOPS, or launch overhead in denominator",
            "int8_ops": int8_ops, "fp16_ops": fp16_ops,
            "int8_peak_tops": peak_i8, "fp16_peak_tflops": peak_f16,
            "ideal_ms": aggregate_ms, "target_80_percent_ms": aggregate_ms/0.8,
        },
        "secondary_phase_bound": {
            "definition": "Sum of sequential phase maxima of compute and compulsory matrix-weight traffic",
            "advertised_bandwidth_GB_s": bandwidth,
            "ideal_ms": phase_ms, "target_80_percent_ms": phase_ms/0.8,
            "phases": dict(by_phase),
        },
        "measured_hardware": {
            "int8_tops": probe["int8_tops"],
            "copy_bandwidth_GB_s": probe["bandwidth"]["GB_per_second"],
        },
        "limitations": [
            "Assumes all compressed matrix work can run W8A8; existing FP16 fallback is an implementation cost.",
            "Scalar norm, activation, softmax, quantization and launch costs do not inflate the compute bound.",
            "FP16 matrix throughput is the device-reported dense peak, not an independent probe from this run.",
            "Phase memory bound assumes expert weight cache capacity of at most 18 MiB; verify architecture before using as a claim.",
            "Neither bound includes non-matrix compulsory traffic; this deliberately sets a demanding target.",
            "Global aggregate is the frozen primary target; phase bound cannot silently replace it.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
