#!/usr/bin/env python3
"""Degradation report for the c2t64 OpenVINO W8A8 line on real DROID frames.

Metric is RuntimeCore/acc_report.py's: rel_l2 = ||a-b||_2/||b||_2 in fp64 over
the [15,32] chunk, plus the same restricted to actions[:, :8] (the 8 real DROID
action dims; 8..31 are padding).  Reported: mean / first-8 mean / worst /
worst sample / median over the 40 acc_A samples.

usage: acc_compare.py <ov_dir> <torch_dir> <accA_dir> <torch_synth_dir> <bundle_ref_dir> <out.json>
"""
import json
import os
import pathlib
import sys

import numpy as np

SAMPLES = ["s%02d" % n for n in range(40)]
SYNTH = ["zero", "random"] + ["validation-%d" % i for i in range(6)]


def rel_l2(a, b, dims=None):
    a = np.asarray(a, dtype=np.float64).reshape(15, 32)
    b = np.asarray(b, dtype=np.float64).reshape(15, 32)
    if dims is not None:
        a, b = a[:, :dims], b[:, :dims]
    n = np.linalg.norm(b)
    d = np.linalg.norm(a - b)
    return float(d / n) if n > 0 else (1.0 if d > 0 else 0.0)


def row(pairs):
    """pairs: {sample: (actual, reference)} -> the offline statistics."""
    full = {k: rel_l2(a, b) for k, (a, b) in pairs.items()}
    first8 = {k: rel_l2(a, b, 8) for k, (a, b) in pairs.items()}
    worst = max(full, key=full.get)
    worst8 = max(first8, key=first8.get)
    return {
        "n": len(full),
        "mean": float(np.mean(list(full.values()))),
        "median": float(np.median(list(full.values()))),
        "worst": full[worst], "worst_sample": worst,
        "first8_mean": float(np.mean(list(first8.values()))),
        "first8_worst": first8[worst8], "first8_worst_sample": worst8,
        "per_sample": {k: [full[k], first8[k]] for k in sorted(full)},
    }


def fmt(name, r):
    return "| %s | %d | %.4f | %.4f | %.4f (%s) | %.4f (%s) |" % (
        name, r["n"], r["mean"], r["first8_mean"], r["worst"], r["worst_sample"],
        r["first8_worst"], r["first8_worst_sample"])


def main():
    ov, torch_dir, acc, synth, bundle, outp = (pathlib.Path(p) for p in sys.argv[1:7])
    L = np.load
    ovv = {s: L(ov / ("accA-%s.npy" % s)) for s in SAMPLES}
    b5 = {s: L(torch_dir / ("accA-%s.b5.npy" % s)) for s in SAMPLES}
    b10 = {s: L(torch_dir / ("accA-%s.b10.npy" % s)) for s in SAMPLES}
    denseA = {s: L(acc / s / "actions.dense10.npy") for s in SAMPLES}
    cfgA = {s: L(acc / s / "actions.configA_torch.npy") for s in SAMPLES}

    report = {"rows": {}, "controls": {}, "noise_only": {}}
    report["rows"]["ov_w8a8_vs_torch_bf16_5step_ckptB"] = row({s: (ovv[s], b5[s]) for s in SAMPLES})
    report["rows"]["ov_w8a8_vs_torch_bf16_10step_ckptB"] = row({s: (ovv[s], b10[s]) for s in SAMPLES})
    report["rows"]["torch_bf16_5step_vs_10step_ckptB"] = row({s: (b5[s], b10[s]) for s in SAMPLES})
    report["rows"]["ov_w8a8_vs_accA_dense10_ckptA"] = row({s: (ovv[s], denseA[s]) for s in SAMPLES})
    report["rows"]["torch_bf16_10step_ckptB_vs_accA_dense10_ckptA"] = row(
        {s: (b10[s], denseA[s]) for s in SAMPLES})
    report["rows"]["accA_configA_torch_vs_accA_dense10_ckptA"] = row(
        {s: (cfgA[s], denseA[s]) for s in SAMPLES})

    # Pure-noise control: zeroed camera tensors, s00 prompt and noise.
    n_ov = L(ov / "accA-noiseonly.npy")
    n_b5 = L(torch_dir / "accA-noiseonly.b5.npy")
    n_b10 = L(torch_dir / "accA-noiseonly.b10.npy")
    report["noise_only"] = {
        "ov_vs_torch_bf16_5step": [rel_l2(n_ov, n_b5), rel_l2(n_ov, n_b5, 8)],
        "ov_vs_torch_bf16_10step": [rel_l2(n_ov, n_b10), rel_l2(n_ov, n_b10, 8)],
        "torch_5step_vs_10step": [rel_l2(n_b5, n_b10), rel_l2(n_b5, n_b10, 8)],
        "ov_vs_ov_s00": [rel_l2(n_ov, ovv["s00"]), rel_l2(n_ov, ovv["s00"], 8)],
    }

    # Control 1: this CUDA bf16 reference vs the bundle's own torch-eager
    # (B580/XPU bf16) references, on the 8 synthetic gate cases.
    cuda_vs_xpu = {s: (L(synth / ("accA-%s.b5.npy" % s)), L(bundle / ("torch-eager-%s.npy" % s)))
                   for s in SYNTH}
    report["controls"]["cuda_bf16_5step_vs_bundle_torch_eager_xpu"] = row(cuda_vs_xpu)
    # Control 2: the OV device outputs for the same 8 cases vs both float refs.
    ov_synth = {s: L(ov / ("%s.npy" % s)) for s in SYNTH}
    report["controls"]["ov_w8a8_vs_bundle_torch_eager_xpu_synth"] = row(
        {s: (ov_synth[s], L(bundle / ("torch-eager-%s.npy" % s))) for s in SYNTH})
    report["controls"]["ov_w8a8_vs_cuda_bf16_5step_synth"] = row(
        {s: (ov_synth[s], L(synth / ("accA-%s.b5.npy" % s))) for s in SYNTH})
    # Control 3: the bitwise gate, recomputed from the saved device outputs.
    # The bitwise gate of the SHIPPED (corrected) configuration.  The retired
    # "custom-per-token-fusedpad" references belong to the defective graph, and
    # the corrected build is deliberately NOT bitwise to them.
    ref = os.environ.get("ACC_BITWISE_REF", "custom-corrected")
    bitwise = {s: bool(np.array_equal(ov_synth[s], L(bundle / ("%s-%s.npy" % (ref, s)))))
               for s in SYNTH}
    report["controls"]["bitwise_vs_" + ref.replace("-", "_")] = bitwise

    pathlib.Path(outp).write_text(json.dumps(report, indent=2) + "\n")
    print("| row | n | mean rel-L2 (32d) | first-8 mean | worst (32d) | worst first-8 |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, r in report["rows"].items():
        print(fmt(k, r))
    print()
    for k, r in report["controls"].items():
        if k.startswith("bitwise_vs_"):
            print("bitwise:", all(r.values()), r)
        else:
            print(fmt(k, r))
    print()
    print("noise_only:", json.dumps(report["noise_only"]))


if __name__ == "__main__":
    main()
