#!/usr/bin/env python3
"""Regenerate the c2t64 line's gates from the CORRECTED IR.

The old gates cannot see a defect that both of their arms share:

  * the default gate is bitwise against `custom-per-token-fusedpad`, which is
    another build of the same custom plugin on the same graph;
  * `c2t64-float-gate-envelope.json` records this build's accepted error
    against FP32 and was calibrated on the already-defective graph.

This writes the replacements:

  1. `reference-corrected/<sample>.npy` — the 8 synthetic outputs of the
     SELECTED corrected configuration, i.e. the new bitwise reference.
  2. `c2t64-float-gate-envelope-corrected.json` — per-sample rmse/cosine of
     that configuration against the bundle's FP32 eager references, with the
     envelope set at the documented slack.
  3. `gate-realframe.json` — the real-frame gate: the corrected custom build
     against the STOCK plugin on the same IR over the 40 acc_A frames, all-32
     and task dims, which is the gate that would have caught both defects.

usage: make_gate.py <custom_dir> <stock_dir> <fp32_ref_dir> <torch_dir> <acc_dir> <out_dir>
"""
import json
import pathlib
import sys

import numpy as np

SYNTH = ["zero", "random"] + ["validation-%d" % i for i in range(6)]
SAMPLES = ["s%02d" % n for n in range(40)]
RMSE_SLACK = 1.25      # envelope headroom over the measured value
COSINE_SLACK = 1e-5


def rel_l2(a, b, dims=32):
    a = np.asarray(a, np.float64).reshape(15, 32)[:, :dims]
    b = np.asarray(b, np.float64).reshape(15, 32)[:, :dims]
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n > 0 else 0.0


def compare(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return {"max_abs": float(np.max(np.abs(a - b))),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))}


def main():
    custom, stock, fp32, torch_dir, acc, out = (pathlib.Path(p) for p in sys.argv[1:7])
    (out / "reference-corrected").mkdir(parents=True, exist_ok=True)

    envelope, bitwise = {}, {}
    for s in SYNTH:
        value = np.load(custom / ("%s.npy" % s))
        np.save(out / "reference-corrected" / ("custom-corrected-%s.npy" % s), value)
        bitwise[s] = "custom-corrected-%s.npy" % s
        ref = fp32 / ("torch-eager-fp32-%s.npy" % s)
        if ref.exists():
            c = compare(value, np.load(ref))
            envelope[s] = {"measured": c,
                           "rmse_max": round(c["rmse"] * RMSE_SLACK, 6),
                           "cosine_min": round(c["cosine"] - COSINE_SLACK, 9)}
    (out / "c2t64-float-gate-envelope-corrected.json").write_text(
        json.dumps({"samples": envelope,
                    "derived_from": "the corrected IR + the selected configuration "
                                    "(batch_vision disabled), vs the bundle's FP32 eager references",
                    "rmse_slack": RMSE_SLACK, "cosine_slack": COSINE_SLACK}, indent=2) + "\n")

    # the real-frame gate: custom vs stock on the same IR, and both vs torch
    rows = {}
    per_sample = {}
    for name, d, pat in (("custom_vs_stock", custom, "accA-%s.npy"),):
        pass
    cs32, cs8, ct32, ct8, st32, st8 = [], [], [], [], [], []
    for s in SAMPLES:
        c = np.load(custom / ("accA-%s.npy" % s))
        k = np.load(stock / ("accA-%s.fxdq.npy" % s))
        t = np.load(torch_dir / ("accA-%s.b5.npy" % s))
        cs32.append(rel_l2(c, k)); cs8.append(rel_l2(c, k, 8))
        ct32.append(rel_l2(c, t)); ct8.append(rel_l2(c, t, 8))
        st32.append(rel_l2(k, t)); st8.append(rel_l2(k, t, 8))
        per_sample[s] = {"custom_vs_stock": [cs32[-1], cs8[-1]],
                         "custom_vs_torch": [ct32[-1], ct8[-1]],
                         "stock_vs_torch": [st32[-1], st8[-1]]}
    for key, (a, b) in (("custom_vs_stock", (cs32, cs8)),
                        ("custom_vs_torch_bf16", (ct32, ct8)),
                        ("stock_vs_torch_bf16", (st32, st8))):
        rows[key] = {"n": len(a), "mean": float(np.mean(a)), "first8_mean": float(np.mean(b)),
                     "worst": float(max(a)), "worst_sample": SAMPLES[int(np.argmax(a))]}
    bar = rows["stock_vs_torch_bf16"]["mean"] * 1.5
    gate = {
        "metric": "rel-L2 in fp64 over the [15,32] chunk; first8 = actions[:, :8]",
        "reference": "the SAME IR on the stock OpenVINO GPU plugin with per-token INT8 "
                     "activations (DYNAMIC_QUANTIZATION_GROUP_SIZE=UINT64_MAX)",
        "frames": "acc_A s00..s39, real DROID, checkpoint B, harness noise",
        "rows": rows,
        "bar_1p5x_stock_all32": bar,
        "passes_1p5x_bar": rows["custom_vs_torch_bf16"]["mean"] <= bar,
        "per_sample": per_sample,
    }
    (out / "gate-realframe.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "bitwise-reference-index.json").write_text(json.dumps(bitwise, indent=2) + "\n")

    print("| gate row | n | mean (32d) | first-8 | worst |")
    print("|---|---:|---:|---:|---:|")
    for k, r in rows.items():
        print("| %s | %d | %.4f | %.4f | %.4f (%s) |"
              % (k, r["n"], r["mean"], r["first8_mean"], r["worst"], r["worst_sample"]))
    print()
    print("1.5x-of-stock bar (all-32): %.4f -- custom is %.4f -> %s"
          % (bar, rows["custom_vs_torch_bf16"]["mean"],
             "PASS" if gate["passes_1p5x_bar"] else "FAIL"))
    print("regenerated %d bitwise references and %d envelope entries"
          % (len(bitwise), len(envelope)))


if __name__ == "__main__":
    main()
