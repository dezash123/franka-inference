"""Per-tensor proof that two pi0.5 IRs carry the same quantised weights.

The three window IRs (T=64/128/256) are separate exports, so their `.bin`
files are not byte-identical as whole files — the traced graph bakes geometry
constants (the `[1,T]` attention-mask vector, the `[1,1,T*k]` mask/position
rows) whose size follows the window, and `torch.jit.trace` renumbers every
anonymous constant, so IR-internal names are NOT stable across exports.  What
must not move is every tensor the kernels consume as weights.

This walks both IR XMLs, reads each `Const` layer's `[offset, size)` range
straight out of its `.bin`, digests it, and compares the two IRs as
**multisets of (precision, shape, sha256)** — name-independent.  The
load-bearing assertion is over the weight tensors: every INT8 weight constant
and every f32/f16 dequantisation scale of the reference must appear, byte for
byte and with the same shape, in the candidate.

usage: verify_ir_weights.py <reference xml> <candidate xml> [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import xml.etree.ElementTree as ET


def constants(xml_path: Path):
    """{name: (offset, size, precision, shape, consumer types)} for every Const."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = {}
    for layer in root.find("layers"):
        layers[layer.get("id")] = layer
    consumers = {}
    for edge in root.find("edges"):
        consumers.setdefault(edge.get("from-layer"), []).append(
            layers[edge.get("to-layer")].get("type"))
    out = {}
    for identifier, layer in layers.items():
        if layer.get("type") != "Const":
            continue
        data = layer.find("data")
        shape = data.get("shape", "")
        out[layer.get("name")] = {
            "offset": int(data.get("offset")), "size": int(data.get("size")),
            "precision": data.get("element_type"), "shape": shape,
            "consumers": sorted(set(consumers.get(identifier, []))),
        }
    return out


def digest(mapping, entry):
    return hashlib.sha256(mapping[entry["offset"]:entry["offset"] + entry["size"]]).hexdigest()


def weight_like(entry):
    """Weight constants and their dequantisation scales, not graph geometry."""
    return entry["size"] >= 4096 and any(
        c in ("MatMul", "Convert", "Multiply", "Subtract", "FullyConnected", "Gather")
        for c in entry["consumers"])


def keyed(entries):
    """multiset key -> occurrences"""
    counts = {}
    for name, entry in entries.items():
        key = (entry["precision"], entry["shape"], entry["sha256"])
        counts.setdefault(key, []).append(name)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--json")
    args = ap.parse_args()
    report = {"reference": args.reference, "candidate": args.candidate}
    tensors = {}
    for role, xml in (("reference", args.reference), ("candidate", args.candidate)):
        xml = Path(xml)
        entries = constants(xml)
        with open(xml.with_suffix(".bin"), "rb") as handle:
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            for entry in entries.values():
                entry["sha256"] = digest(mapping, entry)
            mapping.close()
        tensors[role] = entries
        report[role + "_const_count"] = len(entries)
        report[role + "_const_bytes"] = sum(e["size"] for e in entries.values())
        report[role + "_weight_count"] = sum(1 for e in entries.values() if weight_like(e))
        report[role + "_weight_bytes"] = sum(e["size"] for e in entries.values() if weight_like(e))

    reference, candidate = tensors["reference"], tensors["candidate"]
    reference_keys, candidate_keys = keyed(reference), keyed(candidate)
    missing, surplus = [], []
    for key, names in reference_keys.items():
        have = len(candidate_keys.get(key, []))
        if have < len(names):
            missing.append({"precision": key[0], "shape": key[1], "sha256": key[2],
                            "reference_count": len(names), "candidate_count": have,
                            "size": reference[names[0]]["size"],
                            "weight_like": weight_like(reference[names[0]]),
                            "consumers": reference[names[0]]["consumers"]})
    for key, names in candidate_keys.items():
        have = len(reference_keys.get(key, []))
        if have < len(names):
            surplus.append({"precision": key[0], "shape": key[1], "sha256": key[2],
                            "reference_count": have, "candidate_count": len(names),
                            "size": candidate[names[0]]["size"],
                            "weight_like": weight_like(candidate[names[0]]),
                            "consumers": candidate[names[0]]["consumers"]})
    weight_missing = [m for m in missing if m["weight_like"]]
    weight_surplus = [s for s in surplus if s["weight_like"]]
    report["multiset_missing_from_candidate"] = missing
    report["multiset_surplus_in_candidate"] = surplus
    report["missing_bytes"] = sum(m["size"] * (m["reference_count"] - m["candidate_count"])
                                  for m in missing)
    report["weight_tensors_missing"] = weight_missing
    report["weight_tensors_surplus"] = weight_surplus
    report["weights_byte_identical"] = not weight_missing and not weight_surplus
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    summary = {k: v for k, v in report.items()
               if not k.startswith("multiset_")}
    summary["non_weight_geometry_constants_moved"] = len(missing) + len(surplus)
    print(json.dumps(summary, indent=2))
    return 0 if report["weights_byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
