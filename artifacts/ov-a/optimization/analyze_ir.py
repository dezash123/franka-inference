"""Static useful-work inventory. Execute on the dstack CPU runner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def phase(name):
    if "vision_tower" in name:
        return "vision"
    if "gemma_expert" in name:
        return "action_expert"
    if "language_model" in name:
        return "language_prefix"
    if "multi_modal_projector" in name:
        return "image_projection"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("output")
    args = ap.parse_args()
    root = ET.parse(args.xml).getroot()
    layers = {}
    for layer in root.find("layers"):
        outputs = list(layer.findall("output/port"))
        layers[layer.attrib["id"]] = {
            **layer.attrib, "data": dict(layer.find("data").attrib) if layer.find("data") is not None else {},
            "inputs": {}, "outputs": {p.attrib["id"]: [int(d.text) for d in p.findall("dim")] for p in outputs},
        }
    for edge in root.find("edges"):
        e = edge.attrib
        layers[e["to-layer"]]["inputs"][int(e["to-port"])] = (e["from-layer"], e["from-port"])
    dynamic, constants = {}, {}

    def visit(node_id):
        if node_id in dynamic:
            return
        node = layers[node_id]
        parents = [x[0] for _, x in sorted(node["inputs"].items())]
        for parent in parents:
            visit(parent)
        # All model parameters and every source port in this IR are static.
        # ShapeOf is compile-time data even when the values are runtime inputs.
        dynamic[node_id] = (node["type"] == "Parameter" or any(dynamic[p] for p in parents)) \
            and node["type"] != "ShapeOf"
        if node["type"] == "Const":
            constants[node_id] = {node_id}
        elif not dynamic[node_id]:
            constants[node_id] = set().union(*(constants[p] for p in parents))
        else:
            constants[node_id] = set()

    for node_id in layers:
        visit(node_id)
    records, by_phase = [], defaultdict(Counter)
    weight_ranges, total_constant_ranges = set(), set()
    for node in layers.values():
        if node["type"] == "Const":
            data = node["data"]
            total_constant_ranges.add((int(data.get("offset", 0)), int(data.get("size", 0))))
    for node_id, node in layers.items():
        kind = node["type"]
        if kind not in ("MatMul", "ScaledDotProductAttention", "Convolution"):
            continue
        parents = [x for _, x in sorted(node["inputs"].items())]
        inputs = [layers[p]["outputs"][port] for p, port in parents]
        output = next(iter(node["outputs"].values()))
        record = dict(id=node_id, name=node["name"], type=kind, phase=phase(node["name"]),
                      inputs=inputs, output=output, dynamic=dynamic[node_id], attrs=node["data"])
        ops, weight_bytes = 0, 0
        if kind == "MatMul":
            a, b = inputs[:2]
            k = a[-2] if node["data"].get("transpose_a") == "true" else a[-1]
            ops = 2 * math.prod(output) * k
            weight_ids = constants[parents[1][0]]
            weight_constants = [layers[p] for p in weight_ids]
            ranges = {(int(w["data"]["offset"]), int(w["data"]["size"])) for w in weight_constants}
            weight_bytes = sum(size for _, size in ranges)
            if dynamic[node_id]:
                weight_ranges.update(ranges)
            record["constant_weight"] = not dynamic[parents[1][0]]
            record["compressed_i8_weight"] = any(
                w["data"].get("element_type") in ("i8", "u8") for w in weight_constants)
            record["weight_ranges"] = sorted(ranges)
            record["M"] = math.prod(output[:-1])
            record["N"] = output[-1]
            record["K"] = k
        elif kind == "ScaledDotProductAttention":
            q, k, v = inputs[:3]
            batches_heads = math.prod(output[:-2])
            ops = 2 * batches_heads * q[-2] * k[-2] * (q[-1] + v[-1])
        else:
            _, weights = inputs[:2]
            ops = 2 * math.prod(output) * math.prod(weights[1:])
        record["useful_matrix_ops"] = ops
        record["constant_weight_bytes"] = weight_bytes
        records.append(record)
        if dynamic[node_id]:
            p = by_phase[record["phase"]]
            p["matrix_ops"] += ops
            p["weight_reads_if_not_cached_bytes"] += weight_bytes
            p[kind + "_count"] += 1
            p[kind + "_ops"] += ops
            if record.get("compressed_i8_weight"):
                p["compressed_weight_matmul_ops"] += ops
    result = {
        "records": records,
        "by_phase": dict(by_phase),
        "dynamic_useful_matrix_ops": sum(r["useful_matrix_ops"] for r in records if r["dynamic"]),
        "unique_matmul_constant_bytes": sum(size for _, size in weight_ranges),
        "all_unique_constant_bytes": sum(size for _, size in total_constant_ranges),
        "constant_foldable_matrix_ops": sum(r["useful_matrix_ops"] for r in records if not r["dynamic"]),
        "notes": [
            "MAC counts as two operations. No padding or recomputation counted.",
            "SDPA counts QK and probability-times-V only; scalar softmax work is not matrix work.",
            "Source IR precision is not runtime precision: annotate with compiled profiling before claiming roofline.",
            "Weight-read sum assumes no cross-node cache reuse; unique bytes is a weaker compulsory bound.",
            "This inventory does not establish bandwidth, XMX throughput, or the 80 percent target.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
