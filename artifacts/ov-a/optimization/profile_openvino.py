"""Resident-I/O full-workload benchmark and detailed OpenVINO primitive profile.

Run on the dstack B580 runner. Profiles are diagnostic; use --profile 0 for
authoritative latency. Never compare profiler-on timings to profiler-off timings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import openvino as ov
from safetensors.numpy import load_file


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def stats(values):
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "samples_ms": values,
    }


def comparison(a, b):
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return {
        "max_abs": float(np.max(np.abs(x-y))),
        "rmse": float(np.sqrt(np.mean((x-y)**2))),
        "cosine": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))),
    }


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


def runtime_nodes(compiled):
    nodes = []
    for op in compiled.get_runtime_model().get_ordered_ops():
        metadata = {}
        for k, v in op.get_rt_info().items():
            try:
                metadata[str(k)] = v.value
            except Exception:
                metadata[str(k)] = str(v)
        nodes.append({
            "name": op.get_friendly_name(),
            "type": op.get_type_name(),
            "inputs": [
                {"shape": list(x.get_shape()), "dtype": str(x.get_element_type()),
                 "source": x.get_source_output().get_node().get_friendly_name()}
                for x in op.inputs()
            ],
            "outputs": [
                {"shape": list(x.get_shape()), "dtype": str(x.get_element_type())}
                for x in op.outputs()
            ],
            "metadata": metadata,
        })
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workflow/model/pi05_droid_dynamic_w8a8.xml")
    ap.add_argument("--root", default="/workflow")
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--profile-snapshots", choices=("final", "all"), default="final",
                    help="Final counters avoid a long host-side counter read between every inference")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--config", help="JSON object of additional plugin properties")
    ap.add_argument("--fuse-qkv", help="Comma-separated scopes: vision,language,expert")
    ap.add_argument("--batch-vision", action="store_true")
    ap.add_argument("--plugins", help="Plugin registry XML, for an isolated custom GPU plugin")
    ap.add_argument("--align-vision-mlp", action="store_true")
    ap.add_argument("--fuse-vision-padding", action="store_true")
    ap.add_argument("--cache-tag", default="stock")
    ap.add_argument("--validation-samples", default="zero,random")
    ap.add_argument("--validation-repeats", type=int, default=1,
                    help="Additional shuffled passes through all cases before timing, for execution-order stress")
    ap.add_argument("--trace-region", action="store_true",
                    help="Enable PTI collection only around measured iterations")
    ap.add_argument("--require-bitwise", help="Reference backend that must match every validation output exactly")
    ap.add_argument("--require-float-gate", help="Envelope JSON: every output must stay within the accepted W8A8 error versus the FP32 eager reference")
    ap.add_argument("--require-gateup-count", type=int, help="Require this many experimental fused INT8 gate/up nodes")
    ap.add_argument("--require-qkv-split-count", type=int)
    ap.add_argument("--require-packed-rope-count", type=int)
    ap.add_argument("--require-qkv-rope-count", type=int)
    ap.add_argument("--require-qkv-rope-kv-count", type=int)
    ap.add_argument("--require-rms-quantize-count", type=int)
    ap.add_argument("--require-mvn-quantize-count", type=int)
    ap.add_argument("--require-adaln-quantize-count", type=int)
    ap.add_argument("--require-large-quantize-count", type=int)
    ap.add_argument("--require-vision-quantize-count", type=int)
    ap.add_argument("--require-transpose-quantize-count", type=int)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (root / "benchmark.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    core = ov.Core(args.plugins) if args.plugins else ov.Core()
    device = os.environ.get("OV_DEVICE") or next(
        d for d in core.available_devices
        if d.startswith("GPU") and "B580" in core.get_property(d, "FULL_DEVICE_NAME"))
    assert device in core.available_devices and "B580" in core.get_property(device, "FULL_DEVICE_NAME")
    properties = {}
    for key in core.get_property(device, "SUPPORTED_PROPERTIES"):
        try:
            properties[str(key)] = str(core.get_property(device, str(key)))
        except Exception as error:
            properties[str(key)] = type(error).__name__
    write(out / "device-properties.json", properties)
    print("READ_MODEL", time.time(), flush=True)
    model = core.read_model(args.model)
    if args.batch_vision:
        from graph_fusions import batch_vision
        batch_report = batch_vision(model)
        write(out / "vision-batching-report.json", batch_report)
        print("VISION_BATCHED", json.dumps(batch_report), flush=True)
    if args.align_vision_mlp:
        from graph_fusions import align_compressed_inner
        alignment_report = align_compressed_inner(model, fuse_padding=args.fuse_vision_padding)
        write(out / "alignment-report.json", alignment_report)
        print("ALIGNED_MATMULS", len(alignment_report), flush=True)
    if args.fuse_qkv:
        from graph_fusions import fuse_qkv
        fusion_report = fuse_qkv(model, tuple(args.fuse_qkv.split(",")))
        write(out / "fusion-report.json", fusion_report)
        print("QKV_FUSED_GROUPS", len(fusion_report), flush=True)
    model.output(0).get_tensor().set_names({"actions"})
    pp = ov.preprocess.PrePostProcessor(model)
    for i, port in enumerate(model.inputs):
        if port.get_element_type() == ov.Type.boolean:
            pp.input(i).tensor().set_element_type(ov.Type.u8)
        elif port.get_element_type() == ov.Type.i64:
            pp.input(i).tensor().set_element_type(ov.Type.i32)
    model = pp.build()
    config = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": ov.properties.streams.Num(1),
        "PERF_COUNT": bool(args.profile),
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        # A cache entry compiled without profiling loses oneDNN event counters
        # when later imported with PERF_COUNT=True. Keep the caches separate.
        "CACHE_DIR": str(root / (
            f"openvino-cache-{args.cache_tag}-profiled" if args.profile
            else f"openvino-cache-{args.cache_tag}")),
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": args.group_size,
    }
    if args.config:
        config.update(json.loads(args.config))
    write(out / "config.json", config)
    started = time.perf_counter()
    print("COMPILE_START", time.time(), flush=True)
    compiled = core.compile_model(model, device, config)
    compile_seconds = time.perf_counter() - started
    print("COMPILED", compile_seconds, flush=True)
    nodes = runtime_nodes(compiled)
    write(out / "runtime-nodes.json", nodes)
    fused_gateups = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                    and n["metadata"].get("primitiveType", "").startswith("gateup")]
    if args.require_gateup_count is not None:
        assert len(fused_gateups) == args.require_gateup_count, ("Missing fused gate/up nodes", len(fused_gateups))
    fused_qkv_splits = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                       and n["metadata"].get("primitiveType", "").startswith("qkv_split")]
    if args.require_qkv_split_count is not None:
        assert len(fused_qkv_splits) == args.require_qkv_split_count, ("Missing fused QKV splits", len(fused_qkv_splits))
    packed_ropes = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                    and n["metadata"].get("primitiveType", "").startswith("packed_rope")]
    if args.require_packed_rope_count is not None:
        assert len(packed_ropes) == args.require_packed_rope_count, ("Missing packed RoPE nodes", len(packed_ropes))
    qkv_ropes = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                and n["metadata"].get("primitiveType", "").startswith("qkv_rope")]
    if args.require_qkv_rope_count is not None:
        assert len(qkv_ropes) == args.require_qkv_rope_count, ("Missing combined QKV/RoPE nodes", len(qkv_ropes))
    qkv_rope_kvs = [n for n in qkv_ropes
                    if n["metadata"].get("primitiveType", "").startswith("qkv_rope_kv")]
    if args.require_qkv_rope_kv_count is not None:
        assert len(qkv_rope_kvs) == args.require_qkv_rope_kv_count, (
            "Missing combined QKV/RoPE/KV nodes", len(qkv_rope_kvs))
        remaining = [n["name"] for n in nodes
                     if n["metadata"].get("layerType") == "Concat"
                     and "gemma_expert" in n["name"]
                     and any(o["shape"] == [1, 1, 983, 256] for o in n["outputs"])]
        assert not remaining, ("Expert K/V concatenations remain", remaining)
    fused_rms_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                           and n["metadata"].get("primitiveType", "").startswith("rms_quantize")]
    fused_mvn_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                           and n["metadata"].get("primitiveType", "").startswith("mvn_quantize")]
    if args.require_mvn_quantize_count is not None:
        assert len(fused_mvn_quantizers) == args.require_mvn_quantize_count, (
            "Missing fused MVN/quantizer nodes", len(fused_mvn_quantizers))
    fused_adaln_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                             and n["metadata"].get("primitiveType", "").startswith("adaln_quantize")]
    if args.require_adaln_quantize_count is not None:
        assert len(fused_adaln_quantizers) == args.require_adaln_quantize_count, (
            "Missing fused adaptive RMS/quantizer nodes", len(fused_adaln_quantizers))
    large_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                        and n["metadata"].get("primitiveType", "").startswith("large_quantize")]
    vision_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                         and n["metadata"].get("primitiveType", "").startswith("vision_quantize")]
    transpose_quantizers = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"
                            and n["metadata"].get("primitiveType", "").startswith("transpose_quantize")]
    if args.require_large_quantize_count is not None:
        assert len(large_quantizers) == args.require_large_quantize_count, (
            "Missing large quantizers", len(large_quantizers))
    if args.require_vision_quantize_count is not None:
        assert len(vision_quantizers) == args.require_vision_quantize_count, (
            "Missing vision quantizers", len(vision_quantizers))
    if args.require_transpose_quantize_count is not None:
        assert len(transpose_quantizers) == args.require_transpose_quantize_count, (
            "Missing transpose quantizers", len(transpose_quantizers))
    if args.require_rms_quantize_count is not None:
        assert len(fused_rms_quantizers) == args.require_rms_quantize_count, ("Missing fused RMS/quantizers", len(fused_rms_quantizers))
    int8_fc = [n for n in nodes
               if n["metadata"].get("layerType") == "FullyConnected"
               and n["metadata"].get("runtimePrecision") in ("i8", "u8")]
    assert int8_fc, "No confirmed W8A8 FullyConnected nodes"
    if os.environ.get("PI05_OUT_OF_ORDER", "0") != "0":
        assert all(n["metadata"].get("primitiveType", "").startswith("jit:gemm")
                   for n in int8_fc), "Out-of-order mode must retain native XMX INT8 matrices"
    context = core.get_default_context(device)
    request = compiled.create_infer_request()
    inputs = []
    for port in compiled.inputs:
        tensor = context.create_tensor(port.get_element_type(), ov.Shape(port.shape),
                                       {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
        request.set_tensor(port.get_any_name(), tensor)
        inputs.append(tensor)
    port = compiled.output(0)
    output = context.create_tensor(port.get_element_type(), ov.Shape(port.shape),
                                   {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
    request.set_tensor(port.get_any_name(), output)
    host = ov.Tensor(port.get_element_type(), port.shape)

    def upload(sample):
        values = load_file(str(root / f"{sample}.safetensors"))
        for i, (port, tensor) in enumerate(zip(compiled.inputs, inputs, strict=True)):
            name = port.get_any_name()
            value = values[name] if name in values else values[f"input.{1236+i}"]
            ov.Tensor(value.astype(port.get_element_type().to_dtype(), copy=False)).copy_to(tensor)

    def infer():
        request.start_async()
        request.wait()

    outputs, checks = {}, {}
    for sample in args.validation_samples.split(","):
        upload(sample)
        infer()
        output.copy_to(host)
        value = host.data.copy()
        assert value.shape == (1, 15, 32) and np.isfinite(value).all()
        outputs[sample] = value
        np.save(out / f"{sample}.npy", value)
        checks[sample] = {}
        for backend in ("torch-eager", "openvino-dynamic-w8a8", "custom-per-token-fusedpad", "custom-corrected"):
            path = root / "reference" / f"{backend}-{sample}.npy"
            if path.exists():
                checks[sample][backend] = comparison(value, np.load(path))
        print("CORRECTNESS", sample, checks[sample], flush=True)
        if args.require_bitwise:
            path = root / "reference" / f"{args.require_bitwise}-{sample}.npy"
            assert path.exists(), f"Missing required reference: {path}"
            assert np.array_equal(value, np.load(path)), f"Bitwise validation failed: {sample}"
        if args.require_float_gate:
            envelope = json.loads(Path(args.require_float_gate).read_text())["samples"][sample]
            path = root / "reference" / f"torch-eager-fp32-{sample}.npy"
            assert path.exists(), f"Missing FP32 reference: {path}"
            gate = comparison(value, np.load(path))
            checks[sample]["torch-eager-fp32"] = gate
            assert gate["rmse"] <= envelope["rmse_max"] and gate["cosine"] >= envelope["cosine_min"], \
                f"Float gate failed: {sample} {gate} envelope {envelope}"
    assert not np.array_equal(outputs["zero"], outputs["random"])
    assert args.validation_repeats >= 1
    rng = np.random.default_rng(917062)
    for repeat in range(1, args.validation_repeats):
        for sample in rng.permutation(list(outputs)):
            upload(str(sample))
            infer()
            output.copy_to(host)
            assert np.array_equal(host.data, outputs[sample]), (
                "Changed/repeated-input ordering failure", repeat, sample)
    if args.validation_repeats > 1:
        print("SHUFFLED_VALIDATION_PASSED", args.validation_repeats * len(outputs), flush=True)
    upload("zero")
    for _ in range(args.warmups):
        infer()
    samples, profiles, enqueue_samples = [], [], []
    if args.trace_region:
        os.environ["PTI_ENABLE_COLLECTION"] = "1"
    gap_ms = float(os.environ.get("LANE_GAP_MS", "0") or 0)
    for iteration in range(args.iterations):
        if gap_ms:
            time.sleep(gap_ms / 1000.0)
        started = time.perf_counter_ns()
        request.start_async()
        enqueued = time.perf_counter_ns()
        request.wait()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        enqueue_samples.append((enqueued - started) / 1e6)
        if args.profile and args.profile_snapshots == "all":
            profiles.append([
                {"name": p.node_name, "type": p.node_type, "exec_type": p.exec_type,
                 "status": str(p.status), "real_us": p.real_time.total_seconds() * 1e6,
                 "cpu_us": p.cpu_time.total_seconds() * 1e6}
                for p in request.get_profiling_info()
            ])
    if args.trace_region:
        os.environ["PTI_ENABLE_COLLECTION"] = "0"
    if args.profile and args.profile_snapshots == "final":
        profiles.append([
            {"name": p.node_name, "type": p.node_type, "exec_type": p.exec_type,
             "status": str(p.status), "real_us": p.real_time.total_seconds() * 1e6,
             "cpu_us": p.cpu_time.total_seconds() * 1e6}
            for p in request.get_profiling_info()
        ])
    if args.profile and os.environ.get("PI05_VISION_QKV_VIEWS"):
        vision_splits = [p for p in profiles[-1] if "vision_model" in p["name"]
                         and "qkv_fused" in p["name"] and "/split.out" in p["name"]]
        assert len(vision_splits) == 81, ("Missing vision QKV views", len(vision_splits))
        assert all(p["status"] == "Status.OPTIMIZED_OUT" for p in vision_splits), \
            ("Vision QKV copies still execute", vision_splits)
    output.copy_to(host)
    assert np.array_equal(host.data, outputs["zero"]), "Repeated inference changed output for identical input"
    write(out / "profile-samples.json", profiles)
    if profiles:
        # Plugin counters may be running averages. Preserve every raw snapshot;
        # the final snapshot is the primary attribution, not independent samples.
        node_map = {n["name"]: n for n in nodes}
        by_type, by_phase, by_impl = defaultdict(float), defaultdict(float), defaultdict(float)
        rows = []
        for p in profiles[-1]:
            node = node_map.get(p["name"], {})
            original = node.get("metadata", {}).get("originalLayersNames", p["name"])
            node_phase = phase(original)
            if node_phase == "other" and p["type"] == "DynamicQuantize" and node.get("inputs"):
                node_phase = phase(node["inputs"][0]["source"])
            row = dict(p, phase=node_phase, inputs=node.get("inputs"),
                       outputs=node.get("outputs"), metadata=node.get("metadata"))
            rows.append(row)
            by_type[p["type"]] += p["real_us"]
            by_phase[row["phase"]] += p["real_us"]
            by_impl[p["exec_type"]] += p["real_us"]
        rows.sort(key=lambda r: r["real_us"], reverse=True)
        write(out / "profile-ranked.json", rows)
        summary = {
            "sum_device_us": sum(p["real_us"] for p in profiles[-1]),
            "by_type_us": dict(sorted(by_type.items(), key=lambda p: -p[1])),
            "by_phase_us": dict(sorted(by_phase.items(), key=lambda p: -p[1])),
            "by_impl_us": dict(sorted(by_impl.items(), key=lambda p: -p[1])),
            "executed_primitive_count": sum(p["status"] == "Status.EXECUTED" for p in profiles[-1]),
            "counter_note": "Final plugin counter snapshot; sums are not elapsed time when kernels overlap.",
        }
        write(out / "profile-summary.json", summary)
        print("PROFILE_SUMMARY", json.dumps(summary), flush=True)
        print("TOP_PRIMITIVES", json.dumps(rows[:15]), flush=True)
    result = {
        "tuning_environment": {k: v for k, v in os.environ.items() if k.startswith("PI05_")},
        "timing": stats(samples), "host_enqueue": stats(enqueue_samples), "profiling_enabled": bool(args.profile),
        "compile_seconds": compile_seconds, "int8_fc_count": len(int8_fc),
        "fused_int8_gateup_count": len(fused_gateups),
        "fused_qkv_split_count": len(fused_qkv_splits),
        "packed_rope_count": len(packed_ropes),
        "qkv_rope_count": len(qkv_ropes),
        "qkv_rope_kv_count": len(qkv_rope_kvs),
        "fused_rms_quantize_count": len(fused_rms_quantizers),
        "fused_mvn_quantize_count": len(fused_mvn_quantizers),
        "fused_adaln_quantize_count": len(fused_adaln_quantizers),
        "large_quantize_count": len(large_quantizers),
        "vision_quantize_count": len(vision_quantizers),
        "fused_transpose_quantize_count": len(transpose_quantizers),
        "logical_int8_matrix_count": len(int8_fc) + 2*len(fused_gateups),
        "sanity_checks": checks, "changed_input_check": True, "repeated_input_bitwise_stable": True,
        "validation_repeats": args.validation_repeats,
        "version": ov.__version__, "config": config, "resident_device_io": True,
        "warmups": args.warmups, "iterations": args.iterations, "flow_steps": 5,
        "arguments": vars(args),
    }
    write(out / "result.json", result)
    print("RESULT", json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
