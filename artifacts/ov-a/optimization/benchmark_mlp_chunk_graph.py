"""Screen dependency-driven overlap of complete checkpoint prefix MLP chunks.

All chunks form one graph with shared original weights and complete outputs.
This is a layer probe, not complete Pi0.5 validation or latency.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import openvino as ov
from openvino import opset13 as op

from profile_openvino import runtime_nodes, stats

ap = argparse.ArgumentParser()
ap.add_argument("--plugins", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--iterations", type=int, default=50)
ap.add_argument("--profile", type=int, default=0)
ap.add_argument("--trace-region", action="store_true")
ap.add_argument("--out-of-order", type=int, choices=(0, 1), default=0)
ap.add_argument("--reference-dir", help="Independent previously validated mode output folder")
ap.add_argument("--downstream-rms", action="store_true",
                help="Keep residual sums internal by consuming them in row RMS normalization")
args = ap.parse_args()
assert args.chunks[0] == 1 and set(args.chunks) <= {1, 2, 4, 8}
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
lock = open("/workflow/benchmark.lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)
for k in list(os.environ):
    if k.startswith("PI05_"):
        del os.environ[k]
settings = json.loads(Path("/workflow/optimization/r5-vision-quant-w128-c1.json").read_text())
os.environ.update(settings)
os.environ.update(PI05_PREFIX_MLP_CHUNKS="1", PI05_DQ_REUSE="1",
                  PI05_OUT_OF_ORDER=str(args.out_of_order))
# Match the selected down-projection geometry for every token count; retain
# the exact INT32 dot product and the native dequantization/residual epilogue.
os.environ["PI05_GEMM_TABLE"] = ";".join(
    f"2048,{968 // count},16384,-1,32,32|wg 4x8" for count in args.chunks)
device = os.environ.get("OV_DEVICE", "GPU.0")
core = ov.Core(args.plugins)
full = core.read_model("/workflow/model/pi05_droid_dynamic_w8a8.xml")
ops = full.get_ordered_ops()
weights = {}
for name in ("gate", "up", "down"):
    fc = next(n for n in ops if n.get_type_name() == "MatMul" and
              f"language_model.layers.0.mlp.{name}_proj/" in n.friendly_name)
    weights[name] = fc.input_value(1)
dtype = fc.get_output_element_type(0)
ctx = core.get_default_context(device)
compiled_cases = {}
report = {"arguments": vars(args), "device": device,
          "total_tokens": 968, "tuning_environment": {
              k: v for k, v in os.environ.items() if k.startswith("PI05_")},
          "validation": [], "trials": [], "profile": {},
          "note": "One dependency graph per complete prefix MLP; original checkpoint weights and all rows included."}


def save():
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


for count in args.chunks:
    rows = 968 // count
    shape = [1, 968, 2048]
    x = op.parameter(shape, dtype, name="input")
    residual = op.parameter(shape, dtype, name="residual")
    if count == 1:
        x_parts, r_parts = [x.output(0)], [residual.output(0)]
    else:
        axis = op.constant(np.int64(1))
        x_parts = list(op.split(x, axis, count).outputs())
        r_parts = list(op.split(residual, axis, count).outputs())
    pieces = []
    for chunk, (part, resid) in enumerate(zip(x_parts, r_parts)):
        gate = op.matmul(part, weights["gate"], False, True, name=f"prefix_gate_{chunk}")
        up = op.matmul(part, weights["up"], False, True, name=f"prefix_up_{chunk}")
        product = op.multiply(op.gelu(gate, "tanh"), up, name=f"prefix_product_{chunk}")
        down = op.matmul(product, weights["down"], False, True, name=f"prefix_down_{chunk}")
        piece = op.add(down, resid)
        if args.downstream_rms:
            # Keep each residual internal with identical consumers. Normalizing
            # after Concat made down emit FP16 for chunks but FP32 for the control.
            # This probe deliberately measures MLP + row RMS with an FP32
            # residual boundary; full-model integration needs its own checks.
            piece = op.convert(piece, ov.Type.f32)
            square = op.multiply(piece, piece)
            mean = op.reduce_mean(square, op.constant(np.int64(-1)), True)
            denom = op.sqrt(op.add(mean, op.constant(np.float32(1e-6))))
            piece = op.divide(piece, denom)
        pieces.append(piece)
    y = pieces[0] if count == 1 else op.concat(pieces, 1)
    model = ov.Model([y], [x, residual])
    pp = ov.preprocess.PrePostProcessor(model)
    for i in range(2):
        pp.input(i).tensor().set_element_type(ov.Type.f16)
    # Full-model prefix down+residual outputs are FP16. Match that boundary
    # for every graph, including the unsplit control.
    output_type = ov.Type.f32 if args.downstream_rms else ov.Type.f16
    pp.output().tensor().set_element_type(output_type)
    model = pp.build()
    print("COMPILE_START", count, time.time(), flush=True)
    compiled = core.compile_model(model, device, {
        "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": ov.properties.streams.Num(1),
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615",
        "PERF_COUNT": bool(args.profile)})
    nodes = runtime_nodes(compiled)
    (out / f"runtime-{count}.json").write_text(json.dumps(nodes, indent=2))
    custom = [n for n in nodes if n["metadata"].get("layerType") == "CustomGPUPrimitive"]
    assert len(custom) == 2 * count, [(n["name"], n["metadata"]) for n in custom]
    matrices = [n for n in nodes if n["metadata"].get("layerType") == "FullyConnected"]
    assert len(matrices) == count, matrices
    assert all(n["metadata"].get("runtimePrecision") == "i8" and
               n["metadata"].get("primitiveType", "").startswith("jit:gemm")
               for n in matrices), matrices
    if args.downstream_rms:
        assert all(n["metadata"].get("outputPrecisions") == "f32"
                   for n in matrices), "Inconsistent residual precision boundary"
    requests = []
    for chunk in range(1):
        req = compiled.create_infer_request()
        inputs = []
        for port in compiled.inputs:
            tensor = ctx.create_tensor(ov.Type.f16, port.shape,
                                       {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
            req.set_tensor(port.get_any_name(), tensor)
            inputs.append(tensor)
        dst = ctx.create_tensor(output_type, compiled.output().shape,
                               {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
        req.set_tensor(compiled.output().get_any_name(), dst)
        host = ov.Tensor(output_type, compiled.output().shape)
        requests.append((req, inputs, dst, host))
    compiled_cases[count] = (compiled, requests)
    print("COMPILE_DONE", count, time.time(), flush=True)


def set_inputs(count, x, residual):
    for _, inputs, _, _ in compiled_cases[count][1]:
        for tensor, value in zip(inputs, (x, residual)):
            ov.Tensor(np.ascontiguousarray(value)).copy_to(tensor)


def infer(count):
    for req, _, _, _ in compiled_cases[count][1]:
        req.start_async()
        req.wait()


def read_output(count):
    parts = []
    for _, _, dst, host in compiled_cases[count][1]:
        dst.copy_to(host)
        parts.append(host.data.copy())
    return np.concatenate(parts, axis=1)


rng = np.random.default_rng(9682048)
x = rng.normal(0, 0.7, (1, 968, 2048)).astype(np.float16)
residual = rng.normal(0, 0.1, x.shape).astype(np.float16)
cases = [("normal", x), ("negative", -x), ("zero", np.zeros_like(x)),
         ("tiny", (x * np.float16(0.0001)).astype(np.float16)),
         ("larger", (x * np.float16(4)).astype(np.float16)),
         ("repeat-normal", x)]
last_expected = None
for name, value in cases:
    for count in args.chunks:
        set_inputs(count, value, residual)
        infer(count)
        actual = read_output(count)
        assert np.isfinite(actual).all(), (name, count)
        for _, tensors, _, _ in compiled_cases[count][1]:
            for input_name, tensor, expected_input in zip(("input", "residual"), tensors, (value, residual)):
                copied = ov.Tensor(ov.Type.f16, tensor.get_shape())
                tensor.copy_to(copied)
                np.testing.assert_array_equal(copied.data, expected_input,
                    err_msg=f"input mutated: {input_name}, chunks={count}")
        if count == 1:
            expected = actual
            if args.reference_dir:
                np.testing.assert_array_equal(
                    expected, np.load(Path(args.reference_dir) / f"{name}.npy"),
                    err_msg=f"{name} differs between queue modes")
            np.save(out / f"{name}.npy", expected)
            if name == "normal":
                first_expected = expected.copy()
            if name == "repeat-normal":
                np.testing.assert_array_equal(expected, first_expected)
        else:
            if not np.array_equal(actual, expected):
                np.save(out / f"{name}-chunks{count}-failed.npy", actual)
                report["failure"] = {"case": name, "chunks": count,
                    "rows": [{"chunk": j,
                        "max_abs": float(np.max(np.abs(
                            actual[:, j * (968 // count):(j+1) * (968 // count)].astype(np.float32) -
                            expected[:, j * (968 // count):(j+1) * (968 // count)].astype(np.float32))))}
                        for j in range(count)]}
                save()
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name} chunks={count}")
        report["validation"].append({"case": name, "chunks": count, "bitwise": True})
    last_expected = expected.copy()
    save()
print("VALIDATION_PASSED", len(report["validation"]), flush=True)

orders = [args.chunks, list(reversed(args.chunks)), args.chunks]
for repeat, order in enumerate(orders):
    for count in order:
        for _ in range(10):
            infer(count)
        if args.trace_region:
            os.environ["PTI_ENABLE_COLLECTION"] = "1"
        times = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            infer(count)
            times.append((time.perf_counter_ns() - started) / 1e6)
        if args.trace_region:
            os.environ["PTI_ENABLE_COLLECTION"] = "0"
        np.testing.assert_array_equal(read_output(count), last_expected)
        row = {"repeat": repeat, "chunks": count, "timing": stats(times)}
        report["trials"].append(row)
        if args.profile:
            report["profile"][str(count)] = [
                {"chunk": chunk, "name": p.node_name, "implementation": p.exec_type,
                 "us": p.real_time.total_seconds() * 1e6, "status": str(p.status)}
                for chunk, (req, _, _, _) in enumerate(compiled_cases[count][1])
                for p in req.get_profiling_info()]
        save()
        print(json.dumps({"repeat": repeat, "chunks": count,
                          "median_ms": row["timing"]["median_ms"],
                          "profiled": bool(args.profile)}), flush=True)
