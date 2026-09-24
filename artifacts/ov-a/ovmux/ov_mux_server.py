"""Three-window (T=64/128/256) OpenVINO W8A8 pi0.5 DROID server.

One process, one JSON-lines chunk API on stdin/stdout, identical to
`fused --serve`: a request names a case directory, the reply carries the
device-side milliseconds and the window that served it, and the actions land
in `<dir>/actions.f32` as `[AT,32]` little-endian fp32.

Why it is a router over three worker processes and not three compiled models
in one process: the custom GPU plugin resolves `PI05_PREFIX_TOKENS` through a
function-local `static` (`pi05_shapes.hpp::pi05_prefix_tokens()`, written by
`optimization/patch_prefix_tokens.py`), so the value is latched **once per
process** and every prefix-length-gated lowering — gate/up, large-quantize,
packed RoPE, QKV split, RMS/transpose quantize, the expert K/V concat — would
silently fall back for two of the three windows.  One process per window gives
each geometry its own latch and its own shape tables.  The router adds one
pipe round trip (measured and reported separately from the device ms).

Protocol (SpeedSets' schema; matches `fused --serve`):

    -> startup                       <- {"ready":true, ...stack...}
    {"cmd":"chunk","episode":..,"dir":..}
                                     <- {"ok":true,"ms":<device ms>,
                                         "reset":false,"window":64|128|256,
                                         "text_valid":N,"router_ms":<host ms>}
    {"cmd":"reset","episode":..}     <- {"ok":true}
    {"cmd":"quit"}                   <- {"ok":true}   then exit
    errors                           <- {"ok":false,"err":"..."}

A chunk whose prompt needs more than 256 text tokens is refused
(`{"ok":false,"err":"window_overflow ..."}`); it is never truncated.

Numerics: the shipped `c2t64` configuration of
`evidence/w8a8-c2t64-20260922/RESULTS.md` §11.2 — corrected IR (int32 CumSum
export fix), custom plugin, every custom kernel family, `fuse_qkv` +
`align_vision_mlp` + `fuse_vision_padding`, and `batch_vision` OFF because the
plugin's batch-2 camera tower is defective (§9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WINDOWS = (64, 128, 256)
ACTION_HORIZON = 15
ACTION_DIM = 32


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def drm_memory(pid: int) -> dict:
    """Per-process GPU memory from the xe/i915 DRM fdinfo, in bytes."""
    totals: dict[str, int] = {}
    root = Path(f"/proc/{pid}/fdinfo")
    seen = set()
    for entry in sorted(root.glob("*")) if root.exists() else []:
        try:
            text = entry.read_text()
        except OSError:
            continue
        if "drm-client-id" not in text:
            continue
        fields = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        client = fields.get("drm-client-id")
        if client in seen:
            continue
        seen.add(client)
        for key, value in fields.items():
            if not key.startswith("drm-") or "memory" in key or key == "drm-client-id":
                continue
            parts = value.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1] in ("KiB", "MiB", "GiB", "B"):
                scale = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}[parts[1]]
                totals[key] = totals.get(key, 0) + int(parts[0]) * scale
    return totals


# --------------------------------------------------------------------------
# case directories
# --------------------------------------------------------------------------

def load_case(directory: Path):
    """(images[2,3,224,224], token ids, valid count, noise[AT,32])."""
    import numpy as np

    images = np.load(directory / "siglip.in.npy").astype(np.float32, copy=False)
    assert images.shape == (2, 3, 224, 224), ("siglip.in.npy shape", images.shape)
    noise = np.load(directory / "noise.npy").astype(np.float32, copy=False)
    assert noise.shape == (ACTION_HORIZON, ACTION_DIM), ("noise.npy shape", noise.shape)

    meta = {}
    meta_path = directory / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    tokens = None
    if (directory / "tokens.npy").exists():
        tokens = np.load(directory / "tokens.npy").reshape(-1).astype(np.int64, copy=False)
    else:
        for key in ("token_ids", "tokens", "prompt_token_ids"):
            if isinstance(meta.get(key), list):
                tokens = np.asarray(meta[key], dtype=np.int64)
                break
    assert tokens is not None, f"no token ids in {directory} (tokens.npy or meta.json:token_ids)"

    valid = None
    if (directory / "text_valid.txt").exists():
        valid = int((directory / "text_valid.txt").read_text().split()[0])
    elif (directory / "tokens_mask.npy").exists():
        valid = int(np.load(directory / "tokens_mask.npy").reshape(-1).astype(bool).sum())
    else:
        for key in ("text_valid", "token_count", "num_tokens"):
            if isinstance(meta.get(key), int):
                valid = meta[key]
                break
    assert valid is not None, f"no text_valid in {directory}"
    assert valid <= tokens.size, ("text_valid exceeds the stored token ids", valid, tokens.size)
    return images, tokens, valid, noise


# --------------------------------------------------------------------------
# worker: one window, one compiled model
# --------------------------------------------------------------------------

def worker(args) -> int:
    import numpy as np
    import openvino as ov

    sys.path.insert(0, args.opt_dir)
    from graph_fusions import align_compressed_inner, fuse_qkv

    window = args.window
    core = ov.Core(args.plugin) if args.plugin else ov.Core()
    device = args.device
    assert device in core.available_devices, (device, core.available_devices)

    started = time.perf_counter()
    model = core.read_model(args.ir)
    # batch_vision is deliberately absent: §9 of the w8a8-c2t64 evidence.
    align_report = align_compressed_inner(model, fuse_padding=True)
    qkv_report = fuse_qkv(model, ("vision", "language", "expert"))
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
        "PERF_COUNT": False,
        "INFERENCE_PRECISION_HINT": ov.Type.f16,
        # A string, as the lane passes it: the uint64 sentinel overflows the
        # pybind int converter.
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "18446744073709551615",
    }
    if args.cache_dir:
        # The OpenVINO cache key covers the model and the plugin config but not
        # the PI05_* tuning environment, which selects different kernels.  Key
        # the cache on it explicitly or a re-tuned run silently imports the
        # previous variant's binaries.
        tuning_key = hashlib.sha256(json.dumps(
            {k: v for k, v in os.environ.items() if k.startswith("PI05_")},
            sort_keys=True).encode()).hexdigest()[:12]
        config["CACHE_DIR"] = f"{args.cache_dir}-t{window}-{tuning_key}"
    compiled = core.compile_model(model, device, config)
    compile_seconds = time.perf_counter() - started

    nodes = []
    for node in compiled.get_runtime_model().get_ordered_ops():
        meta = {}
        for key, value in node.get_rt_info().items():
            try:
                meta[str(key)] = value.value
            except Exception:
                meta[str(key)] = str(value)
        nodes.append(meta)

    def custom(prefix):
        return sum(1 for m in nodes
                   if m.get("layerType") == "CustomGPUPrimitive"
                   and str(m.get("primitiveType", "")).startswith(prefix))

    int8_fc = sum(1 for m in nodes if m.get("layerType") == "FullyConnected"
                  and m.get("runtimePrecision") in ("i8", "u8"))
    kernels = {
        "int8_fc": int8_fc,
        "gateup": custom("gateup"),
        "qkv_split": custom("qkv_split"),
        "qkv_rope": custom("qkv_rope"),
        "qkv_rope_kv": custom("qkv_rope_kv"),
        "packed_rope": custom("packed_rope"),
        "rms_quantize": custom("rms_quantize"),
        "mvn_quantize": custom("mvn_quantize"),
        "adaln_quantize": custom("adaln_quantize"),
        "large_quantize": custom("large_quantize"),
        "vision_quantize": custom("vision_quantize"),
        "transpose_quantize": custom("transpose_quantize"),
        "logical_int8_matrices": int8_fc + 2 * custom("gateup"),
    }

    # Resident device I/O, exactly as the timed lane does it.
    context = core.get_default_context(device)
    request = compiled.create_infer_request()
    ports, tensors = [], []
    for port in compiled.inputs:
        tensor = context.create_tensor(port.get_element_type(), ov.Shape(port.shape),
                                       {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
        request.set_tensor(port.get_any_name(), tensor)
        ports.append(port)
        tensors.append(tensor)
    out_port = compiled.output(0)
    out_tensor = context.create_tensor(out_port.get_element_type(), ov.Shape(out_port.shape),
                                       {"SHARED_MEM_TYPE": "USM_DEVICE_BUFFER"})
    request.set_tensor(out_port.get_any_name(), out_tensor)
    host_out = ov.Tensor(out_port.get_element_type(), out_port.shape)

    # Classify the ports by shape, so a port rename (the corrected export
    # renamed 2630/4265 -> 2665/4300) cannot silently misfeed the model.
    slots = {"image": [], "img_mask": [], "tokens": None, "token_mask": None, "noise": None}
    for index, port in enumerate(ports):
        shape = list(port.shape)
        kind = port.get_element_type()
        if shape == [1, 3, 224, 224]:
            slots["image"].append(index)
        elif shape == [1]:
            slots["img_mask"].append(index)
        elif shape == [1, window] and kind in (ov.Type.i32, ov.Type.i64):
            slots["tokens"] = index
        elif shape == [1, window]:
            slots["token_mask"] = index
        elif shape == [1, ACTION_HORIZON, ACTION_DIM]:
            slots["noise"] = index
    assert len(slots["image"]) == 2 and len(slots["img_mask"]) == 2, slots
    assert None not in (slots["tokens"], slots["token_mask"], slots["noise"]), slots

    dtypes = [p.get_element_type().to_dtype() for p in ports]
    staging = [np.zeros(list(p.shape), dtype=d) for p, d in zip(ports, dtypes)]
    for index in slots["img_mask"]:
        staging[index][:] = 1

    def feed(images, tokens, valid, noise):
        staging[slots["image"][0]][0] = images[0]
        staging[slots["image"][1]][0] = images[1]
        row = staging[slots["tokens"]]
        row[:] = 0
        row[0, :valid] = tokens[:valid]
        mask = staging[slots["token_mask"]]
        mask[:] = 0
        mask[0, :valid] = 1
        staging[slots["noise"]][0] = noise
        for value, tensor in zip(staging, tensors):
            ov.Tensor(value).copy_to(tensor)

    def reply(payload):
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    reply({"ready": True, "window": window, "compile_seconds": compile_seconds,
           "kernels": kernels, "aligned_matmuls": len(align_report),
           "fused_qkv_groups": len(qkv_report),
           "openvino": ov.__version__, "device": device,
           "device_name": core.get_property(device, "FULL_DEVICE_NAME"),
           "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request_payload = json.loads(line)
            command = request_payload.get("cmd")
            if command == "quit":
                reply({"ok": True})
                break
            if command == "reset":
                reply({"ok": True})
                continue
            if command == "memory":
                reply({"ok": True, "memory": drm_memory(os.getpid())})
                continue
            if command != "chunk":
                reply({"ok": False, "err": f"unknown command {command!r}"})
                continue
            directory = Path(request_payload["dir"])
            images, tokens, valid, noise = load_case(directory)
            assert valid <= window, ("text_valid exceeds this worker's window", valid, window)
            feed(images, tokens, valid, noise)
            started_ns = time.perf_counter_ns()
            request.start_async()
            request.wait()
            device_ms = (time.perf_counter_ns() - started_ns) / 1e6
            out_tensor.copy_to(host_out)
            actions = host_out.data.reshape(ACTION_HORIZON, ACTION_DIM)
            assert np.isfinite(actions).all(), "non-finite actions"
            target = Path(request_payload.get("out") or (directory / "actions.f32"))
            target.write_bytes(np.ascontiguousarray(actions, dtype="<f4").tobytes())
            reply({"ok": True, "ms": device_ms, "reset": False, "window": window,
                   "text_valid": valid})
        except Exception as error:  # a bad case must not kill the server
            reply({"ok": False, "err": f"{type(error).__name__}: {error}"})
    return 0


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------

class Worker:
    def __init__(self, window, args, tuning_path):
        self.window = window
        self.tuning_path = Path(tuning_path)
        tuning = json.loads(self.tuning_path.read_text())
        assert all(k.startswith("PI05_") and isinstance(v, str) for k, v in tuning.items())
        expected = str(512 + window)
        assert tuning.get("PI05_PREFIX_TOKENS") == expected, (
            "tuning file is not retargeted for this window",
            window, tuning.get("PI05_PREFIX_TOKENS"), expected)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PI05_")}
        env.update(tuning)
        command = [sys.executable, os.path.abspath(__file__), "--worker",
                   "--window", str(window), "--ir", str(args.irs[window]),
                   "--device", args.device, "--opt-dir", args.opt_dir]
        if args.plugin:
            command += ["--plugin", args.plugin]
        if args.cache_dir:
            command += ["--cache-dir", args.cache_dir]
        self.process = subprocess.Popen(
            command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        self.tuning = tuning

    def call(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"window {self.window} worker died")
        return json.loads(line)

    def ready(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"window {self.window} worker failed before ready")
        payload = json.loads(line)
        assert payload.get("ready"), payload
        self.info = payload
        return payload


def router(args) -> int:
    windows = sorted(args.irs)
    workers = {}
    stack = {"role": "ov_mux_server", "windows": windows,
             "plugin_xml": args.plugin, "device": args.device,
             "clock_policy": args.clock_policy,
             "batch_vision": False,
             "rewrites": ["fuse_qkv", "align_vision_mlp", "fuse_vision_padding"],
             "server_sha256": sha256(os.path.abspath(__file__))}
    if args.plugin:
        plugin_so = Path(args.plugin).with_name("libopenvino_intel_gpu_plugin.so")
        if plugin_so.exists():
            stack["plugin_so_sha256"] = sha256(plugin_so)
    for window in windows:
        # Sequential, never concurrent: three simultaneous compiles make the GPU
        # plugin size its allocations against a device that its own siblings are
        # still claiming, which lands two of the three window's weights in GTT
        # (host memory) instead of VRAM, and wedges the build.  One at a time
        # each worker sees the true free VRAM.
        workers[window] = Worker(window, args, args.tunings[window])
        info = workers[window].ready()
        stack[f"t{window}"] = {
            "ir_xml": str(args.irs[window]),
            "ir_xml_sha256": sha256(args.irs[window]),
            "ir_bin_sha256": sha256(Path(args.irs[window]).with_suffix(".bin")),
            "tuning_json": str(args.tunings[window]),
            "tuning_sha256": sha256(args.tunings[window]),
            "prefix_tokens": 512 + window,
            "compile_seconds": info["compile_seconds"],
            "kernels": info["kernels"],
            "worker_pid": info["pid"],
        }
        stack.setdefault("openvino", info["openvino"])
        stack.setdefault("device_name", info["device_name"])
    memory = {}
    for window in windows:
        reply = workers[window].call({"cmd": "memory"})
        memory[f"t{window}"] = reply.get("memory", {})
    stack["gpu_memory_bytes"] = memory

    def emit(payload):
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    emit(dict(stack, ready=True))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        started_ns = time.perf_counter_ns()
        try:
            payload = json.loads(line)
        except Exception as error:
            emit({"ok": False, "err": f"bad request line: {error}"})
            continue
        command = payload.get("cmd")
        if command == "quit":
            for window in windows:
                try:
                    workers[window].call({"cmd": "quit"})
                except Exception:
                    pass
            emit({"ok": True})
            break
        if command == "reset":
            emit({"ok": True})
            continue
        if command == "memory":
            emit({"ok": True, "memory": {f"t{w}": workers[w].call({"cmd": "memory"}).get("memory", {})
                                         for w in windows}})
            continue
        if command != "chunk":
            emit({"ok": False, "err": f"unknown command {command!r}"})
            continue
        try:
            directory = Path(payload["dir"])
            forced = payload.get("window")
            if forced is not None:
                valid_count = None
                window = int(forced)
                assert window in workers, f"no worker for window {window}"
            else:
                valid_count = _text_valid(directory)
                window = next((w for w in windows if valid_count <= w), None)
                if window is None:
                    emit({"ok": False, "err": f"window_overflow text_valid={valid_count} "
                                              f"exceeds the largest window {windows[-1]}"})
                    continue
        except Exception as error:
            emit({"ok": False, "err": f"{type(error).__name__}: {error}"})
            continue
        reply = workers[window].call(payload)
        if reply.get("ok"):
            reply["router_ms"] = (time.perf_counter_ns() - started_ns) / 1e6
        emit(reply)
    return 0


def _text_valid(directory: Path) -> int:
    path = directory / "text_valid.txt"
    if path.exists():
        return int(path.read_text().split()[0])
    meta = json.loads((directory / "meta.json").read_text())
    for key in ("text_valid", "token_count", "num_tokens"):
        if isinstance(meta.get(key), int):
            return meta[key]
    raise AssertionError(f"no text_valid in {directory}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir64")
    ap.add_argument("--ir128")
    ap.add_argument("--ir256")
    ap.add_argument("--tuning64")
    ap.add_argument("--tuning128")
    ap.add_argument("--tuning256")
    ap.add_argument("--plugin", help="plugins.xml of the custom GPU plugin")
    ap.add_argument("--device", default=os.environ.get("OV_DEVICE", "GPU.0"))
    ap.add_argument("--opt-dir", default="/workflow/optimization",
                    help="directory holding graph_fusions.py")
    ap.add_argument("--cache-dir", help="compiled-model cache prefix; '-t<window>' is appended")
    ap.add_argument("--clock-policy", default=os.environ.get("OV_CLOCK_POLICY", "unset"),
                    help="recorded verbatim in the ready line, e.g. 'min_freq=2850'")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--window", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--ir", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    args.irs = {w: getattr(args, f"ir{w}") for w in WINDOWS if getattr(args, f"ir{w}")}
    args.tunings = {w: getattr(args, f"tuning{w}") for w in WINDOWS if getattr(args, f"ir{w}")}
    assert args.irs, "at least one of --ir64/--ir128/--ir256 is required"
    assert all(args.tunings.values()), f"every selected window needs a tuning file: {args.tunings}"
    return router(args)


if __name__ == "__main__":
    raise SystemExit(main())
