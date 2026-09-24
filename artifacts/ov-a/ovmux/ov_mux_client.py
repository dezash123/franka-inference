"""Drive `ov_mux_server.py` over its JSON-lines API: fidelity, latency, routing.

This is the lane's own harness (SpeedSets' `run_speed.py`/`run_speed_ov.py`
speaks the same protocol and produces the campaign's `speed.json`).

  fidelity  <cases dir> --window W [--refs <dir> --ref-suffix .b5.npy] --out <dir>
            one chunk per case directory, forced onto one window, actions
            saved as <out>/<case>.npy, rel-L2 against the torch references.
  bench     --case <dir> [--window W] --iters N --gap-ms G
            back-to-back or served-cadence latency of one chunk.
  route     <cases dir>
            routing histogram only, one chunk per case.

Every mode prints one JSON object and writes it to --json if given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def rel_l2(a, b, dims=None) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(15, 32)
    b = np.asarray(b, dtype=np.float64).reshape(15, 32)
    if dims is not None:
        a, b = a[:, :dims], b[:, :dims]
    norm = np.linalg.norm(b)
    delta = np.linalg.norm(a - b)
    return float(delta / norm) if norm > 0 else (1.0 if delta > 0 else 0.0)


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


class Server:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("server exited before the ready line")
        self.ready = json.loads(line)
        assert self.ready.get("ready"), self.ready

    def call(self, payload):
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("server died")
        return json.loads(line)

    def close(self):
        try:
            self.call({"cmd": "quit"})
        except Exception:
            pass
        self.process.wait(timeout=60)


def cases_of(root: Path, pattern: str):
    return sorted(d for d in root.glob(pattern) if (d / "siglip.in.npy").exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fidelity", "bench", "route"))
    ap.add_argument("cases", nargs="?")
    ap.add_argument("--server-cmd", required=True, help="full ov_mux_server.py command line")
    ap.add_argument("--pattern", default="s[0-9][0-9]")
    ap.add_argument("--window", type=int)
    ap.add_argument("--refs", help="directory of torch reference .npy")
    ap.add_argument("--ref-prefix", default="accA-")
    ap.add_argument("--ref-suffix", default=".b5.npy")
    ap.add_argument("--compare-to", help="directory of another window's saved actions")
    ap.add_argument("--out", help="directory for the produced actions")
    ap.add_argument("--case", help="bench: the single case directory")
    ap.add_argument("--actions-dir", help="write actions here instead of into the case dir")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--gap-ms", type=float, default=0.0)
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args()

    server = Server(args.server_cmd.split())
    report = {"stack": server.ready, "mode": args.mode, "arguments": vars(args)}
    try:
        if args.mode == "bench":
            directory = Path(args.case)
            payload = {"cmd": "chunk", "episode": "bench", "dir": str(directory)}
            if args.window:
                payload["window"] = args.window
            for _ in range(args.warmups):
                reply = server.call(payload)
                assert reply.get("ok"), reply
            device, wall, router = [], [], []
            for _ in range(args.iters):
                if args.gap_ms:
                    time.sleep(args.gap_ms / 1000.0)
                started = time.perf_counter_ns()
                reply = server.call(payload)
                wall.append((time.perf_counter_ns() - started) / 1e6)
                assert reply.get("ok"), reply
                device.append(reply["ms"])
                router.append(reply["router_ms"] - reply["ms"])
                window = reply["window"]
            report.update(window=window, gap_ms=args.gap_ms,
                          device_ms=stats(device), wall_ms=stats(wall),
                          router_overhead_ms=stats(router),
                          device_samples_ms=device)
        else:
            root = Path(args.cases)
            directories = cases_of(root, args.pattern)
            assert directories, f"no case directories under {root}/{args.pattern}"
            out = Path(args.out) if args.out else None
            if out:
                out.mkdir(parents=True, exist_ok=True)
            scratch = Path(args.actions_dir) if args.actions_dir else None
            if scratch:
                scratch.mkdir(parents=True, exist_ok=True)
            rows, windows, device = {}, {}, []
            for directory in directories:
                name = str(directory.relative_to(root)).replace("/", "_")
                payload = {"cmd": "chunk", "episode": name, "dir": str(directory)}
                if args.window:
                    payload["window"] = args.window
                # A read-only case set (a mirrored speed set) cannot take the
                # actions beside its inputs.
                target = (scratch / f"{name}.f32") if scratch else (directory / "actions.f32")
                if scratch:
                    payload["out"] = str(target)
                reply = server.call(payload)
                assert reply.get("ok"), (name, reply)
                windows[reply["window"]] = windows.get(reply["window"], 0) + 1
                device.append(reply["ms"])
                actions = np.frombuffer(target.read_bytes(), dtype="<f4").reshape(15, 32)
                if out:
                    np.save(out / f"{name}.npy", actions)
                row = {"window": reply["window"], "text_valid": reply["text_valid"],
                       "ms": reply["ms"]}
                if args.refs:
                    reference = np.load(
                        Path(args.refs) / f"{args.ref_prefix}{name}{args.ref_suffix}")
                    row["rel_l2"] = rel_l2(actions, reference)
                    row["rel_l2_task"] = rel_l2(actions, reference, 8)
                if args.compare_to:
                    other = np.load(Path(args.compare_to) / f"{name}.npy")
                    row["vs_other_rel_l2"] = rel_l2(actions, other)
                    row["vs_other_rel_l2_task"] = rel_l2(actions, other, 8)
                    row["vs_other_bitwise"] = bool(np.array_equal(actions, other))
                rows[name] = row
            report["windows"] = windows
            report["per_case"] = rows
            report["device_ms"] = stats(device)
            for key in ("rel_l2", "rel_l2_task", "vs_other_rel_l2", "vs_other_rel_l2_task"):
                values = [r[key] for r in rows.values() if key in r]
                if values:
                    worst = max(rows, key=lambda k: rows[k].get(key, -1))
                    report[key] = {"mean": float(np.mean(values)),
                                   "median": float(np.median(values)),
                                   "worst": float(max(values)), "worst_case": worst}
            if any("vs_other_bitwise" in r for r in rows.values()):
                report["vs_other_bitwise_all"] = all(r["vs_other_bitwise"] for r in rows.values())
    finally:
        server.close()

    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
