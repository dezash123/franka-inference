"""Attribution probe for the c2t64 line: run ANY of the IRs on the STOCK
OpenVINO GPU plugin with the activation and arithmetic knobs exposed.

  PROBE_IR    directory holding pi05_droid_dynamic_w8a8.{xml,bin}   (required)
  PROBE_ROOT  lane root holding the <sample>.safetensors            (required)
  PROBE_TAG   output suffix, e.g. w8only / w8f32 / gptq-w8only
  PROBE_F32   set -> INFERENCE_PRECISION_HINT=f32 instead of f16
  PROBE_DQ    dynamic activation quantisation group size (default 0 = OFF)

With PROBE_DQ=0 the INT8 weights are decompressed and the activations stay in
the arithmetic precision, so the three configurations

   f16 + DQ off  |  f32 + DQ off  |  the B580 custom build (per-token INT8)

separate WEIGHT quantisation, ARITHMETIC precision and ACTIVATION quantisation
on the same graph and the same inputs.

usage: gpu_probe.py <device> <out_dir> <sample> [...]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
from safetensors.numpy import load_file

device = sys.argv[1]
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
samples = sys.argv[3:]
ir = Path(os.environ["PROBE_IR"])
root = Path(os.environ["PROBE_ROOT"])
tag = os.environ.get("PROBE_TAG", "w8only")
f32 = bool(os.environ.get("PROBE_F32"))
dq = int(os.environ.get("PROBE_DQ", "0"))

core = ov.Core()
model = core.read_model(str(ir / "pi05_droid_dynamic_w8a8.xml"))
pp = ov.preprocess.PrePostProcessor(model)
for i, port in enumerate(model.inputs):
    if port.get_element_type() == ov.Type.boolean:
        pp.input(i).tensor().set_element_type(ov.Type.u8)
    elif port.get_element_type() == ov.Type.i64:
        pp.input(i).tensor().set_element_type(ov.Type.i32)
model = pp.build()
config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": ov.properties.streams.Num(1),
          "INFERENCE_PRECISION_HINT": ov.Type.f32 if f32 else ov.Type.f16,
          "CACHE_DIR": str(root / ("openvino-cache-probe-" + tag)),
          "DYNAMIC_QUANTIZATION_GROUP_SIZE": dq}
started = time.perf_counter()
compiled = core.compile_model(model, device, config)
print("COMPILED", tag, "f32" if f32 else "f16", "dq", dq,
      round(time.perf_counter() - started, 1), flush=True)
request = compiled.create_infer_request()
for sample in samples:
    values = load_file(str(root / (sample + ".safetensors")))
    feed = {}
    for port in compiled.inputs:
        name = port.get_any_name()
        feed[name] = values[name].astype(port.get_element_type().to_dtype(), copy=False)
    result = request.infer(feed)
    value = list(result.values())[0]
    assert value.shape == (1, 15, 32) and np.isfinite(value).all()
    np.save(out / (sample + "." + tag + ".npy"), value)
    print("SAMPLE", sample, "norm", round(float(np.linalg.norm(value)), 4), flush=True)
print("DONE", flush=True)
