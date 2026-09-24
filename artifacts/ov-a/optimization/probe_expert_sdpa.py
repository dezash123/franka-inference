"""Standalone check and timing of expert_sdpa.cl on the real device."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyopencl as cl

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=str(Path(__file__).with_name("expert_sdpa.cl")))
ap.add_argument("--prefetch-mb", type=float, default=0.0)
ap.add_argument("--prefetch-groups", type=int, default=12)
ap.add_argument("--grf", type=int, default=256)
ap.add_argument("--iterations", type=int, default=200)
ap.add_argument("--extra-opts", default="")
ap.add_argument("--inputs", help="optional safetensors with q,k,v,mask")
args = ap.parse_args()

HEADS, TOKENS, KEYS, HEAD_DIM = 8, 15, 983, 256
SCALE = HEAD_DIM ** -0.5
ATTN_GROUPS = (TOKENS + 1) // 2

ctx = cl.Context(dev_type=cl.device_type.GPU)
dev = ctx.devices[int(os.environ.get("PI05_DEVICE_INDEX", "0"))]
q = cl.CommandQueue(ctx, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)
rng = np.random.default_rng(1234)
if args.inputs:
    from safetensors.numpy import load_file
    t = load_file(args.inputs)
    Q, K, V, mask = t["q"], t["k"], t["v"], t["mask"]
else:
    Q = rng.standard_normal((TOKENS, HEADS * HEAD_DIM)).astype(np.float16)
    K = rng.standard_normal((KEYS, HEAD_DIM)).astype(np.float16)
    V = rng.standard_normal((KEYS, HEAD_DIM)).astype(np.float16)
    mask = np.zeros((TOKENS, KEYS), np.float32)
    mask[:, 900:968] = -np.inf
    for t in range(TOKENS):
        mask[t, 968 + t + 1:] = -np.inf
    mask = mask.astype(np.float16)

Qf = Q.astype(np.float32).reshape(TOKENS, HEADS, HEAD_DIM)
S = np.einsum("thd,kd->thk", Qf, K.astype(np.float32)) * SCALE + mask.astype(np.float32)[:, None, :]
S = S - S.max(-1, keepdims=True)
P = np.exp(S); P /= P.sum(-1, keepdims=True)
O_ref = np.einsum("thk,kd->htd", P, V.astype(np.float32))

opts = (f"-cl-std=CL3.0 -cl-mad-enable -cl-intel-{args.grf}-GRF-per-thread "
        f"-DHEADS={HEADS} -DTOKENS={TOKENS} -DKEYS={KEYS} -DHEAD_DIM={HEAD_DIM} -DSCALE={SCALE!r}f "
        + args.extra_opts)
prefetch_groups = args.prefetch_groups if args.prefetch_mb > 0 else 0
if prefetch_groups:
    opts += f" -DPREFETCH_GROUPS={prefetch_groups}"
started = time.perf_counter()
prg = cl.Program(ctx, Path(args.source).read_text()).build(options=opts)
print(f"built in {time.perf_counter()-started:.1f}s grf={args.grf} prefetch_groups={prefetch_groups}")

mf = cl.mem_flags
dQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Q))
dK = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(K))
dV = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(V))
dM = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(mask))
dO = cl.Buffer(ctx, mf.WRITE_ONLY, HEADS * TOKENS * HEAD_DIM * 2)
pf_bytes = int(args.prefetch_mb * (1 << 20)) // 64 * 64
dPF = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rng.integers(0, 255, max(pf_bytes, 64), dtype=np.uint8))
dEvict = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(64 << 20, np.uint8))
touch = cl.Kernel(cl.Program(ctx, """
__kernel void touch(__global const uint16* p, uint n, __global uint* o){uint16 a=0;
for(uint i=get_global_id(0);i<n;i+=get_global_size(0))a+=p[i]; if(a.s0==7u)o[0]=a.s1;}""").build(), "touch")
dScratch = cl.Buffer(ctx, mf.READ_WRITE, 64)

kern = cl.Kernel(prg, "expert_sdpa")
priv = kern.get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE, dev)
try:
    spill = kern.get_work_group_info(0x4109, dev)
except Exception:
    spill = "n/a"
print(f"expert_sdpa: private_mem={priv} spill={spill}")
kargs = [dQ, dK, dV, dM, dO]
if prefetch_groups:
    kargs += [dPF, np.uint32(pf_bytes // 64)]
kern.set_args(*kargs)
lsz = 16 * 16
gsz = lsz * (ATTN_GROUPS + prefetch_groups)

def evict():
    touch.set_args(dEvict, np.uint32((64 << 20) // 64), dScratch)
    cl.enqueue_nd_range_kernel(q, touch, (256 * 160,), (256,))

def run():
    return cl.enqueue_nd_range_kernel(q, kern, (gsz,), (lsz,))

run().wait()
out = np.empty((HEADS, TOKENS, HEAD_DIM), np.float16)
cl.enqueue_copy(q, out, dO).wait()
err = out.astype(np.float32) - O_ref
print(f"max_abs={np.abs(err).max():.5f} rmse={np.sqrt((err**2).mean()):.6f} ref_rms={np.sqrt((O_ref**2).mean()):.4f} finite={np.isfinite(out).all()}")

def us(e):
    return (e.profile.end - e.profile.start) / 1e3
cold, warm = [], []
# Enqueue everything back to back so the GPU stays clocked up, as in the model.
events = []
for i in range(args.iterations):
    evict(); e1 = run(); e2 = run(); events.append((e1, e2))
q.finish()
for e1, e2 in events:
    cold.append(us(e1)); warm.append(us(e2))
print(f"expert_sdpa median {np.median(cold):.2f} us cold K/V, {np.median(warm):.2f} us warm")
if prefetch_groups:
    rd, base = [], []
    for i in range(50):
        evict(); run()
        touch.set_args(dPF, np.uint32(pf_bytes // 64), dScratch)
        e3 = cl.enqueue_nd_range_kernel(q, touch, (256 * 160,), (256,)); e3.wait(); rd.append(us(e3))
        evict()
        touch.set_args(dPF, np.uint32(pf_bytes // 64), dScratch)
        e3 = cl.enqueue_nd_range_kernel(q, touch, (256 * 160,), (256,)); e3.wait(); base.append(us(e3))
    print(f"prefetched {args.prefetch_mb} MB: re-read after attention {np.median(rd):.2f} us vs cold {np.median(base):.2f} us")
