"""L2 residency probe: does data read by one kernel stay cached for the next?

Measures a bandwidth-bound read kernel over an 8 MB buffer, alone and
immediately after a prefetch kernel touched the same buffer, on an in-order
queue. Also measures whether a "prefetch fused into a latency-bound kernel"
pattern costs extra time.
"""
import os
import time
import numpy as np
import pyopencl as cl

SRC = r"""
// Sum-reduce a byte buffer; every work-item reads 64 B per iteration.
__kernel void stream_read(__global const uint16* p, uint n_vec, __global uint* out) {
    uint gid = get_global_id(0), gsz = get_global_size(0);
    uint16 acc = 0;
    for (uint i = gid; i < n_vec; i += gsz) acc += p[i];
    uint s = acc.s0+acc.s1+acc.s2+acc.s3+acc.s4+acc.s5+acc.s6+acc.s7+acc.s8+acc.s9+acc.sa+acc.sb+acc.sc+acc.sd+acc.se+acc.sf;
    if (s == 0x12345678u) out[0] = s;
}
// Latency-bound kernel: `lat_groups` workgroups spin on a dependent chain for
// ~16 us; the remaining workgroups stream `p` into cache.
__kernel void latency_plus_prefetch(__global const uint16* p, uint n_vec, uint lat_groups,
                                    uint spin, __global uint* out) {
    uint g = get_group_id(0);
    if (g < lat_groups) {
        float x = (float)get_local_id(0);
        for (uint i = 0; i < spin; ++i) x = native_sin(x) * 1.0001f + 0.5f;
        if (x == 12345.0f) out[1] = 1;
        return;
    }
    uint pf_groups = get_num_groups(0) - lat_groups;
    uint pid = (g - lat_groups) * get_local_size(0) + get_local_id(0);
    uint psz = pf_groups * get_local_size(0);
    uint16 acc = 0;
    for (uint i = pid; i < n_vec; i += psz) acc += p[i];
    if (acc.s0 == 0x12345678u) out[2] = acc.s1;
}
"""

ctx = cl.Context(dev_type=cl.device_type.GPU)
dev = ctx.devices[int(os.environ.get("PI05_L2_DEVICE", "0"))]
print("device", dev.name, "global cache", dev.global_mem_cache_size)
q = cl.CommandQueue(ctx, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)
prg = cl.Program(ctx, SRC).build()
mb = int(os.environ.get("PI05_L2_MB", "8"))
n = mb << 20
n_vec = n // 64
host = np.random.randint(0, 255, n, dtype=np.uint8)
buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host)
evict = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(64 << 20, np.uint8))
out = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 64)
lsz = 256
gsz = lsz * 160

def ns(ev):
    ev.wait(); return (ev.profile.end - ev.profile.start) / 1000.0

def run_read(b, nv):
    return prg.stream_read(q, (gsz,), (lsz,), b, np.uint32(nv), out)

def evict_cache():
    run_read(evict, (64 << 20) // 64).wait()

def median(f, k=15):
    return float(np.median([f() for _ in range(k)]))

# 1. cold read vs warm read
def cold():
    evict_cache(); return ns(run_read(buf, n_vec))
def warm():
    evict_cache(); run_read(buf, n_vec).wait(); return ns(run_read(buf, n_vec))
c, w = median(cold), median(warm)
print(f"{mb} MB cold read {c:.1f} us ({n/c/1e3:.0f} GB/s); warm read after prior kernel {w:.1f} us ({n/w/1e3:.0f} GB/s)")

# 2. latency kernel alone vs with fused prefetch; then the following read
spin = int(os.environ.get("PI05_L2_SPIN", "4000"))
def lat_only():
    evict_cache()
    return ns(prg.latency_plus_prefetch(q, (8 * lsz,), (lsz,), buf, np.uint32(0), np.uint32(8), np.uint32(spin), out))
def lat_pf():
    evict_cache()
    return ns(prg.latency_plus_prefetch(q, (gsz,), (lsz,), buf, np.uint32(n_vec), np.uint32(8), np.uint32(spin), out))
def read_after_lat_pf():
    evict_cache()
    prg.latency_plus_prefetch(q, (gsz,), (lsz,), buf, np.uint32(n_vec), np.uint32(8), np.uint32(spin), out).wait()
    return ns(run_read(buf, n_vec))
lo, lp, ra = median(lat_only), median(lat_pf), median(read_after_lat_pf)
print(f"latency kernel alone {lo:.1f} us; with fused {mb} MB prefetch {lp:.1f} us; read after fused prefetch {ra:.1f} us")
