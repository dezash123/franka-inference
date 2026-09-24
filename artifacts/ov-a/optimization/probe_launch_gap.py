"""Launch-gap floor on B580: N tiny kernels enqueued back-to-back on an in-order queue."""
import time
import numpy as np
import pyopencl as cl

SRC = r"""
__kernel void tiny(__global float* x) { uint i = get_global_id(0); x[i] = x[i] * 1.0001f + 1.0f; }
"""
plat = [p for p in cl.get_platforms() if p.get_devices(cl.device_type.GPU)][0]; ctx = cl.Context(devices=[plat.get_devices(cl.device_type.GPU)[0]])
dev = ctx.devices[0]
prg = cl.Program(ctx, SRC).build()
buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(1 << 16, np.float32))
k = prg.tiny; k.set_args(buf)
for label, props in [("in-order", 0), ("out-of-order", cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)]:
    q = cl.CommandQueue(ctx, dev, properties=props)
    for gsz in (256, 15 * 1024, 1 << 16):
        for n in (200, 2000):
            for _ in range(2):
                cl.enqueue_nd_range_kernel(q, k, (gsz,), (256 if gsz >= 256 else gsz,)); q.finish()
            t0 = time.perf_counter()
            for i in range(n):
                cl.enqueue_nd_range_kernel(q, k, (gsz,), (256,))
            q.finish()
            t = (time.perf_counter() - t0) * 1e6 / n
            print(f"{label:12s} gsz={gsz:6d} n={n:5d}: {t:6.2f} us per kernel (wall/n)")
# with an event chain (each kernel waits for the previous event)
q = cl.CommandQueue(ctx, dev, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
n = 2000
t0 = time.perf_counter(); ev = None
for i in range(n):
    ev = cl.enqueue_nd_range_kernel(q, k, (15 * 1024,), (256,), wait_for=[ev] if ev else None)
q.finish()
print(f"ooo+event chain gsz=15360 n={n}: {(time.perf_counter()-t0)*1e6/n:6.2f} us per kernel")
# profiled: gap between consecutive kernels' end/start
q = cl.CommandQueue(ctx, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)
evs = [cl.enqueue_nd_range_kernel(q, k, (15 * 1024,), (256,)) for i in range(500)]
q.finish()
gaps = [(evs[i + 1].profile.start - evs[i].profile.end) / 1e3 for i in range(len(evs) - 1)]
durs = [(e.profile.end - e.profile.start) / 1e3 for e in evs]
print(f"profiled in-order: kernel {np.median(durs):.2f} us, gap median {np.median(gaps):.2f} us, p90 {np.percentile(gaps, 90):.2f}")
