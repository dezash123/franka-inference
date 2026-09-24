"""Persistent-kernel feasibility on B580: grid-barrier cost and resident capacity.

Launches N workgroups that all spin on an atomic counter B times (a
software grid barrier). If N exceeds the resident capacity the kernel
deadlocks, so the probe uses a watchdog (kernel aborts when a barrier waits
longer than a budget). Reports per-barrier cost.
"""
import os, sys, time
import numpy as np
import pyopencl as cl

SRC = r"""
// Sense-reversing grid barrier on a global counter. `gen` counts barriers.
inline void grid_barrier(__global volatile atomic_uint* count, __global volatile atomic_uint* gen,
                         uint nwg, uint* local_gen, __global uint* aborted, uint budget) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        uint my_gen = *local_gen;
        uint arrived = atomic_fetch_add_explicit(count, 1u, memory_order_acq_rel, memory_scope_device);
        if (arrived == nwg - 1) {
            atomic_store_explicit(count, 0u, memory_order_relaxed, memory_scope_device);
            atomic_store_explicit(gen, my_gen + 1, memory_order_release, memory_scope_device);
        } else {
            uint spins = 0;
            while (atomic_load_explicit(gen, memory_order_acquire, memory_scope_device) == my_gen) {
                if (++spins > budget) { *aborted = 1; break; }
            }
        }
        *local_gen = my_gen + 1;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
__kernel void persistent(__global atomic_uint* count, __global atomic_uint* gen, uint nwg, uint iters,
                         __global uint* aborted, __global float* work, uint work_n) {
    uint local_gen = 0;
    float acc = 0.0f;
    for (uint it = 0; it < iters; ++it) {
        // token work: every WG touches a slice of `work`
        for (uint i = get_global_id(0); i < work_n; i += get_global_size(0)) acc += work[i];
        grid_barrier(count, gen, nwg, &local_gen, aborted, 20000000u);
        if (*aborted) return;
    }
    if (acc == 12345.678f) work[0] = acc;
}
"""
ctx = cl.Context(dev_type=cl.device_type.GPU)
dev = ctx.devices[0]
q = cl.CommandQueue(ctx, dev, properties=cl.command_queue_properties.PROFILING_ENABLE)
prg = cl.Program(ctx, SRC).build(options="-cl-std=CL3.0")
k = cl.Kernel(prg, "persistent")
mf = cl.mem_flags
zeros = np.zeros(2, np.uint32)
count = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros[:1])
gen = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros[1:])
aborted = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, np.uint32))
work = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(1 << 20, np.float32))
print("device", dev.name, "CUs", dev.max_compute_units, "max wg", dev.max_work_group_size)
for lsz, nwg in [(256, 40), (256, 80), (256, 120), (256, 160), (256, 200), (128, 160), (128, 320), (512, 80)]:
    for iters in (200,):
        cl.enqueue_copy(q, count, np.zeros(1, np.uint32)); cl.enqueue_copy(q, gen, np.zeros(1, np.uint32)); cl.enqueue_copy(q, aborted, np.zeros(1, np.uint32))
        k.set_args(count, gen, np.uint32(nwg), np.uint32(iters), aborted, work, np.uint32(0))
        e = cl.enqueue_nd_range_kernel(q, k, (lsz * nwg,), (lsz,)); 
        try:
            e.wait()
        except Exception as ex:
            print(lsz, nwg, "failed", ex); continue
        ab = np.zeros(1, np.uint32); cl.enqueue_copy(q, ab, aborted).wait()
        us = (e.profile.end - e.profile.start) / 1e3
        print(f"lsz={lsz} nwg={nwg} iters={iters}: {'ABORTED (not all resident)' if ab[0] else f'{us/iters:.2f} us per barrier'}")
