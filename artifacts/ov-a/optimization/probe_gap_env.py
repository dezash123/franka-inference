"""Back-to-back kernel gap with a low-overhead ctypes enqueue loop (env vars from the caller)."""
import ctypes, os, time
import numpy as np
import pyopencl as cl

SRC = "__kernel void tiny(__global float* x) { uint i = get_global_id(0); x[i] = x[i] * 1.0001f + 1.0f; }"
plat = [p for p in cl.get_platforms() if p.get_devices(cl.device_type.GPU)][0]
ctx = cl.Context(devices=[plat.get_devices(cl.device_type.GPU)[0]])
dev = ctx.devices[0]
prg = cl.Program(ctx, SRC).build()
buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.zeros(1 << 16, np.float32))
k = prg.tiny; k.set_args(buf)
props = int(os.environ.get("QPROPS", "0"))
q = cl.CommandQueue(ctx, dev, properties=props)
lib = ctypes.CDLL("libOpenCL.so.1")
enq = lib.clEnqueueNDRangeKernel
enq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
fin = lib.clFinish; fin.argtypes = [ctypes.c_void_p]
gsz = (ctypes.c_size_t * 1)(int(os.environ.get("GSZ", "15360"))); lsz = (ctypes.c_size_t * 1)(256)
qp, kp = q.int_ptr, k.int_ptr
def run(n):
    t0 = time.perf_counter()
    for i in range(n):
        enq(qp, kp, 1, None, gsz, lsz, 0, None, None)
    t1 = time.perf_counter(); fin(qp); t2 = time.perf_counter()
    return (t1 - t0) * 1e6 / n, (t2 - t0) * 1e6 / n
for _ in range(3): run(500)
e, t = run(6000)
print(f"{os.environ.get('LABEL','base'):50s} enqueue {e:5.2f} us/kernel, total {t:5.2f} us/kernel")
