// L2 prefetch helper. Streams weight buffers through the cache so the
// following memory-bound matrices read them warm. Results are discarded; the
// tiny output only exists to give the graph a dependency edge.
//
// Arguments: dependency input (unused), PREFETCH_INPUTS buffers whose byte
// limits are the compile-time BYTES0..BYTES2, then the dummy output.
__kernel void pi05_prefetch(__global const uchar* dependency,
                            __global const uint16* buf0,
#if PREFETCH_INPUTS > 1
                            __global const uint16* buf1,
#endif
#if PREFETCH_INPUTS > 2
                            __global const uint16* buf2,
#endif
                            __global uint* out) {
    const uint gid = get_global_id(0), gsz = get_global_size(0);
    uint16 acc = (uint16)0;
    for (uint i = gid; i < BYTES0 / 64; i += gsz) acc += buf0[i];
#if PREFETCH_INPUTS > 1
    for (uint i = gid; i < BYTES1 / 64; i += gsz) acc += buf1[i];
#endif
#if PREFETCH_INPUTS > 2
    for (uint i = gid; i < BYTES2 / 64; i += gsz) acc += buf2[i];
#endif
    if (acc.s0 == 0x9e3779b9u && acc.s1 == 0x7f4a7c15u && acc.s2 == dependency[0]) out[0] = acc.s3;
}
