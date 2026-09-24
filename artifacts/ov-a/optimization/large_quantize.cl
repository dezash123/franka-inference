#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#ifndef WORKERS
#define WORKERS 256
#endif
#ifndef CACHE_INPUT
#define CACHE_INPUT 1
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WORKERS, 1, 1)))
kernel void large_quantize(global const half* input, global char* output,
                           global half* scales) {
    const uint row = get_group_id(1);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint lid = get_local_id(0);
    local float maxima[WORKERS/16];
    half8 values[16384/(WORKERS*8)];
    float maximum = convert_float((half)0.003h);
    #pragma unroll
    for (int i = 0; i < 16384/(WORKERS*8); ++i) {
        half8 value = as_half8(intel_sub_group_block_read_us8(
            (global const ushort*)(input + row*16384 + i*WORKERS*8 + sg*128)));
        #if CACHE_INPUT
        values[i] = value;
        #endif
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            maximum = fmax(maximum, convert_float(fabs(value[j])));
    }
    maximum = sub_group_reduce_max(maximum);
    if (lane == 0) maxima[sg] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    maximum = 0.0f;
    #pragma unroll
    for (uint i = lane; i < WORKERS/16; i += 16)
        maximum = fmax(maximum, maxima[i]);
    maximum = sub_group_reduce_max(maximum);
    half scale = (half)127.0h / convert_half(maximum);
    if (lid == 0) scales[row] = (half)1.0h / scale;
    #pragma unroll
    for (int i = 0; i < 16384/(WORKERS*8); ++i) {
        #if CACHE_INPUT
        half8 value = values[i];
        #else
        half8 value = as_half8(intel_sub_group_block_read_us8(
            (global const ushort*)(input + row*16384 + i*WORKERS*8 + sg*128)));
        #endif
        char8 quantized = convert_char8_rte(value * (half8)scale);
        intel_sub_group_block_write_uc8(
            (global uchar*)(output + row*16384 + i*WORKERS*8 + sg*128),
            as_uchar8(quantized));
    }
}
