#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
kernel void adaln_quantize(global const half* input, global const half* modulation,
                          global char* output, global half* scales
#ifdef PI05_PF_BYTES0
                          , global const uint16* pf0
#endif
                          ) {
#ifdef PI05_PF_BYTES0
    // Trailing workgroups stream the consumer matrix's weights into L2.
    if (get_group_id(1) >= PI05_PF_NORMAL) {
        const uint pg = get_group_id(1) - PI05_PF_NORMAL;
        const uint plid = pg * get_local_size(0) + get_local_id(0);
        const uint stride = PI05_PF_GROUPS * get_local_size(0);
        uint16 acc = (uint16)0;
        for (uint i = plid; i < PI05_PF_BYTES0 / 64; i += stride) acc += pf0[i];
        if (acc.s0 == 0x9e3779b9u && acc.s1 == 0x7f4a7c15u) scales[0] = 0.0h;
        return;
    }
#endif
    const uint row = get_group_id(1);
    const uint lid = get_local_id(0);
    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    float rms = 0.0f;
    local float sums[8];
    half8 raw = as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input + row*1024 + sg*128)));
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        rms += native_powr(convert_float(raw[i]), 2);
    rms = sub_group_reduce_add(rms);
    if (lane == 0) sums[sg] = rms;
    barrier(CLK_LOCAL_MEM_FENCE);
    rms = lane < 8 ? sums[lane] : 0.0f;
    #pragma unroll
    for (uint stride = 4; stride > 0; stride /= 2)
        rms += intel_sub_group_shuffle_xor(rms, stride);
    rms = sub_group_broadcast(rms, 0)/1024;
    rms = native_powr(sqrt(rms + PI05_RMS_EPSILON), -1);
    half8 gamma = as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(modulation + sg*128)));
    half8 beta = as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(modulation + 1024 + sg*128)));
    half8 normalized;
    float maximum = convert_float((half)0.003h);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        half base = convert_half_rte(rms * convert_float(raw[i]));
        half scale = gamma[i] + (half)1.0h;
        normalized[i] = fma(base, scale, beta[i]);
        maximum = fmax(maximum, convert_float(fabs(normalized[i])));
    }
    maximum = sub_group_reduce_max(maximum);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) sums[sg] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    maximum = sub_group_reduce_max(lane < 8 ? sums[lane] : 0.0f);
    half quant_scale = (half)127.0h / convert_half(maximum);
    if (lid == 0) scales[row] = (half)1.0h / quant_scale;
    char8 quantized;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        quantized[i] = convert_char_rte(normalized[i] * quant_scale);
    intel_sub_group_block_write_uc8((global uchar*)(output + row*1024 + sg*128),
                                    as_uchar8(quantized));
}
