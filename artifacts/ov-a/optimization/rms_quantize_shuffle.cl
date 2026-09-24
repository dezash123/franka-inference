#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

// Match native RMS's 256 workers / eight elements and reduction order.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void rms_quantize(global const half* input, global const half* gamma,
                         global char* output, global half* scales) {
    const uint row = get_group_id(1);
    const uint lid = get_local_id(0);
    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    const uint offset = sg * 128 + lane;
    float data[8];
    float rms = 0.0f;
    local float sums[16];
    half8 raw = as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input + row*2048 + sg*128)));
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = convert_float(raw[i]);
        rms += native_powr(v, 2);
        data[i] = v;
    }
    rms = sub_group_reduce_add(rms);
    if (lane == 0) sums[sg] = rms;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Lane zero follows the same 8,4,2,1 addition tree as native SLM.
    // Broadcast only that lane's result; other lanes have different orders.
    rms = sums[lane];
    #pragma unroll
    for (uint stride = 8; stride > 0; stride /= 2)
        rms += intel_sub_group_shuffle_xor(rms, stride);
    rms = sub_group_broadcast(rms, 0)/2048;
    rms = native_powr(sqrt(rms + PI05_RMS_EPSILON), -1);
    half8 weights = as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(gamma + sg*128)));
    half normalized[8];
    float maximum = convert_float((half)0.003h);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        normalized[i] = convert_half_rte((rms * data[i]) * convert_float(weights[i]));
        maximum = fmax(maximum, convert_float(fabs(normalized[i])));
    }
    maximum = sub_group_reduce_max(maximum);
    // All subgroups have consumed sums[0] before reusing the shared array.
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) sums[sg] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Max is exact under regrouping; each subgroup reads the sixteen maxima.
    maximum = sub_group_reduce_max(sums[lane]);
    half quant_scale = (half)127.0h / convert_half(maximum);
    if (lid == 0) scales[row] = (half)1.0h / quant_scale;
    char8 quantized;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        quantized[i] = convert_char_rte(normalized[i] * quant_scale);
    intel_sub_group_block_write_uc8((global uchar*)(output + row*2048 + sg*128),
                                    as_uchar8(quantized));
}
