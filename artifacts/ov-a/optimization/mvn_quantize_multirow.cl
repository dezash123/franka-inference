#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

// Several rows per workgroup; each row keeps the exact 256-worker reduction
// order of mvn_quantize_reduce.cl (8 subgroups of 32, then an 8-lane reduce).
#ifndef ROWS
#define ROWS 2
#endif

inline float sum_group(float value, local float* scratch, uint sg, uint lane) {
    value = sub_group_reduce_add(value);
    if (lane == 0) scratch[sg] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    value = sub_group_reduce_add(lane < 8 ? scratch[lane] : 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    return value;
}
inline float max_group(float value, local float* scratch, uint sg, uint lane) {
    value = sub_group_reduce_max(value);
    if (lane == 0) scratch[sg] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    return sub_group_reduce_max(lane < 8 ? scratch[lane] : 0.0f);
}

__attribute__((intel_reqd_sub_group_size(32)))
__attribute__((reqd_work_group_size(256*ROWS, 1, 1)))
kernel void mvn_quantize(global const half* input, global const half* gamma,
                         global const half* beta, global char* output,
                         global half* scales) {
    const uint sub = get_local_id(0) / 256;
    const uint row = get_group_id(1)*ROWS + sub;
    const uint lid = get_local_id(0) % 256;
    const uint sg = get_sub_group_id() - sub*8;
    const uint lane = get_sub_group_local_id();
    const uint count = lid < 128 ? 5 : 4;
    float data[5];
    local float scratch_all[ROWS*8];
    local float* scratch = scratch_all + sub*8;
    float mean = 0.0f;
    for (uint i = 0; i < count; ++i) {
        data[i] = convert_float(input[row*1152 + lid + i*256]);
        mean += data[i];
    }
    mean = sum_group(mean,scratch,sg,lane)/1152;
    float variance = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float delta = data[i] - mean;
        variance = fma(delta, delta, variance);
    }
    variance = sum_group(variance,scratch,sg,lane);
    variance = native_powr(variance/1152 + PI05_MVN_EPSILON, -0.5f);

    half normalized[5];
    float maximum = convert_float((half)0.003h);
    for (uint i = 0; i < count; ++i) {
        uint col = lid + i*256;
        half value = (convert_half(data[i]) - convert_half(mean)) * convert_half(variance);
        normalized[i] = fma(value, gamma[col], beta[col]);
        maximum = fmax(maximum, convert_float(fabs(normalized[i])));
    }
    maximum = max_group(maximum,scratch,sg,lane);
    half quant_scale = (half)127.0h / convert_half(maximum);
    if (lid == 0) scales[row] = (half)1.0h / quant_scale;
    for (uint i = 0; i < count; ++i)
        output[row*1152 + lid + i*256] = convert_char_rte(normalized[i] * quant_scale);
}
