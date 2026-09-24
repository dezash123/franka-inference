#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WORKERS, 1, 1)))
kernel void transpose_quantize(global const half* input,
#ifdef PI05_COMBINE_SPLITS
                               global const float* scratch,
#endif
                               global char* output, global half* scales) {
    const uint row = get_group_id(1);
    const uint token = row % TOKENS;
    const uint batch = row / TOKENS;
    const uint lid = get_local_id(0);
    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    local float maxima[WORKERS/16];
    half values[CHANNELS/WORKERS];
    float maximum = convert_float((half)0.003h);
#ifdef PI05_COMBINE_SPLITS
    // Split-K attention combine (same arithmetic as the stock finalization):
    // scratch = partials [H][Q][S][D] f32, sums [H][Q][S], maxima [H][Q][S].
    global const float* sums = scratch + (uint)HEADS*TOKENS*PI05_COMBINE_SPLITS*HEAD_DIM;
    global const float* split_max = sums + (uint)HEADS*TOKENS*PI05_COMBINE_SPLITS;
    #pragma unroll
    for (int i = 0; i < CHANNELS/WORKERS; ++i) {
        uint channel = lid + i*WORKERS;
        const uint head = channel/HEAD_DIM, dim = channel%HEAD_DIM;
        const uint hq = (batch*HEADS + head)*TOKENS + token;
        float global_max = -INFINITY;
        #pragma unroll
        for (int s = 0; s < PI05_COMBINE_SPLITS; ++s) global_max = fmax(global_max, split_max[hq*PI05_COMBINE_SPLITS + s]);
        float weights[PI05_COMBINE_SPLITS];
        float global_sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < PI05_COMBINE_SPLITS; ++s) {
            weights[s] = sums[hq*PI05_COMBINE_SPLITS + s] * native_exp(split_max[hq*PI05_COMBINE_SPLITS + s] - global_max);
            global_sum += weights[s];
        }
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < PI05_COMBINE_SPLITS; ++s)
            acc += scratch[(hq*PI05_COMBINE_SPLITS + s)*HEAD_DIM + dim] * weights[s];
        values[i] = convert_half(acc / global_sum);
        maximum = fmax(maximum, convert_float(fabs(values[i])));
    }
#else
    #pragma unroll
    for (int i = 0; i < CHANNELS/WORKERS; ++i) {
        uint channel = lid + i*WORKERS;
        values[i] = input[((batch*HEADS + channel/HEAD_DIM)*TOKENS + token)*HEAD_DIM + channel%HEAD_DIM];
        maximum = fmax(maximum, convert_float(fabs(values[i])));
    }
#endif
    maximum = sub_group_reduce_max(maximum);
    if (lane == 0) maxima[sg] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    maximum = sub_group_reduce_max(lane < WORKERS/16 ? maxima[lane] : 0.0f);
    half scale = (half)127.0h / convert_half(maximum);
    if (lid == 0) scales[row] = (half)1.0h / scale;
    #pragma unroll
    for (int i = 0; i < CHANNELS/WORKERS; ++i) {
        char value = convert_char_rte(values[i] * scale);
        intel_sub_group_block_write_uc((global uchar*)(output + row*CHANNELS + i*WORKERS + sg*16),
                                       as_uchar(value));
    }
}
