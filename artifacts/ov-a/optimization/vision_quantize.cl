// Exact per-token quantization for the padded vision MLP, 768 rows x 4352.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#ifndef WORKERS
#define WORKERS 128
#endif
#ifndef CACHE_INPUT
#define CACHE_INPUT 1
#endif
#define CHANNELS 4352
#define CHUNKS ((CHANNELS+WORKERS*8-1)/(WORKERS*8))

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WORKERS,1,1)))
kernel void vision_quantize(global const half* input, global char* output,
                            global half* scales) {
    const uint row=get_group_id(1);
    const uint sg=get_sub_group_id(), lane=get_sub_group_local_id();
    local float partial[WORKERS/16];
    half8 values[CHUNKS];
    half8 max8=(half8)0.003h;
    #pragma unroll
    for (uint i=0;i<CHUNKS;++i) {
        uint base=i*WORKERS*8+sg*128;
        // CHANNELS is a multiple of 128, so every active block is complete.
        half8 v=(half8)0;
        if (base<CHANNELS)
            v=as_half8(intel_sub_group_block_read_us8(
                (global const ushort*)(input+row*CHANNELS+base)));
        #if CACHE_INPUT
        values[i]=v;
        #endif
        max8=fmax(max8,fabs(v));
    }
    half4 max4=fmax(max8.lo,max8.hi);
    half2 max2=fmax(max4.lo,max4.hi);
    float maximum=sub_group_reduce_max(convert_float(fmax(max2.s0,max2.s1)));
    if (lane==0) partial[sg]=maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    maximum=0;
    #pragma unroll
    for (uint i=lane;i<WORKERS/16;i+=16) maximum=fmax(maximum,partial[i]);
    maximum=sub_group_reduce_max(maximum);
    half scale=(half)127.0h/convert_half(maximum);
    if (get_local_id(0)==0) scales[row]=(half)1.0h/scale;
    #pragma unroll
    for (uint i=0;i<CHUNKS;++i) {
        uint base=i*WORKERS*8+sg*128;
        if (base<CHANNELS) {
            #if CACHE_INPUT
            half8 v=values[i];
            #else
            half8 v=as_half8(intel_sub_group_block_read_us8(
                (global const ushort*)(input+row*CHANNELS+base)));
            #endif
            char8 q=convert_char8_rte(v*(half8)scale);
            intel_sub_group_block_write_uc8(
                (global uchar*)(output+row*CHANNELS+base),as_uchar8(q));
        }
    }
}
