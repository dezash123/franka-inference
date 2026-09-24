// Coalesced subgroup operations for prefix copies and expert rotary math.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef PREFIX_TOKENS
#define PREFIX_TOKENS 968
#endif
#ifndef WORKERS
#define WORKERS 64
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WORKERS,1,1)))
kernel void qkv_rope_kv(global const half* restrict input,
                       global const half* restrict cos,
                       global const half* restrict sin,
                       global const half* restrict prefix_k,
                       global const half* restrict prefix_v,
                       global half* restrict q,
                       global half* restrict k,
                       global half* restrict v) {
    int i=get_global_id(0);
    const int prefix_blocks=PREFIX_TOKENS*256/32;
    if (i<prefix_blocks) {
        int base=(i/16)*512;
        #pragma unroll
        for (int offset=0;offset<512;offset+=128) {
            // Each subgroup copies two rows; an odd prefix length leaves the
            // last pair half empty and must not touch the expert rows.
            if ((base+offset)/256>=PREFIX_TOKENS) break;
            ushort8 keys=intel_sub_group_block_read_us8(
                (global const ushort*)(prefix_k+base+offset));
            ushort8 values=intel_sub_group_block_read_us8(
                (global const ushort*)(prefix_v+base+offset));
            intel_sub_group_block_write_us8((global ushort*)(k+base+offset),keys);
            intel_sub_group_block_write_us8((global ushort*)(v+base+offset),values);
        }
        return;
    }
    int tile=(i-prefix_blocks)/16;
    int h=tile%10;
    int p=tile/10;
    if (p>=15) return;
    int input_idx=p*2560+h*256;
    half8 in1=as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input+input_idx)));
    half8 in2=as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input+input_idx+128)));
    if (h==9) {
        intel_sub_group_block_write_us8((global ushort*)(v+(PREFIX_TOKENS+p)*256),as_ushort8(in1));
        intel_sub_group_block_write_us8((global ushort*)(v+(PREFIX_TOKENS+p)*256+128),as_ushort8(in2));
        return;
    }
    int cos_idx=p*256;
    half8 cos1=as_half8(intel_sub_group_block_read_us8((global const ushort*)(cos+cos_idx)));
    half8 cos2=as_half8(intel_sub_group_block_read_us8((global const ushort*)(cos+cos_idx+128)));
    half8 sin1=as_half8(intel_sub_group_block_read_us8((global const ushort*)(sin+cos_idx)));
    half8 sin2=as_half8(intel_sub_group_block_read_us8((global const ushort*)(sin+cos_idx+128)));
    half8 out1=cos1*in1-sin1*in2;
    half8 out2=cos2*in2+sin2*in1;
    global half* output=h<8?q+(h*15+p)*256:k+(PREFIX_TOKENS+p)*256;
    intel_sub_group_block_write_us8((global ushort*)output,as_ushort8(out1));
    intel_sub_group_block_write_us8((global ushort*)(output+128),as_ushort8(out2));
}
