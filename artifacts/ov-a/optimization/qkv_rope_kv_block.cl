// Same fused operation, with subgroup-coalesced immutable prefix copies.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef PREFIX_TOKENS
#define PREFIX_TOKENS 968
#endif
#ifndef WORKERS
#define WORKERS 128
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
                       global half* restrict v
#ifdef PI05_PF_BYTES0
                       , global const uint16* restrict pf0
#endif
                       ) {
#ifdef PI05_PF_BYTES0
    // Trailing workgroups stream the o_proj weights into L2 for after attention.
    if (get_group_id(0) >= PI05_PF_NORMAL) {
        const uint pg = get_group_id(0) - PI05_PF_NORMAL;
        const uint plid = pg * get_local_size(0) + get_local_id(0);
        const uint stride = PI05_PF_GROUPS * get_local_size(0);
        uint16 acc = (uint16)0;
        for (uint i = plid; i < PI05_PF_BYTES0 / 64; i += stride) acc += pf0[i];
        if (acc.s0 == 0x9e3779b9u && acc.s1 == 0x7f4a7c15u) q[0] = 0.0h;
        return;
    }
#endif
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
    i-=prefix_blocks;
    if (i>=15*10*8) return;
    int r=(i%8)*16;
    int h=(i/8)%10;
    int p=i/(8*10);
    int input_idx=p*2560+h*256;
    half16 in1=*(global const half16*)(input+input_idx+r);
    half16 in2=*(global const half16*)(input+input_idx+128+r);
    if (h==9) {
        *(global half16*)(v+(PREFIX_TOKENS+p)*256+r)=in1;
        *(global half16*)(v+(PREFIX_TOKENS+p)*256+128+r)=in2;
        return;
    }
    int cos_idx=p*256;
    half16 cos1=*(global const half16*)(cos+cos_idx+r);
    half16 cos2=*(global const half16*)(cos+cos_idx+128+r);
    half16 sin1=*(global const half16*)(sin+cos_idx+r);
    half16 sin2=*(global const half16*)(sin+cos_idx+128+r);
    half16 out1=cos1*in1-sin1*in2;
    half16 out2=cos2*in2+sin2*in1;
    global half* output=h<8?q+(h*15+p)*256:k+(PREFIX_TOKENS+p)*256;
    *(global half16*)(output+r)=out1;
    *(global half16*)(output+128+r)=out2;
}
