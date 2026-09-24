// Fused expert rotary/QKV materialization and immutable prefix K/V copy.
// Every invocation owns its complete K/V outputs; no cross-step mutation.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef PREFIX_TOKENS
#define PREFIX_TOKENS 968
#endif
#ifndef WORKERS
#define WORKERS 128
#endif
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
        int base=i*32;
        half16 k0=vload16(0,prefix_k+base);
        half16 k1=vload16(0,prefix_k+base+16);
        half16 v0=vload16(0,prefix_v+base);
        half16 v1=vload16(0,prefix_v+base+16);
        vstore16(k0,0,k+base);
        vstore16(k1,0,k+base+16);
        vstore16(v0,0,v+base);
        vstore16(v1,0,v+base+16);
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
