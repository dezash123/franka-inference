// Coalesced subgroup loads/stores with the verified FP16 rotary expressions.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
__attribute__((intel_reqd_sub_group_size(16)))
kernel void qkv_rope(global const half* restrict input,
                    global const half* restrict cos,
                    global const half* restrict sin,
                    global half* restrict q,
                    global half* restrict k,
                    global half* restrict v) {
    int tile=get_global_id(0)/16;
    int h=tile%10;
    int p=tile/10;
    if (p>=TOKENS) return;
    int input_idx=p*2560+h*256;
    half8 in1=as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input+input_idx)));
    half8 in2=as_half8(intel_sub_group_block_read_us8(
        (global const ushort*)(input+input_idx+128)));
    if (h==9) {
        intel_sub_group_block_write_us8((global ushort*)(v+p*256),as_ushort8(in1));
        intel_sub_group_block_write_us8((global ushort*)(v+p*256+128),as_ushort8(in2));
        return;
    }
    int cos_idx=p*256;
    half8 cos1=as_half8(intel_sub_group_block_read_us8((global const ushort*)(cos+cos_idx)));
    half8 cos2=as_half8(intel_sub_group_block_read_us8((global const ushort*)(cos+cos_idx+128)));
    half8 sin1=as_half8(intel_sub_group_block_read_us8((global const ushort*)(sin+cos_idx)));
    half8 sin2=as_half8(intel_sub_group_block_read_us8((global const ushort*)(sin+cos_idx+128)));
    half8 out1=cos1*in1-sin1*in2;
    half8 out2=cos2*in2+sin2*in1;
    global half* output=h<8?q+(h*TOKENS+p)*256:k+p*256;
    intel_sub_group_block_write_us8((global ushort*)output,as_ushort8(out1));
    intel_sub_group_block_write_us8((global ushort*)(output+128),as_ushort8(out2));
}
