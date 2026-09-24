// Preserve native RotateHalf vector arithmetic and three-dimensional dispatch.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void packed_rope(global const half* restrict input,
                        global const half* restrict cos,
                        global const half* restrict sin,
                        global half* restrict output) {
    int h=get_global_id(1);
    int p=get_global_id(2)/8;
    int r=(get_global_id(2)%8)*16;
    int input_idx=p*2560+CHANNEL_OFFSET+h*256;
    int cos_idx=p*256;
    int output_idx=(h*TOKENS+p)*256;
    half16 in1=*(global const half16*)(input+input_idx+r);
    half16 in2=*(global const half16*)(input+input_idx+128+r);
    half16 cos1=*(global const half16*)(cos+cos_idx+r);
    half16 cos2=*(global const half16*)(cos+cos_idx+128+r);
    half16 sin1=*(global const half16*)(sin+cos_idx+r);
    half16 sin2=*(global const half16*)(sin+cos_idx+128+r);
    half16 out1=cos1*in1-sin1*in2;
    half16 out2=cos2*in2+sin2*in1;
    *(global half16*)(output+output_idx+r)=out1;
    *(global half16*)(output+output_idx+128+r)=out2;
}
