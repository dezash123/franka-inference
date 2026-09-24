// Q and K are read directly by packed_rope when their consumers are eligible.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifndef PI05_SKIP_QK
#define PI05_SKIP_QK 0
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128,1,1)))
kernel void qkv_split(global const half* restrict input,
                      global half* restrict q, global half* restrict k,
                      global half* restrict v) {
    int index=get_global_id(0)*8;
#if PI05_SKIP_QK
    if (index>=ROWS*V_WIDTH) return;
    int row=index/V_WIDTH, col=index%V_WIDTH;
    half8 value=vload8(0,input+row*(Q_WIDTH+K_WIDTH+V_WIDTH)+Q_WIDTH+K_WIDTH+col);
    vstore8(value,0,v+index);
#else
    if (index>=ROWS*(Q_WIDTH+K_WIDTH+V_WIDTH)) return;
    int row=index/(Q_WIDTH+K_WIDTH+V_WIDTH);
    int col=index%(Q_WIDTH+K_WIDTH+V_WIDTH);
    half8 value=vload8(0,input+index);
    if (col<Q_WIDTH) {
        vstore8(value,0,q+row*Q_WIDTH+col);
    } else if (col<Q_WIDTH+K_WIDTH) {
        vstore8(value,0,k+row*K_WIDTH+col-Q_WIDTH);
    } else {
        vstore8(value,0,v+row*V_WIDTH+col-Q_WIDTH-K_WIDTH);
    }
#endif
}
