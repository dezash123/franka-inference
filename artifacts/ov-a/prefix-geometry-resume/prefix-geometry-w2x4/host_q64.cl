// CPU offline compilation; microkernel markers patched with gemmstone::fuse.
// Gate/up with GELU epilogue writing INT8 activations quantized per token over
// 64-channel groups (one workgroup column of two 32-channel subgroups), plus
// the FP16 group scales consumed by the grouped down_proj GEMM.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#define DECLARE_2D_TILE_OPS(...)
#include "gateup.h"

#define TILE_AT(t,i,j) ((t).x[(i)/ugemm_gateup_c_type_block0 + \
    ugemm_gateup_c_type_nblock0*((j)/ugemm_gateup_c_type_block1)] \
    [((i)%ugemm_gateup_c_type_block0)/16 + \
    ((j)%ugemm_gateup_c_type_block1)*(ugemm_gateup_c_type_block0/16)])

#if ugemm_gateup_sg_tile_m != 64 || ugemm_gateup_sg_per_wg_m != 2
#error Quantized epilogue assumes 64-row subgroup tiles and two m-subgroups per group.
#endif
#if ugemm_gateup_barrier_count != 0 || ugemm_gateup_slm_size != 0
#error Tail-subgroup skip requires an independent, barrier-free microkernel.
#endif
#define GROUP_CHANNELS 64

// GELU(g) * u in the FP16 rounding of the FP16 epilogue.
inline float2 gateup_activation(int2 gv, int2 uv, float token_scale, float2 gsc, float2 usc) {
    float2 g = (convert_float2(gv)*token_scale)*gsc;
    float2 u = (convert_float2(uv)*token_scale)*usc;
    u = convert_float2(convert_half2_rte(u));
    float2 t = g*g;
    t = t*0.044715f;
    t = fma(t,g,g);
    t = t*(-2.0f*0.7978845f*1.442695f);
    float2 activation = native_recip(1.0f+native_exp2(t))*g;
    return convert_float2(convert_half2_rte(activation*u));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16*ugemm_gateup_sg_per_wg_m,ugemm_gateup_sg_per_wg_n,1)))
kernel void gateup(
    global const char* restrict wg, global const char* restrict wu, global const char* restrict x,
    global const half* restrict sg, global const half* restrict su, global const half* restrict sx,
    global char* restrict y, global half* restrict yscale,
    global int* restrict debug_gate, global int* restrict debug_up,
    int channels, int tokens, int inner) {
    local char slm[ugemm_gateup_slm_size > 0 ? ugemm_gateup_slm_size : 1];
    local float group_max[ugemm_gateup_sg_per_wg_m][ugemm_gateup_sg_per_wg_n][ugemm_gateup_sg_tile_n];
    int lm = sub_group_broadcast((uint)(get_local_id(0)/16), 0);
    int ln = sub_group_broadcast((uint)get_local_id(1), 0);
    int cm = get_group_id(1)*ugemm_gateup_wg_tile_m;
    int tn = get_group_id(0)*ugemm_gateup_wg_tile_n;
    const int lane = get_sub_group_local_id();
    // Tail subgroups take part in the barrier but neither compute nor write.
    const bool active = tn+ln*ugemm_gateup_sg_tile_n < tokens;
    ugemm_gateup_c_type pair;
    if (active)
        pair = ugemm_gateup(wg, inner, x, inner, channels*2, tokens, inner, cm, tn, 0, lm, ln, slm);
    int channel_base = cm/2 + lm*(ugemm_gateup_sg_tile_m/2);
    float2 gsc = (float2)(convert_float(sg[channel_base+lane]), convert_float(sg[channel_base+16+lane]));
    float2 usc = (float2)(convert_float(su[channel_base+lane]), convert_float(su[channel_base+16+lane]));
    float token_scales[ugemm_gateup_sg_tile_n/16];
    #pragma unroll
    for (int j=0; j<ugemm_gateup_sg_tile_n; j+=16) {
        const int token=tn+ln*ugemm_gateup_sg_tile_n+j+lane;
        token_scales[j/16]=token<tokens ? convert_float(sx[token]) : 0.0f;
    }
    // Pass 1: activations (FP16-rounded) kept in registers; per-token maximum over this subgroup's 32 channels.
    half2 vals[ugemm_gateup_sg_tile_n];
    if (active) {
        #pragma unroll
        for (int j = 0; j < ugemm_gateup_sg_tile_n; ++j) {
            int2 gv = (int2)(TILE_AT(pair,0,j), TILE_AT(pair,16,j));
            int2 uv = (int2)(TILE_AT(pair,32,j), TILE_AT(pair,48,j));
            float2 v = gateup_activation(gv, uv, sub_group_broadcast(token_scales[j/16],j%16), gsc, usc);
            vals[j] = convert_half2_rte(v);
            float m = fmax(fabs(v.s0), fabs(v.s1));
            m = sub_group_reduce_max(m);
            if (lane == 0) group_max[lm][ln][j] = m;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!active) return;
    // Pass 2: quantize with the 64-channel group maximum and write INT8 + scale.
    const int group_index = channel_base/GROUP_CHANNELS;
    const int groups = channels/GROUP_CHANNELS;
    #pragma unroll
    for (int j = 0; j < ugemm_gateup_sg_tile_n; ++j) {
        int token = tn + ln*ugemm_gateup_sg_tile_n + j;
        if (token < tokens) {
            float maximum = fmax(fmax(group_max[0][ln][j], group_max[1][ln][j]), convert_float((half)0.003h));
            half quant_scale = (half)127.0h / convert_half(maximum);
            char2 quantized = convert_char2_rte(vals[j] * (half2)quant_scale);
            intel_sub_group_block_write_uc2((global uchar*)(y+token*channels+channel_base), as_uchar2(quantized));
            if (lm == 0 && lane == 0) yscale[token*groups+group_index] = (half)1.0h / quant_scale;
        }
    }
}
