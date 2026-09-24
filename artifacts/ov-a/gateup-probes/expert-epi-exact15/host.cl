// CPU offline compilation; microkernel markers patched with gemmstone::fuse.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define DECLARE_2D_TILE_OPS(...)
#include "gateup.h"
#if ugemm_gateup_sg_tile_n != 16 || ugemm_gateup_sg_per_wg_n != 1
#error Exact-15 epilogue requires the expert tile.
#endif

#define TILE_AT(t,i,j) ((t).x[(i)/ugemm_gateup_c_type_block0 + \
    ugemm_gateup_c_type_nblock0*((j)/ugemm_gateup_c_type_block1)] \
    [((i)%ugemm_gateup_c_type_block0)/16 + \
    ((j)%ugemm_gateup_c_type_block1)*(ugemm_gateup_c_type_block0/16)])

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16*ugemm_gateup_sg_per_wg_m,ugemm_gateup_sg_per_wg_n,1)))
kernel void gateup(
    global const char* restrict wg, global const char* restrict wu, global const char* restrict x,
    global const half* restrict sg, global const half* restrict su, global const half* restrict sx,
    global half* restrict y, global int* restrict debug_gate, global int* restrict debug_up,
    int channels, int tokens, int inner) {
    local char slm[ugemm_gateup_slm_size > 0 ? ugemm_gateup_slm_size : 1];
    int lm = sub_group_broadcast((uint)(get_local_id(0)/16), 0);
    int ln = sub_group_broadcast((uint)get_local_id(1), 0);
    // Adjacent workgroups reuse the same weight tile across token blocks.
    int cm = get_group_id(1)*ugemm_gateup_wg_tile_m;
    int tn = get_group_id(0)*ugemm_gateup_wg_tile_n;
    // Interleave gate/up weights within each channel tile. One XMX call owns
    // both accumulators, avoiding the fixed-register overlap of two shims.
    ugemm_gateup_c_type pair = ugemm_gateup(
        wg, inner, x, inner, channels*2, tokens, inner, cm, tn, 0, lm, ln, slm);
    int channel_base = cm/2 + lm*(ugemm_gateup_sg_tile_m/2);
    float gate_scales[ugemm_gateup_sg_tile_m/32];
    float up_scales[ugemm_gateup_sg_tile_m/32];
    #pragma unroll
    for (int i = 0; i < ugemm_gateup_sg_tile_m/2; i += 16) {
        int channel = channel_base+i+get_sub_group_local_id();
        gate_scales[i/16] = convert_float(sg[channel]);
        up_scales[i/16] = convert_float(su[channel]);
    }

    // One guarded scale read per lane; the expert has 15 valid tokens.
    const int scale_token=tn+ln*ugemm_gateup_sg_tile_n+get_sub_group_local_id();
    const float token_scale=scale_token<tokens ? convert_float(sx[scale_token]) : 0.0f;
    #pragma unroll
    for (int j = 0; j < 15; ++j) {
        #if ugemm_gateup_sg_tile_m == 64
        int token = tn + ln*ugemm_gateup_sg_tile_n + j;
        if (1) {
            int2 gv = (int2)(TILE_AT(pair,0,j), TILE_AT(pair,16,j));
            int2 uv = (int2)(TILE_AT(pair,32,j), TILE_AT(pair,48,j));
            #if DEBUG_ACCUM
            int channel = channel_base + get_sub_group_local_id();
            debug_gate[token*channels+channel] = gv.s0;
            debug_gate[token*channels+channel+16] = gv.s1;
            debug_up[token*channels+channel] = uv.s0;
            debug_up[token*channels+channel+16] = uv.s1;
            #endif
            float2 g = (convert_float2(gv)*sub_group_broadcast(token_scale,j))*(float2)(gate_scales[0],gate_scales[1]);
            float2 u = (convert_float2(uv)*sub_group_broadcast(token_scale,j))*(float2)(up_scales[0],up_scales[1]);
            u = convert_float2(convert_half2_rte(u));
            float2 t = g*g;
            t = t*0.044715f;
            t = fma(t,g,g);
            t = t*(-2.0f*0.7978845f*1.442695f);
            float2 activation = native_recip(1.0f+native_exp2(t))*g;
            intel_sub_group_block_write_us2(
                (global ushort*)(y+token*channels+channel_base),
                as_ushort2(convert_half2_rte(activation*u)));
        }
        #else
        #pragma unroll
        for (int i = 0; i < ugemm_gateup_sg_tile_m/2; i += 16) {
            int channel = channel_base + i + get_sub_group_local_id();
            int token = tn + ln*ugemm_gateup_sg_tile_n + j;
            // Channels are an exact multiple of the workgroup tile.
            if (1) {
                int gv = TILE_AT(pair, i, j);
                int uv = TILE_AT(pair, i+ugemm_gateup_sg_tile_m/2, j);
                #if DEBUG_ACCUM
                debug_gate[token*channels+channel] = gv;
                debug_up[token*channels+channel] = uv;
                #endif
                float g = (convert_float(gv)*sub_group_broadcast(token_scale,j))*gate_scales[i/16];
                float u = (convert_float(uv)*sub_group_broadcast(token_scale,j))*up_scales[i/16];
                u = convert_float(convert_half_rte(u));
                float t = g*g;
                t = t*0.044715f;
                t = fma(t,g,g);
                t = t*(-2.0f*0.7978845f*1.442695f);
                float activation = native_recip(1.0f+native_exp2(t))*g;
                intel_sub_group_block_write_us(
                    (global ushort*)(y+token*channels+channel_base+i),
                    as_ushort(convert_half_rte(activation*u)));
            }
        }
        #endif
    }
}
