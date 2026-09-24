// Expert attention for Pi0.5 (MQA): HEADS=8 query heads x TOKENS=15 tokens
// over one shared K/V head with KEYS=983 keys and HEAD_DIM=256.
//
// Transposed formulation so that no cross-lane data movement is needed:
//   S^T[key][col]   = K[key][:] . Q[col][:]      (A = K rows via 2D block
//                                                 reads, B = Q^T columns held
//                                                 in registers)
//   O^T[dim][col]  += V^T[dim][key] * P^T[key][col]
// A workgroup handles two tokens (16 dpas columns = 2 tokens x 8 heads).
// Its 16 subgroups cover KEY_BLOCKS key blocks x DIM_SPLITS dim ranges; the
// key blocks are merged through local memory with a stable rescale.
//
// Q: [TOKENS][HEADS*HEAD_DIM] f16 (token-major), K, V: [KEYS][HEAD_DIM] f16
// mask: [TOKENS][KEYS] f16 additive, out: [HEADS][TOKENS][HEAD_DIM] f16.
// Workgroups beyond the attention ones stream `prefetch` into L2 and exit.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#define SG_SIZE 16
#define COLS 16                       // 2 tokens x 8 heads
#ifndef KEY_BLOCKS
#define KEY_BLOCKS 4
#endif
#define DIM_SPLITS (16 / KEY_BLOCKS)
#define KEYS_PER_BLOCK ((KEYS + KEY_BLOCKS * 16 - 1) / (KEY_BLOCKS * 16) * 16)
#define DIM_RANGE (HEAD_DIM / DIM_SPLITS)
#define DT_PER_RANGE (DIM_RANGE / 8)
#define K_STEPS (HEAD_DIM / 16)
#define ATTN_GROUPS ((TOKENS + 1) / 2)
#ifndef PREFETCH_GROUPS
#define PREFETCH_GROUPS 0
#endif

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__attribute__((reqd_work_group_size(SG_SIZE * 16, 1, 1)))
__kernel void expert_sdpa(__global const half* Q, __global const half* K,
                          __global const half* V, __global const half* mask,
                          __global half* out
#if PREFETCH_GROUPS
                          , __global const uint16* prefetch, uint prefetch_vec
#endif
                          ) {
    const uint group = get_group_id(0);
#if PREFETCH_GROUPS
    if (group >= ATTN_GROUPS) {
        const uint pg = group - ATTN_GROUPS;
        const uint lid = pg * get_local_size(0) + get_local_id(0);
        const uint stride = PREFETCH_GROUPS * get_local_size(0);
        uint16 acc = 0;
        for (uint i = lid; i < prefetch_vec; i += stride) acc += prefetch[i];
        if (acc.s0 == 0x9e3779b9u && acc.s1 == 0x7f4a7c15u) out[0] = 0.0h;
        return;
    }
#endif
    // Partials: [block][dim tile][col][8 dims] as half8 -> contiguous stores.
    __local half8 slm_acc[KEY_BLOCKS][HEAD_DIM / 8][COLS];    // 64 KB
    __local float slm_m[KEY_BLOCKS][COLS];
    __local float slm_l[KEY_BLOCKS][COLS];

    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint block = sg / DIM_SPLITS;
    const uint dsplit = sg % DIM_SPLITS;
    const uint tok0 = group * 2;
    const uint col_tok = tok0 + (lane >> 3);
    const uint col_head = lane & 7;
    // Columns past the last token compute on token 0 and are never stored.
    const uint safe_tok = col_tok < TOKENS ? col_tok : 0;
    __global const half* q_col = Q + safe_tok * (HEADS * HEAD_DIM) + col_head * HEAD_DIM;
    __global const half* mask_row = mask + safe_tok * KEYS;
    const uint key0 = block * KEYS_PER_BLOCK;
    const uint d0 = dsplit * DIM_RANGE;

    // Q^T column operands for every k step, resident for the whole key loop.
    int8 qb[K_STEPS];
    #pragma unroll
    for (uint ks = 0; ks < K_STEPS; ++ks) qb[ks] = *(__global const int8*)(q_col + ks * 16);

    float m = -INFINITY, l = 0.0f;
    float8 acc[DT_PER_RANGE];
    #pragma unroll
    for (uint dt = 0; dt < DT_PER_RANGE; ++dt) acc[dt] = (float8)0.0f;

    #pragma unroll 1
    for (uint kt = 0; kt < KEYS_PER_BLOCK / 16; ++kt) {
        const uint kbase = key0 + kt * 16;
        float8 s[2];
        #pragma unroll
        for (uint t = 0; t < 2; ++t) {
            // K tile: 8 key rows x 256 k as 8 two-block 2D reads; each lane
            // receives its k column across the 8 rows (dpas A layout). Rows
            // past KEYS read as zero.
            const int y = (int)(kbase + t * 8);
            float8 st = (float8)0.0f;
            #pragma unroll
            for (uint kb = 0; kb < K_STEPS / 2; ++kb) {
                ushort d[16];
                intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)K, HEAD_DIM * 2, KEYS,
                                                          HEAD_DIM * 2, (int2)((int)(kb * 32), y), d);
                short8 a0 = as_short8((ushort8)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
                short8 a1 = as_short8((ushort8)(d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]));
                st = intel_sub_group_f16_f16_matrix_mad_k16(a0, qb[2 * kb], st);
                st = intel_sub_group_f16_f16_matrix_mad_k16(a1, qb[2 * kb + 1], st);
            }
            // Mask: clamped index loads (no branches), then -inf past the end.
            const uint mk = kbase + t * 8;
#define ML(i) convert_float(mask_row[min(mk + i, (uint)(KEYS - 1))])
            float8 madd = (float8)(ML(0), ML(1), ML(2), ML(3), ML(4), ML(5), ML(6), ML(7));
#undef ML
            const int8 past = (int8)(mk + 0 >= KEYS, mk + 1 >= KEYS, mk + 2 >= KEYS, mk + 3 >= KEYS,
                                     mk + 4 >= KEYS, mk + 5 >= KEYS, mk + 6 >= KEYS, mk + 7 >= KEYS);
            madd = select(madd, (float8)(-INFINITY), -past);
            s[t] = st * (float)SCALE + madd;
        }
        // Online softmax per column (lane).
        const float8 mx8 = fmax(s[0], s[1]);
        const float mx = fmax(fmax(fmax(mx8.s0, mx8.s1), fmax(mx8.s2, mx8.s3)),
                              fmax(fmax(mx8.s4, mx8.s5), fmax(mx8.s6, mx8.s7)));
        const float m_new = fmax(m, mx);
        const float alpha = isinf(m_new) ? 1.0f : native_exp(m - m_new);
        const float8 base = (float8)(isinf(m_new) ? 0.0f : m_new);
        const half8 p0 = convert_half8(native_exp(s[0] - base));
        const half8 p1 = convert_half8(native_exp(s[1] - base));
        const float8 psum = convert_float8(p0) + convert_float8(p1);
        l = l * alpha + (psum.s0 + psum.s1 + psum.s2 + psum.s3 + psum.s4 + psum.s5 + psum.s6 + psum.s7);
        m = m_new;
        // B operand: lane = column, ints = key pairs (2j, 2j+1).
        int8 pb;
        pb.s0 = as_int((half2)(p0.s0, p0.s1)); pb.s1 = as_int((half2)(p0.s2, p0.s3));
        pb.s2 = as_int((half2)(p0.s4, p0.s5)); pb.s3 = as_int((half2)(p0.s6, p0.s7));
        pb.s4 = as_int((half2)(p1.s0, p1.s1)); pb.s5 = as_int((half2)(p1.s2, p1.s3));
        pb.s6 = as_int((half2)(p1.s4, p1.s5)); pb.s7 = as_int((half2)(p1.s6, p1.s7));
        // A operand: V^T tile, lane holds 8 dims of key (kbase + lane).
        const uint vkey = min(kbase + lane, (uint)(KEYS - 1));
        __global const half* vrow = V + vkey * HEAD_DIM + d0;
        #pragma unroll
        for (uint dt = 0; dt < DT_PER_RANGE; ++dt) {
            short8 va = as_short8(vload8(0, vrow + dt * 8));
            acc[dt] = intel_sub_group_f16_f16_matrix_mad_k16(va, pb, acc[dt] * alpha);
        }
    }
    // Publish this block's partial: one half8 store per dim tile.
    #pragma unroll
    for (uint dt = 0; dt < DT_PER_RANGE; ++dt)
        slm_acc[block][d0 / 8 + dt][lane] = convert_half8(acc[dt]);
    if (dsplit == 0) { slm_m[block][lane] = m; slm_l[block][lane] = l; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge key blocks: work-item w -> column w/16, dims (w%16)*16..+15.
    const uint w = get_local_id(0);
    const uint col = w >> 4, dg = w & 15;
    const uint otok = tok0 + (col >> 3), ohead = col & 7;
    if (otok >= TOKENS) return;
    float mtot = -INFINITY;
    #pragma unroll
    for (uint b = 0; b < KEY_BLOCKS; ++b) mtot = fmax(mtot, slm_m[b][col]);
    float ltot = 0.0f;
    float8 o0 = (float8)0.0f, o1 = (float8)0.0f;
    #pragma unroll
    for (uint b = 0; b < KEY_BLOCKS; ++b) {
        const float mb = slm_m[b][col];
        const float wb = isinf(mb) ? 0.0f : native_exp(mb - mtot);
        ltot += slm_l[b][col] * wb;
        o0 += convert_float8(slm_acc[b][dg * 2][col]) * wb;
        o1 += convert_float8(slm_acc[b][dg * 2 + 1][col]) * wb;
    }
    const float inv = 1.0f / ltot;
    __global half* dst = out + (ohead * TOKENS + otok) * HEAD_DIM + dg * 16;
    vstore8(convert_half8(o0 * inv), 0, dst);
    vstore8(convert_half8(o1 * inv), 0, dst + 8);
}
