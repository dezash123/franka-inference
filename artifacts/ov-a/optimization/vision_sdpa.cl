// Vision attention for Pi0.5 SigLIP towers: BATCH x HEADS heads, QUERIES
// queries over KEYS keys, head size HEAD_DIM (72, padded to 80 in registers).
// One workgroup per (batch, head); subgroup s owns query rows s*8..s*8+7.
//   S  = Q K^T   : A = Q rows (2D block read), B = K rows (contiguous per lane)
//   O  = P V     : A = P (score layout), B = V via VNNI transform block read
// Q, K, V: [BATCH][HEADS][KEYS][HEAD_DIM] f16, out: same layout as Q.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#define SG_SIZE 16
#define K_STEPS ((HEAD_DIM + 15) / 16)
#define KEY_TILES (KEYS / 16)
#define ROW_TILES (QUERIES / 8)
#define STRIDE_BYTES (HEAD_DIM * 2)

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__attribute__((reqd_work_group_size(SG_SIZE * ROW_TILES, 1, 1)))
__kernel void vision_sdpa(__global const half* Q, __global const half* K,
                          __global const half* V, __global half* out) {
    const uint bh = get_group_id(0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const size_t head_off = (size_t)bh * KEYS * HEAD_DIM;
    __global const half* q = Q + head_off;
    __global const half* k = K + head_off;
    __global const half* v = V + head_off;
    const int row0 = (int)(sg * 8);

    // Q tile per k step: lane holds column (ks*16 + lane) of 8 rows; columns
    // past HEAD_DIM read as zero.
    short8 qa[K_STEPS];
    #pragma unroll
    for (uint ks = 0; ks < K_STEPS; ++ks) {
        ushort d[8];
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)q, STRIDE_BYTES, QUERIES, STRIDE_BYTES,
                                                  (int2)((int)(ks * 16), row0), d);
        qa[ks] = as_short8((ushort8)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
    }

    // Scores: lane holds key (kt*16 + lane) for the 8 rows. K chunks for the
    // next key tile are loaded while the current tile's dpas chain runs.
#define LOAD_K(kt, dst) { \
        __global const half* krow = k + ((kt) * 16 + lane) * HEAD_DIM; \
        _Pragma("unroll") for (uint ks = 0; ks < K_STEPS; ++ks) { \
            if (ks + 1 < K_STEPS || HEAD_DIM % 16 == 0) dst[ks] = *(__global const int8*)(krow + ks * 16); \
            else { half8 t8 = vload8(0, krow + ks * 16); dst[ks] = as_int8((half16)(t8, (half8)0.0h)); } \
        } }
    float8 s[KEY_TILES];
    float8 rowmax = (float8)(-INFINITY);
    int8 kb_a[K_STEPS], kb_b[K_STEPS];
    LOAD_K(0, kb_a);
    #pragma unroll
    for (uint kt = 0; kt < KEY_TILES; ++kt) {
        if (kt + 1 < KEY_TILES) { if (kt & 1) LOAD_K(kt + 1, kb_a) else LOAD_K(kt + 1, kb_b) }
        float8 acc = (float8)0.0f;
        #pragma unroll
        for (uint ks = 0; ks < K_STEPS; ++ks)
            acc = intel_sub_group_f16_f16_matrix_mad_k16(qa[ks], (kt & 1) ? kb_b[ks] : kb_a[ks], acc);
        acc *= (float)SCALE;
        s[kt] = acc;
        rowmax = fmax(rowmax, acc);
    }
    rowmax.s0 = sub_group_reduce_max(rowmax.s0); rowmax.s1 = sub_group_reduce_max(rowmax.s1);
    rowmax.s2 = sub_group_reduce_max(rowmax.s2); rowmax.s3 = sub_group_reduce_max(rowmax.s3);
    rowmax.s4 = sub_group_reduce_max(rowmax.s4); rowmax.s5 = sub_group_reduce_max(rowmax.s5);
    rowmax.s6 = sub_group_reduce_max(rowmax.s6); rowmax.s7 = sub_group_reduce_max(rowmax.s7);

    float8 rowsum = (float8)0.0f;
    short8 p[KEY_TILES];
    #pragma unroll
    for (uint kt = 0; kt < KEY_TILES; ++kt) {
        half8 ph = convert_half8(native_exp(s[kt] - rowmax));
        rowsum += convert_float8(ph);
        p[kt] = as_short8(ph);
    }
    rowsum.s0 = sub_group_reduce_add(rowsum.s0); rowsum.s1 = sub_group_reduce_add(rowsum.s1);
    rowsum.s2 = sub_group_reduce_add(rowsum.s2); rowsum.s3 = sub_group_reduce_add(rowsum.s3);
    rowsum.s4 = sub_group_reduce_add(rowsum.s4); rowsum.s5 = sub_group_reduce_add(rowsum.s5);
    rowsum.s6 = sub_group_reduce_add(rowsum.s6); rowsum.s7 = sub_group_reduce_add(rowsum.s7);
    const float8 inv = 1.0f / rowsum;

    // O[8 rows][dims]: lane holds dim (dt*16 + lane); V tiles via VNNI read,
    // double-buffered across key tiles.
#define LOAD_V(dt, kt, dst) intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void*)v, STRIDE_BYTES, KEYS, \
        STRIDE_BYTES, (int2)((int)((dt) * 16), (int)((kt) * 16)), dst)
    #pragma unroll
    for (uint dt = 0; dt < K_STEPS; ++dt) {
        float8 acc = (float8)0.0f;
        uint va[8], vb[8];
        LOAD_V(dt, 0, va);
        #pragma unroll
        for (uint kt = 0; kt < KEY_TILES; ++kt) {
            if (kt + 1 < KEY_TILES) { if (kt & 1) LOAD_V(dt, kt + 1, va); else LOAD_V(dt, kt + 1, vb); }
            int8 b = (kt & 1) ? as_int8((uint8)(vb[0], vb[1], vb[2], vb[3], vb[4], vb[5], vb[6], vb[7]))
                              : as_int8((uint8)(va[0], va[1], va[2], va[3], va[4], va[5], va[6], va[7]));
            acc = intel_sub_group_f16_f16_matrix_mad_k16(p[kt], b, acc);
        }
        acc *= inv;
        const uint dim = dt * 16 + lane;
        if (dim < HEAD_DIM) {
            __global half* o = out + head_off + (size_t)row0 * HEAD_DIM + dim;
            o[0 * HEAD_DIM] = convert_half(acc.s0); o[1 * HEAD_DIM] = convert_half(acc.s1);
            o[2 * HEAD_DIM] = convert_half(acc.s2); o[3 * HEAD_DIM] = convert_half(acc.s3);
            o[4 * HEAD_DIM] = convert_half(acc.s4); o[5 * HEAD_DIM] = convert_half(acc.s5);
            o[6 * HEAD_DIM] = convert_half(acc.s6); o[7 * HEAD_DIM] = convert_half(acc.s7);
        }
    }
}
