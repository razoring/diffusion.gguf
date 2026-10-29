#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void get_scale_min_k4_dq(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q4_K * x = (const block_q4_K *) vx;
    const int il = iqs / 64;
    const int is = iqs / 32;
    const int shift = (iqs & 32) ? 4 : 0;
    const int q_idx = 32 * il + (iqs & 31);
    uint8_t sc, m;
    get_scale_min_k4_dq(is, x[ib].scales, sc, m);
    const float d = __half2float(__low2half(x[ib].dm)) * sc;
    const float min_val = __half2float(__high2half(x[ib].dm)) * m;
    v.x = d * ((x[ib].qs[q_idx + 0] >> shift) & 0xF) - min_val;
    v.y = d * ((x[ib].qs[q_idx + 1] >> shift) & 0xF) - min_val;
}

static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q5_K * x = (const block_q5_K *) vx;
    const int il = iqs / 64;
    const int is = iqs / 32;
    const int shift = (iqs & 32) ? 4 : 0;
    const int rem = iqs & 31;
    const int q_idx = 32 * il + rem;
    const uint8_t hm = 1 << is;
    uint8_t sc, m;
    get_scale_min_k4_dq(is, x[ib].scales, sc, m);
    const float d = __half2float(__low2half(x[ib].dm)) * sc;
    const float min_val = __half2float(__high2half(x[ib].dm)) * m;
    v.x = d * (((x[ib].qs[q_idx + 0] >> shift) & 0xF) + ((x[ib].qh[rem + 0] & hm) ? 16 : 0)) - min_val;
    v.y = d * (((x[ib].qs[q_idx + 1] >> shift) & 0xF) + ((x[ib].qh[rem + 1] & hm) ? 16 : 0)) - min_val;
}

static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q6_K * x = (const block_q6_K *) vx;
    const int ip = iqs / 128;
    const int rem = iqs & 127;
    const int qtr = rem / 32;
    const int il = rem & 31;
    const int ql_idx = 64 * ip + ((qtr & 1) ? 32 : 0) + il;
    const int ql_shift = (qtr >= 2) ? 4 : 0;
    const int qh_idx = 32 * ip + il;
    const int qh_shift = 2 * qtr;
    const float d_sc = __half2float(x[ib].d) * x[ib].scales[iqs / 16];
    const int8_t q0 = ((x[ib].ql[ql_idx + 0] >> ql_shift) & 0xF) | (((x[ib].qh[qh_idx + 0] >> qh_shift) & 3) << 4);
    const int8_t q1 = ((x[ib].ql[ql_idx + 1] >> ql_shift) & 0xF) | (((x[ib].qh[qh_idx + 1] >> qh_shift) & 3) << 4);
    v.x = d_sc * (q0 - 32);
    v.y = d_sc * (q1 - 32);

}
