/**
 * @file
 * @brief Shared helpers for the Snitch vector operations (vector_*.h).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "snitch_cluster_cfg.h"
#include "ssr.h"
#include "team_decls.h"
#include "types.h"

/**
 * @brief Inlining of the helpers and of the dot product, always.
 * @details When the precision and the mode are constants at the call site, the
 *          choices made on them in the code disappear, and the result is as
 *          good as a function written for that precision. When they are
 *          run-time values, the code is inlined with the choices in it.
 */
#define SNRT_VEC_INLINE static inline __attribute__((always_inline))

/**
 * @brief Number of compute cores of the cluster, as a compile-time constant,
 *        from the cluster configuration. Same value as
 *        snrt_cluster_compute_core_num().
 * @details The runtime functions (snrt_cluster_core_idx() and the like) are
 *          defined inline in team.h, which snrt.h includes. This header does
 *          not include snrt.h (the runtime headers avoid including it, to
 *          prevent circular includes), so it only sees their declarations when
 *          it is included first; then each use would be a call to the
 *          out-of-line copy, and the core count would not be a constant. The
 *          helpers below do not depend on the include order.
 */
#define SNRT_VEC_COMPUTE_CORES \
    (SNRT_CLUSTER_CORE_NUM - SNRT_CLUSTER_DM_CORE_NUM)

/**
 * @brief Index of the calling core within the cluster (0 ... cores - 1),
 *        read from the hart id. Same value as snrt_cluster_core_idx().
 * @details With a single cluster, the index is the hart id minus the first
 *          hart id; with several clusters, it is taken modulo the number of
 *          cores of a cluster (a constant, so a multiplication, not a
 *          division).
 */
SNRT_VEC_INLINE size_t snrt_vec_core_idx() {
    uint32_t hartid;
    asm("csrr %0, mhartid" : "=r"(hartid));
#if SNRT_CLUSTER_NUM == 1
    return hartid - SNRT_BASE_HARTID;
#else
    return (hartid - SNRT_BASE_HARTID) % SNRT_CLUSTER_CORE_NUM;
#endif
}

/**
 * @brief True on the compute cores (not on the DMA core). Same value as
 *        snrt_is_compute_core().
 */
SNRT_VEC_INLINE bool snrt_vec_is_compute_core() {
    return snrt_vec_core_idx() < SNRT_VEC_COMPUTE_CORES;
}

/**
 * @brief Check that a precision is one of FP64, FP32, FP16 or FP8.
 */
SNRT_VEC_INLINE bool snrt_vec_precision_valid(precision_t precision) {
    return precision == FP64 || precision == FP32 || precision == FP16 ||
           precision == FP8;
}

/**
 * @brief Base-2 logarithm of snrt_vec_lanes(): FP64 = 0, FP32 = 1, FP16 = 2,
 *        FP8 = 3.
 * @details Comparisons, no table: constant data is linked into L3, and a load
 *          from it stalls about ten cycles. Folds to a constant when the
 *          precision is known at compile time.
 * @pre The precision is valid (see snrt_vec_precision_valid()).
 */
SNRT_VEC_INLINE unsigned snrt_vec_lanes_log2(precision_t precision) {
    return precision == FP64 ? 0u : precision == FP32 ? 1u
                                  : precision == FP16 ? 2u : 3u;
}

/**
 * @brief Number of elements of a given precision in one 64-bit word: 1, 2, 4
 *        or 8. A power of two, so a vector of n elements has
 *        n >> snrt_vec_lanes_log2() whole words and n & (lanes - 1) elements
 *        left over, without any division.
 * @pre The precision is valid (see snrt_vec_precision_valid()).
 */
SNRT_VEC_INLINE size_t snrt_vec_lanes(precision_t precision) {
    return (size_t)1 << snrt_vec_lanes_log2(precision);
}

/**
 * @brief Split @p words 64-bit words among @p cores cores, in order.
 * @details Core i gets a contiguous range. The first (words % cores) cores get
 *          one word more than the others, so all words are covered and the
 *          ranges differ in size by at most one word.
 * @param words Total number of words.
 * @param core Index of this core among the participating cores.
 * @param cores Number of participating cores.
 * @param first Output: index of the first word assigned to this core.
 * @param count Output: number of words assigned to this core (may be zero).
 */
SNRT_VEC_INLINE void snrt_vec_split(size_t words, size_t core, size_t cores,
                                  size_t* first, size_t* count) {
    const size_t base = words / cores;
    const size_t rem = words % cores;
    *count = base + (core < rem ? 1 : 0);
    *first = core * base + (core < rem ? core : rem);
}

/**
 * @brief snrt_vec_split() for a number of cores that is a power of two, given
 *        as its base-2 logarithm: shifts and masks, no division.
 * @param words Number of words to divide.
 * @param core Index of the calling core, in [0, 2^cores_log2).
 * @param cores_log2 Base-2 logarithm of the number of cores.
 * @param first Output: index of the first word of the core.
 * @param count Output: number of words of the core.
 */
SNRT_VEC_INLINE void snrt_vec_split_log2(size_t words, size_t core,
                                         size_t cores_log2, size_t* first,
                                         size_t* count) {
    const size_t base = words >> cores_log2;
    const size_t rem = words & (((size_t)1 << cores_log2) - 1);
    *count = base + (core < rem ? 1 : 0);
    *first = core * base + (core < rem ? core : rem);
}

/**
 * @brief Start streaming: SSR data mover 0 reads @p words words from @p src
 *        and data mover 1 writes @p words words to @p dst.
 * @details The streams are consumed by a hardware loop (frep) that reads ft0
 *          and writes ft1. Call snrt_vec_stream_end() afterwards.
 * @pre words > 0 and both pointers are eight-byte aligned.
 */
SNRT_VEC_INLINE void snrt_vec_stream_begin(const void* src, void* dst,
                                         size_t words) {
    snrt_ssr_loop_1d(SNRT_SSR_DM0, words, sizeof(uint64_t));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, words, sizeof(uint64_t));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)src);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, dst);
    snrt_ssr_enable();
}

/**
 * @brief Start streaming: SSR data movers 0 and 1 read @p words words from
 *        @p src0 and @p src1, and data mover 2 writes @p words words to
 *        @p dst.
 * @details The streams are consumed by a hardware loop (frep) that reads ft0
 *          and ft1 and writes ft2. Call snrt_vec_stream_end() afterwards.
 * @pre words > 0 and all the pointers are eight-byte aligned.
 */
SNRT_VEC_INLINE void snrt_vec_stream_begin_2in(const void* src0,
                                             const void* src1, void* dst,
                                             size_t words) {
    snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, words, sizeof(uint64_t));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)src0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)src1);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst);
    snrt_ssr_enable();
}

/**
 * @brief Wait for the FP pipeline to drain and turn the streams off.
 */
SNRT_VEC_INLINE void snrt_vec_stream_end() {
    snrt_fpu_fence();
    snrt_ssr_disable();
}
