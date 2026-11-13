/**
 * @file
 * @brief Shared helpers for the Snitch vector operations (vector_*.h).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ssr.h"
#include "team_decls.h"
#include "types.h"

/**
 * @brief Check that a precision is one of FP64, FP32, FP16 or FP8.
 */
static inline bool snrt_vec_precision_valid(precision_t precision) {
    return precision == FP64 || precision == FP32 || precision == FP16 ||
           precision == FP8;
}

/**
 * @brief Number of elements of a given precision in one 64-bit word.
 * @note precision_t holds the element size in bytes (FP64 = 8, ..., FP8 = 1).
 */
static inline size_t snrt_vec_lanes(precision_t precision) {
    return sizeof(uint64_t) / (size_t)precision;
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
static inline void snrt_vec_split(size_t words, size_t core, size_t cores,
                                  size_t* first, size_t* count) {
    const size_t base = words / cores;
    const size_t rem = words % cores;
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
static inline void snrt_vec_stream_begin(const void* src, void* dst,
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
static inline void snrt_vec_stream_begin_2in(const void* src0,
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
static inline void snrt_vec_stream_end() {
    snrt_fpu_fence();
    snrt_ssr_disable();
}
