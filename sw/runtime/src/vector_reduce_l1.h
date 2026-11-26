/**
 * @file
 * @brief Vector reductions on data that is already in L1 (TCDM): dot product.
 * @details Reduce vectors to one scalar. The functions here take pointers into
 *          the cluster's L1 memory and do no DMA transfer; they are the compute
 *          building blocks for the default (L3) versions, as in the other
 *          vector_*_l1.h files.
 *
 *          Each reduction comes in two versions:
 *          - snrt_vec_<op>_core_l1: runs on the calling core only;
 *          - snrt_vec_<op>_l1: every core of the cluster calls it; the work is
 *            split among the compute cores (word by word, interleaved: core c
 *            takes the words c, c + 8, ...), and their partial results are
 *            combined. Unlike the elementwise functions, it contains barriers,
 *            because the partial results must be combined.
 *
 *          Precision and accumulation: give them as compile-time constants
 *          (written at the call, or the parameters of an always-inlined caller
 *          that receives them). The functions are always inlined, so the code
 *          folds to that configuration. Values known only at run time work,
 *          but then the code of every configuration is inlined at the call:
 *          larger and slower. A caller with a run-time precision switches on
 *          it once, at the top of its kernel, and calls a version of the
 *          kernel with constants.
 *
 *          Accumulation precision. The dot product can accumulate in the
 *          precision of the data (SNRT_ACC_SAME), or in a wider one
 *          (SNRT_ACC_WIDE) using the expanding instructions of the hardware.
 *          Each expanding instruction widens by one step only (FP8 to FP16,
 *          FP16 to FP32), and there is nothing wider than FP64, so:
 *
 *          | data | SNRT_ACC_SAME | SNRT_ACC_WIDE                         |
 *          | ---- | ------------- | ------------------------------------- |
 *          | FP64 | FP64          | not supported                         |
 *          | FP32 | FP32          | not supported                         |
 *          | FP16 | FP16          | products accumulate in FP32; FP32     |
 *          | FP8  | FP8           | products accumulate in FP16; the lanes|
 *          |      |               | are combined in FP32; FP32            |
 *
 *          The result has the precision given by
 *          snrt_vec_dot_result_precision().
 *
 *          Order of summation. To reach one instruction per cycle, each core
 *          uses four independent accumulators (the latency of the multiply-add
 *          is four cycles, see sw/tests/src/fpu_latency.c), the lanes of a
 *          packed word are summed separately, and the cores' partial results
 *          are added as a balanced tree ((0+1)+(2+3))+((4+5)+(6+7)), three
 *          dependent additions instead of seven. Floating-point addition is
 *          not associative,
 *          so the result can differ in the last bits from a sequential sum, and
 *          it depends on the number of cores.
 *
 *          Whole words. A _padded variant of each multicore function takes
 *          the length in 64-bit words and has no code for a partial word; the
 *          caller pads the vectors (never done at run time).
 *
 *          Operations available: dot.
 */

#pragma once

#include "snitch_cluster_cfg.h"
#include "sync_decls.h"
#include "vector_common.h"

//==============================================================================
// Implementation
//==============================================================================

/**
 * @brief Precision in which the products are accumulated.
 */
typedef enum {
    SNRT_ACC_SAME,  ///< the precision of the data
    SNRT_ACC_WIDE   ///< the next wider precision, with expanding instructions
} snrt_acc_t;

/**
 * @brief Check that a dot product exists for a precision and accumulation
 *        mode (see the table at the top of the file).
 */
SNRT_VEC_INLINE bool snrt_vec_dot_supported(precision_t precision,
                                            snrt_acc_t acc) {
    if (!snrt_vec_precision_valid(precision)) return false;
    if (acc == SNRT_ACC_WIDE) return precision == FP16 || precision == FP8;
    return acc == SNRT_ACC_SAME;
}

/**
 * @brief Precision of the result of a dot product.
 * @pre snrt_vec_dot_supported(precision, acc).
 */
SNRT_VEC_INLINE precision_t snrt_vec_dot_result_precision(precision_t precision,
                                                        snrt_acc_t acc) {
    return acc == SNRT_ACC_WIDE ? FP32 : precision;
}

// Four independent accumulators, a0 ... a3, one per hardware-loop instruction.
// Expects `groups`, a0 ... a3 in the caller.
#define SNRT_DOT_GROUPS_ASM(instruction)                                    \
    asm volatile(                                                           \
        "frep.o %[n_frep], 4, 0, 0 \n"                                      \
        instruction " %[a0], ft0, ft1 \n"                                   \
        instruction " %[a1], ft0, ft1 \n"                                   \
        instruction " %[a2], ft0, ft1 \n"                                   \
        instruction " %[a3], ft0, ft1 \n"                                   \
        : [ a0 ] "+f"(a0), [ a1 ] "+f"(a1), [ a2 ] "+f"(a2), [ a3 ] "+f"(a3) \
        : [ n_frep ] "r"(groups - 1)                                        \
        : "ft0", "ft1", "ft2", "memory")

// The words that do not fill a group of four, one per iteration, into a0.
// Expects `rem`, a0 in the caller.
#define SNRT_DOT_REST_ASM(instruction)                                      \
    asm volatile(                                                           \
        "frep.o %[n_frep], 1, 0, 0 \n"                                      \
        instruction " %[a0], ft0, ft1 \n"                                   \
        : [ a0 ] "+f"(a0)                                                   \
        : [ n_frep ] "r"(rem - 1)                                           \
        : "ft0", "ft1", "ft2", "memory")

// One word, without streaming: a0 = instruction(x, y, a0). Expects `in0`,
// `in1`, a0 and the temporaries `x`, `y` in the caller.
#define SNRT_DOT_WORD_ASM(instruction)                                      \
    asm volatile(                                                           \
        "fld %[x], 0(%[in0]) \n"                                            \
        "fld %[y], 0(%[in1]) \n"                                            \
        instruction " %[a0], %[x], %[y] \n"                                 \
        : [ a0 ] "+f"(a0), [ x ] "=&f"(x), [ y ] "=&f"(y)                   \
        : [ in0 ] "r"(in0), [ in1 ] "r"(in1)                                \
        : "memory")

// FP64: the fused multiply-add accumulates into its fourth operand.
#define SNRT_DOT_GROUPS_FP64_ASM()                                          \
    asm volatile(                                                           \
        "frep.o %[n_frep], 4, 0, 0 \n"                                      \
        "fmadd.d %[a0], ft0, ft1, %[a0] \n"                                 \
        "fmadd.d %[a1], ft0, ft1, %[a1] \n"                                 \
        "fmadd.d %[a2], ft0, ft1, %[a2] \n"                                 \
        "fmadd.d %[a3], ft0, ft1, %[a3] \n"                                 \
        : [ a0 ] "+f"(a0), [ a1 ] "+f"(a1), [ a2 ] "+f"(a2), [ a3 ] "+f"(a3) \
        : [ n_frep ] "r"(groups - 1)                                        \
        : "ft0", "ft1", "ft2", "memory")

#define SNRT_DOT_REST_FP64_ASM()                                            \
    asm volatile(                                                           \
        "frep.o %[n_frep], 1, 0, 0 \n"                                      \
        "fmadd.d %[a0], ft0, ft1, %[a0] \n"                                 \
        : [ a0 ] "+f"(a0)                                                   \
        : [ n_frep ] "r"(rem - 1)                                           \
        : "ft0", "ft1", "ft2", "memory")

// dst = dst + src (the accumulators are combined in the accumulator format).
#define SNRT_DOT_ADD_ASM(instruction)                                       \
    asm volatile(instruction " %[d], %[d], %[s] \n"                         \
                 : [ d ] "+f"(dst)                                          \
                 : [ s ] "f"(src))

// d = instruction(d, s): the reductions of the lanes of a word (vfsum,
// vfsumex), where d is a zeroed accumulator.
#define SNRT_DOT_LANES_ASM(instruction)                                     \
    asm volatile(instruction " %[d], %[s] \n"                               \
                 : [ d ] "+f"(d)                                            \
                 : [ s ] "f"(s))

// Store the low element of a register, with a store of exactly its size.
#define SNRT_DOT_STORE_ASM(instruction)                                     \
    asm volatile(instruction " %[v], 0(%[p]) \n"                            \
                 :                                                          \
                 : [ v ] "f"(value), [ p ] "r"(result)                      \
                 : "memory")

/**
 * @brief Add two accumulators of the given accumulator precision:
 *        dst = dst + src, in every lane.
 */
SNRT_VEC_INLINE void snrt_dot_add(precision_t acc_precision, double* dst_p,
                                double src) {
    double dst = *dst_p;
    switch (acc_precision) {
        case FP64:
            SNRT_DOT_ADD_ASM("fadd.d");
            break;
        case FP32:
            SNRT_DOT_ADD_ASM("vfadd.s");
            break;
        case FP16:
            SNRT_DOT_ADD_ASM("vfadd.h");
            break;
        case FP8:
            SNRT_DOT_ADD_ASM("vfadd.b");
            break;
        default:
            break;
    }
    *dst_p = dst;
}

/**
 * @brief Store the element in the low bits of a register, with a store of
 *        exactly the size of the element.
 * @param result Pointer to the destination, naturally aligned.
 */
SNRT_VEC_INLINE void snrt_dot_store(precision_t precision, void* result,
                                  double value) {
    switch (precision) {
        case FP64:
            SNRT_DOT_STORE_ASM("fsd");
            break;
        case FP32:
            SNRT_DOT_STORE_ASM("fsw");
            break;
        case FP16:
            SNRT_DOT_STORE_ASM("fsh");
            break;
        case FP8:
            SNRT_DOT_STORE_ASM("fsb");
            break;
        default:
            break;
    }
}

/**
 * @brief Sum the lanes of an accumulator word into one element, in the low
 *        bits of the returned register.
 * @details Each vfsum adds adjacent lanes (the result lane j is the sum of
 *          lanes 2j and 2j + 1), so a word of 2^k lanes takes k steps. The
 *          expanding sum, used for FP8 with wide accumulation, goes from FP16
 *          lanes to FP32 lanes, and one more step finishes it.
 */
SNRT_VEC_INLINE double snrt_dot_lanes(precision_t precision, snrt_acc_t acc,
                                    double a0) {
    double d = 0.0, s = a0;
    if (precision == FP64) return a0;
    if (acc == SNRT_ACC_WIDE && precision == FP8) {
        SNRT_DOT_LANES_ASM("vfsumex.s.h");  // 4 FP16 lanes -> 2 FP32 lanes
        s = d;
        d = 0.0;
        SNRT_DOT_LANES_ASM("vfsum.s");
        return d;
    }
    switch (acc == SNRT_ACC_WIDE ? FP32 : precision) {
        case FP32:  // 2 lanes
            SNRT_DOT_LANES_ASM("vfsum.s");
            break;
        case FP16:  // 4 lanes
            SNRT_DOT_LANES_ASM("vfsum.h");
            s = d;
            d = 0.0;
            SNRT_DOT_LANES_ASM("vfsum.h");
            break;
        case FP8:  // 8 lanes
            SNRT_DOT_LANES_ASM("vfsum.b");
            s = d;
            d = 0.0;
            SNRT_DOT_LANES_ASM("vfsum.b");
            s = d;
            d = 0.0;
            SNRT_DOT_LANES_ASM("vfsum.b");
            break;
        default:
            break;
    }
    return d;
}

/**
 * @brief Hardware loops of the dot product over two streams, into the four
 *        accumulators.
 */
SNRT_VEC_INLINE void snrt_dot_stream(precision_t precision, snrt_acc_t acc,
                                   size_t words, double* a0_p, double* a1_p,
                                   double* a2_p, double* a3_p) {
    double a0 = *a0_p, a1 = *a1_p, a2 = *a2_p, a3 = *a3_p;
    const size_t groups = words / 4;
    const size_t rem = words % 4;
    if (precision == FP64) {
        if (groups > 0) SNRT_DOT_GROUPS_FP64_ASM();
        if (rem > 0) SNRT_DOT_REST_FP64_ASM();
    } else if (acc == SNRT_ACC_WIDE && precision == FP16) {
        if (groups > 0) SNRT_DOT_GROUPS_ASM("vfdotpex.s.h");
        if (rem > 0) SNRT_DOT_REST_ASM("vfdotpex.s.h");
    } else if (acc == SNRT_ACC_WIDE && precision == FP8) {
        if (groups > 0) SNRT_DOT_GROUPS_ASM("vfdotpex.h.b");
        if (rem > 0) SNRT_DOT_REST_ASM("vfdotpex.h.b");
    } else if (precision == FP32) {
        if (groups > 0) SNRT_DOT_GROUPS_ASM("vfmac.s");
        if (rem > 0) SNRT_DOT_REST_ASM("vfmac.s");
    } else if (precision == FP16) {
        if (groups > 0) SNRT_DOT_GROUPS_ASM("vfmac.h");
        if (rem > 0) SNRT_DOT_REST_ASM("vfmac.h");
    } else {
        if (groups > 0) SNRT_DOT_GROUPS_ASM("vfmac.b");
        if (rem > 0) SNRT_DOT_REST_ASM("vfmac.b");
    }
    *a0_p = a0;
    *a1_p = a1;
    *a2_p = a2;
    *a3_p = a3;
}

/**
 * @brief One word, without streaming: a0 = a0 + (in0 . in1) in the lanes.
 */
SNRT_VEC_INLINE void snrt_dot_word(precision_t precision, snrt_acc_t acc,
                                 const void* in0, const void* in1,
                                 double* a0_p) {
    double a0 = *a0_p, x, y;
    if (acc == SNRT_ACC_WIDE && precision == FP16) {
        SNRT_DOT_WORD_ASM("vfdotpex.s.h");
    } else if (acc == SNRT_ACC_WIDE && precision == FP8) {
        SNRT_DOT_WORD_ASM("vfdotpex.h.b");
    } else if (precision == FP32) {
        SNRT_DOT_WORD_ASM("vfmac.s");
    } else if (precision == FP16) {
        SNRT_DOT_WORD_ASM("vfmac.h");
    } else if (precision == FP8) {
        SNRT_DOT_WORD_ASM("vfmac.b");
    }
    *a0_p = a0;
}

#undef SNRT_DOT_GROUPS_ASM
#undef SNRT_DOT_REST_ASM
#undef SNRT_DOT_WORD_ASM
#undef SNRT_DOT_GROUPS_FP64_ASM
#undef SNRT_DOT_REST_FP64_ASM

//==============================================================================
// Public functions
//==============================================================================

/**
 * @brief Number of bytes of scratch memory that snrt_vec_dot_l1() needs.
 */
SNRT_VEC_INLINE size_t snrt_vec_reduce_scratch_bytes() {
    return SNRT_VEC_COMPUTE_CORES * sizeof(uint64_t);
}

SNRT_VEC_INLINE bool snrt_dot_core_impl(const void* a, const void* b,
                                        void* result, size_t n,
                                        precision_t precision,
                                        snrt_acc_t acc) {
    if (!snrt_vec_dot_supported(precision, acc)) return false;

    const size_t words = n >> snrt_vec_lanes_log2(precision);
    const size_t tail = n & (snrt_vec_lanes(precision) - 1);

    // Accumulator precision: the data, or the next wider one.
    const precision_t acc_precision =
        (acc == SNRT_ACC_WIDE) ? (precision == FP16 ? FP32 : FP16) : precision;

    // Zero is all-zero bits in every format.
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

    // Whole words, streamed.
    if (words > 0) {
        snrt_vec_stream_begin_2read(a, b, words);
        snrt_dot_stream(precision, acc, words, &a0, &a1, &a2, &a3);
        snrt_vec_stream_end();
    }

    // Remaining elements, through zero-padded words (their products are zero).
    if (tail > 0) {
        uint64_t in0, in1;
        const size_t bytes = tail * (size_t)precision;
        const size_t offset = words * sizeof(uint64_t);
        snrt_vec_tail_load(&in0, (const uint8_t*)a + offset, bytes);
        snrt_vec_tail_load(&in1, (const uint8_t*)b + offset, bytes);
        snrt_dot_word(precision, acc, &in0, &in1, &a0);
    }

    // Combine the four accumulators, then the lanes.
    snrt_dot_add(acc_precision, &a0, a1);
    snrt_dot_add(acc_precision, &a2, a3);
    snrt_dot_add(acc_precision, &a0, a2);
    const double value = snrt_dot_lanes(precision, acc, a0);

    snrt_dot_store(snrt_vec_dot_result_precision(precision, acc), result,
                   value);
    snrt_fpu_fence();
    return true;
}

/**
 * @brief Dot product of two vectors in L1, on the calling core only.
 * @details result = sum of a[i] * b[i] for i in [0, n). Whole 64-bit words are
 *          processed by an SSR/FREP loop with four independent accumulators
 *          (FP64: fused multiply-add; FP32, FP16, FP8: vfmac; wide
 *          accumulation: vfdotpex). Elements that do not fill a whole word are
 *          processed through zero-padded temporary words, so no memory outside
 *          the n elements is read. The lanes and the accumulators are then
 *          combined.
 * @param a Pointer to the first element of the first vector.
 * @param b Pointer to the first element of the second vector.
 * @param result Pointer where the result is written: one element, naturally
 *               aligned, of precision snrt_vec_dot_result_precision(). Only
 *               the bytes of that element are written.
 * @param n Number of elements (not bytes). n = 0 gives zero.
 * @param precision Element precision of a and b: FP64, FP32, FP16 or FP8.
 * @param acc Accumulation precision, see the table at the top of the file.
 * @return True if snrt_vec_dot_supported(precision, acc); false otherwise,
 *         without accessing the buffers.
 * @pre a and b point into L1 (TCDM), are eight-byte aligned and cover n
 *      elements. result points into L1 and may be any natural alignment.
 * @pre SSR is disabled, and the calling core supports the selected
 *      instructions.
 * @note The current FP rounding mode is used. The sum order is described at
 *       the top of the file.
 * @note The FP pipeline is drained before returning, so the result is stored
 *       when the call returns.
 */
SNRT_VEC_INLINE bool snrt_vec_dot_core_l1(const void* a, const void* b,
                                          void* result, size_t n,
                                          precision_t precision,
                                          snrt_acc_t acc) {
    return snrt_dot_core_impl(a, b, result, n, precision, acc);
}

/**
 * @brief Add the eight partial results of the cores and store the sum.
 * @details A tree: the eight loads and the four additions of the first level
 *          are independent, so the FPU pipeline stays full, and only three
 *          additions are dependent (a chain of eight would wait four cycles
 *          for each). The partials are in the low lane of zero-padded words, so
 *          a lane-wise add is a scalar add. Waits for the FPU, so the result is
 *          stored on return.
 * @param slots The eight partial results, one 64-bit word each, in core order.
 * @param precision Precision of the partial results and of the sum.
 * @param result Where the sum is written (one element of that precision).
 */
SNRT_VEC_INLINE void snrt_dot_combine8(const uint64_t* slots,
                                       precision_t precision, void* result) {
    double p0, p1, p2, p3, p4, p5, p6, p7;
    asm volatile(
        "fld %0, 0(%8) \n fld %1, 8(%8) \n fld %2, 16(%8) \n"
        "fld %3, 24(%8) \n fld %4, 32(%8) \n fld %5, 40(%8) \n"
        "fld %6, 48(%8) \n fld %7, 56(%8) \n"
        : "=&f"(p0), "=&f"(p1), "=&f"(p2), "=&f"(p3), "=&f"(p4), "=&f"(p5),
          "=&f"(p6), "=&f"(p7)
        : "r"(slots)
        : "memory");
    snrt_dot_add(precision, &p0, p1);
    snrt_dot_add(precision, &p2, p3);
    snrt_dot_add(precision, &p4, p5);
    snrt_dot_add(precision, &p6, p7);
    snrt_dot_add(precision, &p0, p2);
    snrt_dot_add(precision, &p4, p6);
    snrt_dot_add(precision, &p0, p4);
    snrt_dot_store(precision, result, p0);
    snrt_fpu_fence();
}

static_assert(SNRT_VEC_COMPUTE_CORES == 8,
              "the combine function is written for eight cores");

/**
 * @brief Dot product of `words` whole words that are `stride_bytes` apart,
 *        plus an optional partial word, on the calling core.
 * @param a First word of the first vector's part.
 * @param b First word of the second vector's part.
 * @param result Where the result is written (see snrt_vec_dot_core_l1()).
 * @param words Number of whole words (may be 0).
 * @param stride_bytes Distance in bytes between two consecutive words (a
 *        multiple of 8).
 * @param tail_a Start of the partial word of the first vector.
 * @param tail_b Start of the partial word of the second vector.
 * @param tail_bytes Bytes of the partial word that exist (0 to 7; 0 for none).
 */
SNRT_VEC_INLINE bool snrt_dot_strided_impl(
    const void* a, const void* b, void* result, size_t words,
    size_t stride_bytes, const void* tail_a, const void* tail_b,
    size_t tail_bytes, precision_t precision, snrt_acc_t acc) {
    if (!snrt_vec_dot_supported(precision, acc)) return false;

    const precision_t acc_precision =
        (acc == SNRT_ACC_WIDE) ? (precision == FP16 ? FP32 : FP16) : precision;

    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

    if (words > 0) {
        snrt_vec_stream_begin_2read_strided(a, b, words, stride_bytes);
        snrt_dot_stream(precision, acc, words, &a0, &a1, &a2, &a3);
        snrt_vec_stream_end();
    }

    if (tail_bytes > 0) {
        uint64_t in0, in1;
        snrt_vec_tail_load(&in0, tail_a, tail_bytes);
        snrt_vec_tail_load(&in1, tail_b, tail_bytes);
        snrt_dot_word(precision, acc, &in0, &in1, &a0);
    }

    snrt_dot_add(acc_precision, &a0, a1);
    snrt_dot_add(acc_precision, &a2, a3);
    snrt_dot_add(acc_precision, &a0, a2);
    const double value = snrt_dot_lanes(precision, acc, a0);

    snrt_dot_store(snrt_vec_dot_result_precision(precision, acc), result,
                   value);
    snrt_fpu_fence();
    return true;
}

SNRT_VEC_INLINE bool snrt_dot_multi_impl(const void* a, const void* b,
                                            void* result, void* scratch,
                                            size_t n, precision_t precision,
                                            snrt_acc_t acc,
                                            size_t cores_log2) {
    if (!snrt_vec_dot_supported(precision, acc)) return false;
    if (cores_log2 != 0 && cores_log2 != 3) return false;

    const precision_t result_precision =
        snrt_vec_dot_result_precision(precision, acc);
    const size_t lanes = snrt_vec_lanes(precision);
    const size_t cores = (size_t)1 << cores_log2;
    const size_t core = snrt_vec_core_idx();
    uint64_t* slots = (uint64_t*)scratch;

    // One core: contiguous, no split.
    if (cores_log2 == 0) {
        if (core == 0) snrt_dot_core_impl(a, b, result, n, precision, acc);
        snrt_cluster_hw_barrier();
        return true;
    }

    if (core < cores) {
        const size_t words = n >> snrt_vec_lanes_log2(precision);
        const size_t tail_elems = n & (lanes - 1);
        // The words core, core + cores, ... that are below `words`.
        const size_t count =
            words > core ? (words - core + cores - 1) >> cores_log2 : 0;
        // The partial word has the index `words`: its owner is the core whose
        // turn it is.
        const size_t owner = words & (cores - 1);
        const size_t tail_bytes =
            core == owner ? tail_elems * (size_t)precision : 0;
        const size_t tail_offset = words * sizeof(uint64_t);

        ((volatile uint32_t*)&slots[core])[0] = 0;
        ((volatile uint32_t*)&slots[core])[1] = 0;
        snrt_dot_strided_impl(
            (const uint8_t*)a + core * sizeof(uint64_t),
            (const uint8_t*)b + core * sizeof(uint64_t), &slots[core], count,
            cores * sizeof(uint64_t), (const uint8_t*)a + tail_offset,
            (const uint8_t*)b + tail_offset, tail_bytes, precision, acc);
    }
    snrt_cluster_hw_barrier();

    if (core == 0) snrt_dot_combine8(slots, result_precision, result);
    snrt_cluster_hw_barrier();
    return true;
}

/**
 * @brief The multicore dot product of snrt_dot_multi_impl() for vectors
 *        that are a whole number of 64-bit words: no partial word.
 * @details The length is given in words, so there is no split of the length
 *          into words and a remainder, and no code for a partial word. The
 *          precision and the mode are arguments as in the other function.
 */
SNRT_VEC_INLINE bool snrt_dot_padded_multi_impl(
    const void* a, const void* b, void* result, void* scratch, size_t words,
    precision_t precision, snrt_acc_t acc, size_t cores_log2) {
    if (!snrt_vec_dot_supported(precision, acc)) return false;
    if (cores_log2 != 0 && cores_log2 != 3) return false;

    const precision_t result_precision =
        snrt_vec_dot_result_precision(precision, acc);
    const size_t cores = (size_t)1 << cores_log2;
    const size_t core = snrt_vec_core_idx();
    uint64_t* slots = (uint64_t*)scratch;

    if (cores_log2 == 0) {
        if (core == 0)
            snrt_dot_strided_impl(a, b, result, words, sizeof(uint64_t), a, b,
                                  0, precision, acc);
        snrt_cluster_hw_barrier();
        return true;
    }

    if (core < cores) {
        const size_t count =
            words > core ? (words - core + cores - 1) >> cores_log2 : 0;
        ((volatile uint32_t*)&slots[core])[0] = 0;
        ((volatile uint32_t*)&slots[core])[1] = 0;
        snrt_dot_strided_impl((const uint8_t*)a + core * sizeof(uint64_t),
                              (const uint8_t*)b + core * sizeof(uint64_t),
                              &slots[core], count, cores * sizeof(uint64_t), a,
                              b, 0, precision, acc);
    }
    snrt_cluster_hw_barrier();

    if (core == 0) snrt_dot_combine8(slots, result_precision, result);
    snrt_cluster_hw_barrier();
    return true;
}


/**
 * @brief Dot product of two vectors in L1, across 2^cores_log2 of the compute
 *        cores of the cluster.
 * @details Interleaved split: with C cores, core c takes the whole 64-bit
 *          words c, c + C, c + 2C, ... (a stream with a stride of C words,
 *          starting at word c), so the work is balanced to within one word.
 *          The cores use disjoint sets of L1 banks, so two cores never request
 *          the same bank in the same cycle, whatever their timing (with
 *          contiguous slices, cores in lockstep whose slices start in the same
 *          bank lose about a quarter of the time; see
 *          paper_notes/BANK_CONFLICTS.md). The elements that do not fill a
 *          whole word are the partial word with index n_words; the core whose
 *          turn it is (n_words mod C) processes it. With one core, core 0 runs
 *          the whole vector. Each core stores its partial result in its own
 *          slot of the scratch memory; after a barrier, core 0 adds the
 *          partial results as a tree and writes the result; a second barrier
 *          makes it visible to every core. With one core, core 0 writes the
 *          result directly, and there is one barrier. The other cores only
 *          take part in the barriers.
 * @param a Pointer to the first element of the first vector.
 * @param b Pointer to the first element of the second vector.
 * @param result Pointer where the result is written, as in
 *               snrt_vec_dot_core_l1(). Written by core 0.
 * @param scratch Pointer to at least snrt_vec_reduce_scratch_bytes() bytes of
 *                L1 memory, eight-byte aligned, for the partial results. Its
 *                contents are overwritten.
 * @param n Number of elements (not bytes).
 * @param precision Element precision of a and b: FP64, FP32, FP16 or FP8.
 * @param acc Accumulation precision, see the table at the top of the file.
 * @param cores_log2 Base-2 logarithm of the number of cores that compute:
 *                   0 (core 0 alone) or 3 (the eight compute cores). Two
 *                   and four cores are not offered: they were measured and
 *                   never pay (paper_notes/PENDING.md).
 * @return True if snrt_vec_dot_supported(precision, acc) and cores_log2 is 0
 *         or 3; false otherwise, without accessing the buffers and without
 *         any barrier (every core sees the same result, so they all return
 *         early).
 * @pre Every core of the cluster calls this with the same arguments, the DMA
 *      core included, as it takes part in the barriers (it runs no FP
 *      instruction).
 * @pre Same pointer and mode preconditions as snrt_vec_dot_core_l1().
 * @pre For the best speed with several cores, the distance between the two
 *      vectors (b - a, in words) should be a multiple of 8, ideally 16: a
 *      stream of one vector of a core can still meet a stream of the other
 *      vector of another core otherwise (see BANK_CONFLICTS.md).
 * @note When it returns, the result is available to every core.
 */
SNRT_VEC_INLINE bool snrt_vec_dot_cores_l1(const void* a, const void* b,
                                           void* result, void* scratch,
                                           size_t n, precision_t precision,
                                           snrt_acc_t acc, size_t cores_log2) {
    return snrt_dot_multi_impl(a, b, result, scratch, n, precision, acc,
                               cores_log2);
}

/**
 * @brief Dot product of two vectors in L1, across all the compute cores of the
 *        cluster: snrt_vec_dot_cores_l1() with eight cores.
 */
SNRT_VEC_INLINE bool snrt_vec_dot_l1(const void* a, const void* b, void* result,
                                     void* scratch, size_t n,
                                     precision_t precision, snrt_acc_t acc) {
    return snrt_vec_dot_cores_l1(a, b, result, scratch, n, precision, acc, 3);
}

/**
 * @brief snrt_vec_dot_cores_l1() for vectors of a whole number of 64-bit
 *        words, with the length given in WORDS (n_elements = words * lanes).
 * @details There is no partial word, so no code for it: about 5 to 25 cycles
 *          less per call on one core (none for FP64) and 6 to 9 on eight, and
 *          a smaller function. The caller pads the vectors (zero in at least
 *          one of them) when the length is not a whole number of words: the
 *          function never pads at run time.
 */
SNRT_VEC_INLINE bool snrt_vec_dot_padded_cores_l1(
    const void* a, const void* b, void* result, void* scratch, size_t words,
    precision_t precision, snrt_acc_t acc, size_t cores_log2) {
    return snrt_dot_padded_multi_impl(a, b, result, scratch, words, precision,
                                      acc, cores_log2);
}

/**
 * @brief snrt_vec_dot_padded_cores_l1() with all eight compute cores.
 */
SNRT_VEC_INLINE bool snrt_vec_dot_padded_l1(const void* a, const void* b,
                                            void* result, void* scratch,
                                            size_t words, precision_t precision,
                                            snrt_acc_t acc) {
    return snrt_vec_dot_padded_cores_l1(a, b, result, scratch, words,
                                        precision, acc, 3);
}
