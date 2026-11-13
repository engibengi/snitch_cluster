/**
 * @file
 * @brief Vector-vector operations on data that is already in L1 (TCDM).
 * @details Combine two vectors element by element: c[i] = a[i] <op> b[i]. The
 *          functions here take pointers into the cluster's L1 memory and do no
 *          DMA transfer. They are the compute building blocks: the default
 *          vector-vector functions (without the _l1 suffix, in
 *          vector_vector.h) take pointers anywhere, and run these on tiles
 *          they bring into L1.
 *
 *          Each operation comes in two versions:
 *          - snrt_vv_<op>_core_l1: runs on the calling core only;
 *          - snrt_vv_<op>_l1: every compute core of the cluster calls it, and
 *            the work is split among them.
 *
 *          The code has the same two parts as vector_scalar_l1.h: an
 *          implementation shared by all the operations, which differ only in
 *          the instruction they use, and the public functions, one pair per
 *          operation, each a one-line call into the shared code.
 *
 *          Operations available: add.
 */

#pragma once

#include <string.h>

#include "snitch_cluster_cfg.h"
#include "vector_common.h"

//==============================================================================
// Implementation, shared by all the operations
//==============================================================================

typedef enum { SNRT_VV_ADD } snrt_vv_op_t;

// Hardware loop over three streams: ft2 = instruction(ft0, ft1). ft0 and ft1
// stream the two source vectors and ft2 streams the destination.
// Expects `words` in the caller.
#define SNRT_VV_STREAM_ASM(instruction)                    \
    asm volatile(                                          \
        "frep.o %[n_frep], 1, 0, 0 \n"                     \
        instruction " ft2, ft0, ft1 \n"                    \
        :                                                  \
        : [ n_frep ] "r"(words - 1)                        \
        : "ft0", "ft1", "ft2", "memory")

// One word, without streaming: out = instruction(in0, in1). Expects `in0`,
// `in1`, `out`, and the temporaries `x`, `y` and `o` in the caller.
#define SNRT_VV_WORD_ASM(instruction)                      \
    asm volatile(                                          \
        "fld %[x], 0(%[in0]) \n"                           \
        "fld %[y], 0(%[in1]) \n"                           \
        instruction " %[o], %[x], %[y] \n"                 \
        "fsd %[o], 0(%[out]) \n"                           \
        : [ x ] "=&f"(x), [ y ] "=&f"(y), [ o ] "=&f"(o)   \
        : [ in0 ] "r"(in0), [ in1 ] "r"(in1), [ out ] "r"(out) \
        : "memory")

// Expands to a switch on `precision` (in the caller) that applies the macro ASM
// to the instruction of that precision: instr_64, instr_32, instr_16, instr_8.
#define SNRT_VV_BY_PRECISION(ASM, instr_64, instr_32, instr_16, instr_8) \
    switch (precision) {                                                 \
        case FP64:                                                       \
            ASM(instr_64);                                               \
            break;                                                       \
        case FP32:                                                       \
            ASM(instr_32);                                               \
            break;                                                       \
        case FP16:                                                       \
            ASM(instr_16);                                               \
            break;                                                       \
        case FP8:                                                        \
            ASM(instr_8);                                                \
            break;                                                       \
        default:                                                         \
            break;                                                       \
    }

/**
 * @brief Check that an operation exists for a precision.
 * @details Every operation exists for every precision, for now.
 */
static inline bool snrt_vv_supported(snrt_vv_op_t op, precision_t precision) {
    (void)op;
    return snrt_vec_precision_valid(precision);
}

/**
 * @brief Run the hardware loop of an operation over the streams.
 * @details This and snrt_vv_word() hold the instruction of every operation and
 *          precision, in one table each. FP64 has one lane and uses the scalar
 *          instruction; the other precisions use the packed one.
 * @pre The streams are set up (snrt_vec_stream_begin_2in()) and words > 0.
 */
static inline void snrt_vv_stream(snrt_vv_op_t op, precision_t precision,
                                  size_t words) {
    switch (op) {
        case SNRT_VV_ADD:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fadd.d", "vfadd.s",
                                 "vfadd.h", "vfadd.b");
            break;
    }
}

/**
 * @brief Apply an operation to one 64-bit word of each vector, without
 *        streaming.
 * @param in0 Pointer to the first input word (eight-byte aligned).
 * @param in1 Pointer to the second input word (eight-byte aligned).
 * @param out Pointer to the output word (eight-byte aligned, may equal in0 or
 *            in1).
 */
static inline void snrt_vv_word(snrt_vv_op_t op, precision_t precision,
                                const void* in0, const void* in1, void* out) {
    double x, y, o;
    switch (op) {
        case SNRT_VV_ADD:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fadd.d", "vfadd.s",
                                 "vfadd.h", "vfadd.b");
            break;
    }
}

/**
 * @brief Apply an operation to two vectors in L1, on the calling core only.
 *        The public snrt_vv_<op>_core_l1() call this.
 * @details Whole 64-bit words are processed by an SSR/FREP loop. Elements that
 *          do not fill a whole word are processed through zero-padded
 *          temporary words, so no memory outside the n elements is read or
 *          written.
 */
static inline bool snrt_vv_generic_core_l1(snrt_vv_op_t op, const void* a,
                                           const void* b, void* c, size_t n,
                                           precision_t precision) {
    if (!snrt_vv_supported(op, precision)) return false;

    const size_t lanes = snrt_vec_lanes(precision);
    const size_t words = n / lanes;
    const size_t tail = n % lanes;

    // Whole words, streamed.
    if (words > 0) {
#ifdef SNRT_SUPPORTS_FREP
        snrt_vec_stream_begin_2in(a, b, c, words);
        snrt_vv_stream(op, precision, words);
        snrt_vec_stream_end();
#else
        for (size_t i = 0; i < words; ++i)
            snrt_vv_word(op, precision,
                         (const uint8_t*)a + i * sizeof(uint64_t),
                         (const uint8_t*)b + i * sizeof(uint64_t),
                         (uint8_t*)c + i * sizeof(uint64_t));
#endif
    }

    // Remaining elements, through zero-padded words.
    if (tail > 0) {
        __attribute__((aligned(8))) uint8_t in0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        __attribute__((aligned(8))) uint8_t in1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        __attribute__((aligned(8))) uint8_t out[8];
        const size_t bytes = tail * (size_t)precision;
        const size_t offset = words * sizeof(uint64_t);
        memcpy(in0, (const uint8_t*)a + offset, bytes);
        memcpy(in1, (const uint8_t*)b + offset, bytes);
        snrt_vv_word(op, precision, in0, in1, out);
        memcpy((uint8_t*)c + offset, out, bytes);
    }

    snrt_fpu_fence();
    return true;
}

/**
 * @brief Apply an operation to two vectors in L1, across the compute cores of
 *        the cluster. The public snrt_vv_<op>_l1() call this.
 * @details The whole 64-bit words are divided among the compute cores in
 *          contiguous ranges (the first n_words % cores cores get one more);
 *          the last core also takes the elements that do not fill a whole
 *          word. Each core runs snrt_vv_generic_core_l1() on its range. Cores
 *          that are given no work return immediately.
 */
static inline bool snrt_vv_generic_l1(snrt_vv_op_t op, const void* a,
                                      const void* b, void* c, size_t n,
                                      precision_t precision) {
    if (!snrt_vv_supported(op, precision)) return false;
    if (!snrt_is_compute_core()) return true;

    const size_t lanes = snrt_vec_lanes(precision);
    const size_t cores = snrt_cluster_compute_core_num();
    const size_t core = snrt_cluster_core_idx();

    size_t first, count;
    snrt_vec_split(n / lanes, core, cores, &first, &count);

    size_t elems = count * lanes;
    if (core == cores - 1) elems += n % lanes;  // sub-word tail

    const size_t offset = first * sizeof(uint64_t);
    return snrt_vv_generic_core_l1(op, (const uint8_t*)a + offset,
                                   (const uint8_t*)b + offset,
                                   (uint8_t*)c + offset, elems, precision);
}

#undef SNRT_VV_STREAM_ASM
#undef SNRT_VV_WORD_ASM
#undef SNRT_VV_BY_PRECISION

//==============================================================================
// Public functions
//==============================================================================

//------------------------------------------------------------------------------
// Addition
//------------------------------------------------------------------------------

/**
 * @brief Add two vectors in L1 element by element, on the calling core only.
 * @details c[i] = a[i] + b[i] for i in [0, n). Whole 64-bit words are processed
 *          by an SSR/FREP loop (FP64 uses scalar fadd.d; FP32, FP16 and FP8 use
 *          the packed vfadd). Elements that do not fill a whole word are
 *          processed through zero-padded temporary words, so no memory outside
 *          the n elements is read or written.
 * @param a Pointer to the first element of the first input vector.
 * @param b Pointer to the first element of the second input vector.
 * @param c Pointer to the first element of the output vector.
 *          Pointer can be equal to either input pointer (c==a || c==b);
 *          otherwise the regions must not overlap.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre a, b and c point into L1 (TCDM), are eight-byte aligned and cover n
 *      elements. Addresses must be reachable by native CPU loads and stores.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note The FP pipeline is drained before returning, so this core's stores are
 *       complete. The caller synchronizes with other cores and DMA transfers.
 */
inline bool snrt_vv_add_core_l1(const void* a, const void* b, void* c,
                                size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_ADD, a, b, c, n, precision);
}

/**
 * @brief Add two vectors in L1 element by element, across the compute cores of
 *        the cluster.
 * @details Same operation as snrt_vv_add_core_l1(). The whole 64-bit words are
 *          divided among the compute cores in contiguous ranges (the first
 *          n_words % cores cores get one more); the last core also takes the
 *          elements that do not fill a whole word. Cores that are given no
 *          work return immediately. No core touches another core's elements.
 * @param a Pointer to the first element of the first input vector.
 * @param b Pointer to the first element of the second input vector.
 * @param c Pointer to the first element of the output vector. May equal a or
 *          b, as above.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre Every core of the cluster calls this with the same arguments. The DMA
 *      core returns immediately, as it does not run the packed-FP instructions.
 * @pre Same pointer and mode preconditions as snrt_vv_add_core_l1().
 * @note There is no barrier inside. Before reading the result from another core
 *       or by DMA, synchronize with snrt_cluster_hw_barrier().
 */
inline bool snrt_vv_add_l1(const void* a, const void* b, void* c, size_t n,
                           precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_ADD, a, b, c, n, precision);
}
