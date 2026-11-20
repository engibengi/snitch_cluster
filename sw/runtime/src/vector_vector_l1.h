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
 *          Operations available: add, sub, mul, div (not for FP8), min, max,
 *          and axpy (z = alpha * x + y), at the end of the file.
 */

#pragma once

#include <string.h>

#include "snitch_cluster_cfg.h"
#include "vector_common.h"
#include "vector_scalar_l1.h"  // snrt_vs_load, to load the scalar of axpy

//==============================================================================
// Implementation, shared by all the operations
//==============================================================================

typedef enum {
    SNRT_VV_ADD,
    SNRT_VV_SUB,
    SNRT_VV_MUL,
    SNRT_VV_DIV,
    SNRT_VV_MIN,
    SNRT_VV_MAX,
} snrt_vv_op_t;

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
 * @details Every operation exists for every precision, except division in FP8
 *          (the FP backend has no FP8 divider).
 */
SNRT_VEC_INLINE bool snrt_vv_supported(snrt_vv_op_t op, precision_t precision) {
    if (!snrt_vec_precision_valid(precision)) return false;
    if (op == SNRT_VV_DIV && precision == FP8) return false;
    return true;
}

/**
 * @brief Run the hardware loop of an operation over the streams.
 * @details This and snrt_vv_word() hold the instruction of every operation and
 *          precision, in one table each. FP64 has one lane and uses the scalar
 *          instruction; the other precisions use the packed one.
 * @pre The streams are set up (snrt_vec_stream_begin_2in()) and words > 0.
 */
SNRT_VEC_INLINE void snrt_vv_stream(snrt_vv_op_t op, precision_t precision,
                                    size_t words) {
    switch (op) {
        case SNRT_VV_ADD:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fadd.d", "vfadd.s",
                                 "vfadd.h", "vfadd.b");
            break;
        case SNRT_VV_SUB:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fsub.d", "vfsub.s",
                                 "vfsub.h", "vfsub.b");
            break;
        case SNRT_VV_MUL:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fmul.d", "vfmul.s",
                                 "vfmul.h", "vfmul.b");
            break;
        case SNRT_VV_DIV:
            // No FP8 here: vfdiv.b is not supported (see snrt_vv_supported).
            switch (precision) {
                case FP64:
                    SNRT_VV_STREAM_ASM("fdiv.d");
                    break;
                case FP32:
                    SNRT_VV_STREAM_ASM("vfdiv.s");
                    break;
                case FP16:
                    SNRT_VV_STREAM_ASM("vfdiv.h");
                    break;
                default:
                    break;
            }
            break;
        case SNRT_VV_MIN:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fmin.d", "vfmin.s",
                                 "vfmin.h", "vfmin.b");
            break;
        case SNRT_VV_MAX:
            SNRT_VV_BY_PRECISION(SNRT_VV_STREAM_ASM, "fmax.d", "vfmax.s",
                                 "vfmax.h", "vfmax.b");
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
SNRT_VEC_INLINE void snrt_vv_word(snrt_vv_op_t op, precision_t precision,
                                  const void* in0, const void* in1, void* out) {
    double x, y, o;
    switch (op) {
        case SNRT_VV_ADD:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fadd.d", "vfadd.s",
                                 "vfadd.h", "vfadd.b");
            break;
        case SNRT_VV_SUB:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fsub.d", "vfsub.s",
                                 "vfsub.h", "vfsub.b");
            break;
        case SNRT_VV_MUL:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fmul.d", "vfmul.s",
                                 "vfmul.h", "vfmul.b");
            break;
        case SNRT_VV_DIV:
            // No FP8 here: vfdiv.b is not supported (see snrt_vv_supported).
            switch (precision) {
                case FP64:
                    SNRT_VV_WORD_ASM("fdiv.d");
                    break;
                case FP32:
                    SNRT_VV_WORD_ASM("vfdiv.s");
                    break;
                case FP16:
                    SNRT_VV_WORD_ASM("vfdiv.h");
                    break;
                default:
                    break;
            }
            break;
        case SNRT_VV_MIN:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fmin.d", "vfmin.s",
                                 "vfmin.h", "vfmin.b");
            break;
        case SNRT_VV_MAX:
            SNRT_VV_BY_PRECISION(SNRT_VV_WORD_ASM, "fmax.d", "vfmax.s",
                                 "vfmax.h", "vfmax.b");
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
SNRT_VEC_INLINE bool snrt_vv_generic_core_l1(snrt_vv_op_t op, const void* a,
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
        uint64_t in0, in1, out;
        const size_t bytes = tail * (size_t)precision;
        const size_t offset = words * sizeof(uint64_t);
        snrt_vec_tail_load(&in0, (const uint8_t*)a + offset, bytes);
        snrt_vec_tail_load(&in1, (const uint8_t*)b + offset, bytes);
        snrt_vv_word(op, precision, &in0, &in1, &out);
        snrt_vec_tail_store((uint8_t*)c + offset, &out, bytes);
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
SNRT_VEC_INLINE bool snrt_vv_generic_l1(snrt_vv_op_t op, const void* a,
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
SNRT_VEC_INLINE bool snrt_vv_add_core_l1(const void* a, const void* b, void* c,
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
SNRT_VEC_INLINE bool snrt_vv_add_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_ADD, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// Subtraction
//------------------------------------------------------------------------------

/**
 * @brief Subtract two vectors in L1 element by element, on the calling core
 *        only.
 * @details c[i] = a[i] - b[i] for i in [0, n). FP64 uses fsub.d; the others
 *          use vfsub.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_sub_core_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_SUB, a, b, c, n, precision);
}

/**
 * @brief Subtract two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_sub_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_sub_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_SUB, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// Multiplication
//------------------------------------------------------------------------------

/**
 * @brief Multiply two vectors in L1 element by element, on the calling core
 *        only.
 * @details c[i] = a[i] * b[i] for i in [0, n). FP64 uses fmul.d; the others
 *          use vfmul.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_mul_core_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_MUL, a, b, c, n, precision);
}

/**
 * @brief Multiply two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_mul_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_mul_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_MUL, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// Division
//------------------------------------------------------------------------------

/**
 * @brief Divide two vectors in L1 element by element, on the calling core
 *        only.
 * @details c[i] = a[i] / b[i] for i in [0, n). The first vector is the
 *          dividend. FP64 uses fdiv.d; FP32 and FP16 use vfdiv.
 * @return True if the operation exists for the precision; false otherwise,
 *         without accessing the buffers. FP8 is not supported: the FP backend
 *         has no FP8 divider (vfdiv.b is not usable).
 * @note The last, partial word is padded with zeros, so the unused lanes
 *       compute 0 / 0. This sets the FP invalid flag, but the padding lanes
 *       are never stored.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_div_core_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_DIV, a, b, c, n, precision);
}

/**
 * @brief Divide two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_div_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @return As snrt_vv_div_core_l1(): false for FP8.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_div_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_DIV, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// Minimum
//------------------------------------------------------------------------------

/**
 * @brief Take the minimum of two vectors in L1 element by element, on the
 *        calling core only.
 * @details c[i] = min(a[i], b[i]) for i in [0, n). FP64 uses fmin.d; FP32,
 *          FP16 and FP8 use vfmin. The result is one of the two operands:
 *          nothing is rounded.
 * @note NaN and signed-zero handling follows the FP backend's min/max
 *       semantics. The FP rounding mode does not apply to this operation.
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_min_core_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_MIN, a, b, c, n, precision);
}

/**
 * @brief Take the minimum of two vectors in L1 element by element, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vv_min_core_l1(), with the work split as in
 *          snrt_vv_add_l1().
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_min_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_MIN, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// Maximum
//------------------------------------------------------------------------------

/**
 * @brief Take the maximum of two vectors in L1 element by element, on the
 *        calling core only.
 * @details c[i] = max(a[i], b[i]) for i in [0, n). FP64 uses fmax.d; FP32,
 *          FP16 and FP8 use vfmax. The result is one of the two operands:
 *          nothing is rounded.
 * @note NaN and signed-zero handling follows the FP backend's min/max
 *       semantics. The FP rounding mode does not apply to this operation.
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_max_core_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_generic_core_l1(SNRT_VV_MAX, a, b, c, n, precision);
}

/**
 * @brief Take the maximum of two vectors in L1 element by element, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vv_max_core_l1(), with the work split as in
 *          snrt_vv_add_l1().
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_max_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_generic_l1(SNRT_VV_MAX, a, b, c, n, precision);
}

//------------------------------------------------------------------------------
// axpy
//------------------------------------------------------------------------------

// axpy (z = alpha * x + y) is not one of the operations of the table above: it
// has a scalar operand, and in the packed precisions it takes two instructions
// per word (a multiply and an add), so its code is separate.
//
// Why two instructions: vfmac accumulates into its destination register, and
// an SSR register is either a read stream or a write stream, never both, so y
// cannot be streamed in and out of the same register through vfmac. The result
// goes to a third stream, which can point to y (in place) or to another
// vector. FP64 has a fused multiply-add with three independent operands, and
// needs only one instruction.
//
// The multiply and the add that uses its result depend on each other. Measured
// on this hardware (sw/tests/src/fpu_latency.c), the latency is 4 cycles, and
// four independent multiply-adds in flight reach one instruction per cycle.
// So the packed loop handles SNRT_AXPY_UNROLL words per iteration: that many
// multiplies into separate temporaries, then that many adds.
#define SNRT_AXPY_UNROLL 4

// FP64: ft2 = ft0 * scalar + ft1, one fused instruction per word. Expects
// `words` and `scalar` in the caller.
#define SNRT_AXPY_STREAM_FP64_ASM()                                       \
    asm volatile(                                                         \
        "frep.o %[n_frep], 1, 0, 0 \n"                                    \
        "fmadd.d ft2, ft0, %[s], ft1 \n"                                  \
        :                                                                 \
        : [ n_frep ] "r"(words - 1), [ s ] "f"(scalar)                    \
        : "ft0", "ft1", "ft2", "memory")

// Packed: SNRT_AXPY_UNROLL (4) words per iteration, groups iterations.
// ft2 = (ft0 * scalar) + ft1, with temporaries ft3 ... ft6. The reads of ft0
// and ft1 and the writes of ft2 are in stream order. Expects `groups` and
// `scalar` in the caller.
#define SNRT_AXPY_STREAM_GROUPS_ASM(mul, add)                             \
    asm volatile(                                                         \
        "frep.o %[n_frep], 8, 0, 0 \n"                                    \
        mul " ft3, ft0, %[s] \n"                                          \
        mul " ft4, ft0, %[s] \n"                                          \
        mul " ft5, ft0, %[s] \n"                                          \
        mul " ft6, ft0, %[s] \n"                                          \
        add " ft2, ft3, ft1 \n"                                           \
        add " ft2, ft4, ft1 \n"                                           \
        add " ft2, ft5, ft1 \n"                                           \
        add " ft2, ft6, ft1 \n"                                           \
        :                                                                 \
        : [ n_frep ] "r"(groups - 1), [ s ] "f"(scalar)                   \
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "memory")

// Packed: one word per iteration, `rem` iterations (the words that do not
// fill a group). Expects `rem` and `scalar` in the caller.
#define SNRT_AXPY_STREAM_REST_ASM(mul, add)                               \
    asm volatile(                                                         \
        "frep.o %[n_frep], 2, 0, 0 \n"                                    \
        mul " ft3, ft0, %[s] \n"                                          \
        add " ft2, ft3, ft1 \n"                                           \
        :                                                                 \
        : [ n_frep ] "r"(rem - 1), [ s ] "f"(scalar)                      \
        : "ft0", "ft1", "ft2", "ft3", "memory")

// One word, without streaming: out = scalar * in0 + in1. Expects `in0`, `in1`,
// `out`, `scalar`, and the temporaries `x`, `y`, `t` and `o` in the caller.
#define SNRT_AXPY_WORD_ASM(mul, add)                                      \
    asm volatile(                                                         \
        "fld %[x], 0(%[in0]) \n"                                          \
        "fld %[y], 0(%[in1]) \n"                                          \
        mul " %[t], %[x], %[s] \n"                                        \
        add " %[o], %[t], %[y] \n"                                        \
        "fsd %[o], 0(%[out]) \n"                                          \
        : [ x ] "=&f"(x), [ y ] "=&f"(y), [ t ] "=&f"(t), [ o ] "=&f"(o)  \
        : [ in0 ] "r"(in0), [ in1 ] "r"(in1), [ out ] "r"(out),           \
          [ s ] "f"(scalar)                                               \
        : "memory")

// The same, in FP64 (a fused multiply-add).
#define SNRT_AXPY_WORD_FP64_ASM()                                         \
    asm volatile(                                                         \
        "fld %[x], 0(%[in0]) \n"                                          \
        "fld %[y], 0(%[in1]) \n"                                          \
        "fmadd.d %[o], %[x], %[s], %[y] \n"                               \
        "fsd %[o], 0(%[out]) \n"                                          \
        : [ x ] "=&f"(x), [ y ] "=&f"(y), [ o ] "=&f"(o)                  \
        : [ in0 ] "r"(in0), [ in1 ] "r"(in1), [ out ] "r"(out),           \
          [ s ] "f"(scalar)                                               \
        : "memory")

// Applies the macro ASM to the (multiply, add) instructions of the packed
// precision in `precision` (in the caller).
#define SNRT_AXPY_BY_PACKED_PRECISION(ASM)                                \
    switch (precision) {                                                  \
        case FP32:                                                        \
            ASM("vfmul.r.s", "vfadd.s");                                  \
            break;                                                        \
        case FP16:                                                        \
            ASM("vfmul.r.h", "vfadd.h");                                  \
            break;                                                        \
        case FP8:                                                         \
            ASM("vfmul.r.b", "vfadd.b");                                  \
            break;                                                        \
        default:                                                          \
            break;                                                        \
    }

/**
 * @brief Run the hardware loops of axpy over the streams.
 * @pre The three streams are set up (snrt_vec_stream_begin_2in(x, y, z,
 *      words)) and words > 0.
 */
SNRT_VEC_INLINE void snrt_vv_axpy_stream(precision_t precision, size_t words,
                                         double scalar) {
    if (precision == FP64) {
        SNRT_AXPY_STREAM_FP64_ASM();
        return;
    }
    const size_t groups = words / SNRT_AXPY_UNROLL;
    const size_t rem = words % SNRT_AXPY_UNROLL;
    if (groups > 0) SNRT_AXPY_BY_PACKED_PRECISION(SNRT_AXPY_STREAM_GROUPS_ASM);
    if (rem > 0) SNRT_AXPY_BY_PACKED_PRECISION(SNRT_AXPY_STREAM_REST_ASM);
}

/**
 * @brief out = scalar * in0 + in1 for one 64-bit word, without streaming.
 * @param out May equal in1.
 */
SNRT_VEC_INLINE void snrt_vv_axpy_word(precision_t precision, const void* in0,
                                       const void* in1, void* out,
                                       double scalar) {
    double x, y, t, o;
    if (precision == FP64) {
        SNRT_AXPY_WORD_FP64_ASM();
        return;
    }
    SNRT_AXPY_BY_PACKED_PRECISION(SNRT_AXPY_WORD_ASM);
}

#undef SNRT_AXPY_STREAM_FP64_ASM
#undef SNRT_AXPY_STREAM_GROUPS_ASM
#undef SNRT_AXPY_STREAM_REST_ASM
#undef SNRT_AXPY_WORD_ASM
#undef SNRT_AXPY_WORD_FP64_ASM
#undef SNRT_AXPY_BY_PACKED_PRECISION

/**
 * @brief Compute z = alpha * x + y for vectors in L1, on the calling core
 *        only.
 * @details Whole 64-bit words are processed by an SSR/FREP loop: x and y are
 *          streamed in, and z is streamed out. FP64 uses one fused
 *          multiply-add per word (fmadd.d); the packed precisions use a
 *          multiply (vfmul.r) and an add (vfadd), four words per loop
 *          iteration to hide the FPU latency. Elements that do not fill a
 *          whole word are processed through zero-padded temporary words, so no
 *          memory outside the n elements is read or written.
 * @param alpha Pointer to one element (the scalar), of the given precision,
 *              naturally aligned.
 * @param x Pointer to the first element of the first input vector.
 * @param y Pointer to the first element of the second input vector (the one
 *          that is added).
 * @param z Pointer to the first element of the output vector. Pass y to
 *          update y in place (the usual BLAS axpy); z may also equal x.
 *          Otherwise the regions must not overlap.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre x, y and z point into L1 (TCDM), are eight-byte aligned and cover n
 *      elements. Addresses must be reachable by native CPU loads and stores.
 * @pre SSR is disabled, and the calling core supports the selected
 *      instruction.
 * @note The packed precisions round twice (the product, then the sum), so the
 *       result can differ in the last bit from a fused multiply-add. FP64 is
 *       fused (one rounding). The current FP rounding mode is used.
 * @note The FP pipeline is drained before returning, so this core's stores are
 *       complete. The caller synchronizes with other cores and DMA transfers.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_core_l1(const void* alpha, const void* x,
                                          const void* y, void* z, size_t n,
                                          precision_t precision) {
    if (!snrt_vec_precision_valid(precision)) return false;

    const size_t lanes = snrt_vec_lanes(precision);
    const size_t words = n / lanes;
    const size_t tail = n % lanes;

    const double scalar = snrt_vs_load(alpha, precision);

    // Whole words, streamed.
    if (words > 0) {
#ifdef SNRT_SUPPORTS_FREP
        snrt_vec_stream_begin_2in(x, y, z, words);
        snrt_vv_axpy_stream(precision, words, scalar);
        snrt_vec_stream_end();
#else
        for (size_t i = 0; i < words; ++i)
            snrt_vv_axpy_word(precision,
                              (const uint8_t*)x + i * sizeof(uint64_t),
                              (const uint8_t*)y + i * sizeof(uint64_t),
                              (uint8_t*)z + i * sizeof(uint64_t), scalar);
#endif
    }

    // Remaining elements, through zero-padded words.
    if (tail > 0) {
        uint64_t in0, in1, out;
        const size_t bytes = tail * (size_t)precision;
        const size_t offset = words * sizeof(uint64_t);
        snrt_vec_tail_load(&in0, (const uint8_t*)x + offset, bytes);
        snrt_vec_tail_load(&in1, (const uint8_t*)y + offset, bytes);
        snrt_vv_axpy_word(precision, &in0, &in1, &out, scalar);
        snrt_vec_tail_store((uint8_t*)z + offset, &out, bytes);
    }

    snrt_fpu_fence();
    return true;
}

/**
 * @brief Compute z = alpha * x + y for vectors in L1, across the compute cores
 *        of the cluster.
 * @details Same operation as snrt_vv_axpy_core_l1(). The whole 64-bit words are
 *          divided among the compute cores in contiguous ranges (the first
 *          n_words % cores cores get one more); the last core also takes the
 *          elements that do not fill a whole word. Cores that are given no
 *          work return immediately. No core touches another core's elements.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre Every core of the cluster calls this with the same arguments. The DMA
 *      core returns immediately, as it does not run the packed-FP
 *      instructions.
 * @pre Same pointer and mode preconditions as snrt_vv_axpy_core_l1().
 * @note There is no barrier inside. Before reading the result from another
 *       core or by DMA, synchronize with snrt_cluster_hw_barrier().
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_l1(const void* alpha, const void* x,
                                     const void* y, void* z, size_t n,
                                     precision_t precision) {
    if (!snrt_vec_precision_valid(precision)) return false;
    if (!snrt_is_compute_core()) return true;

    const size_t lanes = snrt_vec_lanes(precision);
    const size_t cores = snrt_cluster_compute_core_num();
    const size_t core = snrt_cluster_core_idx();

    size_t first, count;
    snrt_vec_split(n / lanes, core, cores, &first, &count);

    size_t elems = count * lanes;
    if (core == cores - 1) elems += n % lanes;  // sub-word tail

    const size_t offset = first * sizeof(uint64_t);
    return snrt_vv_axpy_core_l1(alpha, (const uint8_t*)x + offset,
                                (const uint8_t*)y + offset,
                                (uint8_t*)z + offset, elems, precision);
}
