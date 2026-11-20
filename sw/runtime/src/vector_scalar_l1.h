/**
 * @file
 * @brief Vector-scalar operations on data that is already in L1 (TCDM).
 * @details Add, subtract, multiply, divide, or take the minimum or maximum of
 *          every element of a vector and one scalar: c[i] = a[i] <op> *s (for
 *          minimum and maximum, c[i] = min(a[i], *s) or max(a[i], *s)). The
 *          functions here take pointers into the cluster's L1 memory and do no
 *          DMA transfer. They are the compute building blocks: the default
 *          vector-scalar functions (without the _l1 suffix, in
 *          vector_scalar.h) take pointers anywhere, and run these on tiles
 *          they bring into L1.
 *
 *          Each operation comes in two versions:
 *          - snrt_vs_<op>_core_l1: runs on the calling core only;
 *          - snrt_vs_<op>_l1: every compute core of the cluster calls
 *            it, and the work is split among them.
 *
 *          The code is organised in two parts. The first part (implementation)
 *          is shared by all the operations, which differ only in the
 *          instruction they use. The second part is the public functions, one
 *          pair per operation, each a one-line call into the shared code.
 *
 *          Memory layout: the eight-core functions read one stream and write
 *          one, and showed no dependence on the distance between a and c: all
 *          32 values of c - a modulo 32 and c == a were measured (FP64 and
 *          FP8, 1000 and 2048 words), the spread is 2 cycles. See
 *          paper_notes/BANK_CONFLICTS.md, section 8.
 */

#pragma once

#include <string.h>

#include "snitch_cluster_cfg.h"
#include "vector_common.h"

//==============================================================================
// Implementation, shared by all the operations
//==============================================================================

typedef enum {
    SNRT_VS_ADD,
    SNRT_VS_SUB,
    SNRT_VS_MUL,
    SNRT_VS_DIV,
    SNRT_VS_MIN,
    SNRT_VS_MAX
} snrt_vs_op_t;

// Load one scalar element into an FP register, using a load of exactly the
// element size (the other bits of the register are not used by the .r forms).
#define SNRT_VS_LOAD_ASM(instruction)                      \
    asm volatile(instruction " %[r], 0(%[p])\n"            \
                 : [ r ] "=f"(scalar)                      \
                 : [ p ] "r"(s)                            \
                 : "memory")

// Hardware loop over a stream: ft1 = instruction(ft0, scalar). ft0 streams the
// source words and ft1 streams the destination words. The .r forms use the low
// lane of `scalar` for every lane. Expects `words` and `scalar` in the caller.
#define SNRT_VS_STREAM_ASM(instruction)                    \
    asm volatile(                                          \
        "frep.o %[n_frep], 1, 0, 0 \n"                     \
        instruction " ft1, ft0, %[s] \n"                   \
        :                                                  \
        : [ n_frep ] "r"(words - 1), [ s ] "f"(scalar)     \
        : "ft0", "ft1", "ft2", "memory")

// One word, without streaming: out = instruction(in, scalar). Expects `in`,
// `out`, `scalar`, and the temporaries `v` and `o` in the caller.
#define SNRT_VS_WORD_ASM(instruction)                      \
    asm volatile(                                          \
        "fld %[v], 0(%[in]) \n"                            \
        instruction " %[o], %[v], %[s] \n"                 \
        "fsd %[o], 0(%[out]) \n"                           \
        : [ v ] "=&f"(v), [ o ] "=&f"(o)                   \
        : [ in ] "r"(in), [ out ] "r"(out), [ s ] "f"(scalar) \
        : "memory")

// Expands to a switch on `precision` (in the caller) that applies the macro ASM
// to the instruction of that precision: instr_64, instr_32, instr_16, instr_8.
#define SNRT_VS_BY_PRECISION(ASM, instr_64, instr_32, instr_16, instr_8) \
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
 * @details FP8 division is not supported: the FP backend has no FP8 divider
 *          (vfdiv.b and vfdiv.r.b are not usable, and neither is the scalar
 *          fdiv.b).
 */
SNRT_VEC_INLINE bool snrt_vs_supported(snrt_vs_op_t op, precision_t precision) {
    if (!snrt_vec_precision_valid(precision)) return false;
    if (op == SNRT_VS_DIV && precision == FP8) return false;
    return true;
}

/**
 * @brief Load the scalar into an FP register.
 * @param s Pointer to one element of the given precision, naturally aligned.
 * @return The register, with the element in its low bits.
 */
SNRT_VEC_INLINE double snrt_vs_load(const void* s, precision_t precision) {
    double scalar = 0.0;
    SNRT_VS_BY_PRECISION(SNRT_VS_LOAD_ASM, "fld", "flw", "flh", "flb");
    return scalar;
}

/**
 * @brief Run the hardware loop of an operation over a stream of words.
 * @details This and snrt_vs_word() hold the instruction of every operation and
 *          precision, in one table each. The .r forms apply the scalar to
 *          every lane. FP64 has one lane and uses the scalar instruction.
 * @pre The streams are set up (snrt_vec_stream_begin()) and words > 0.
 */
SNRT_VEC_INLINE void snrt_vs_stream(snrt_vs_op_t op, precision_t precision,
                                    size_t words, double scalar) {
    switch (op) {
        case SNRT_VS_ADD:
            SNRT_VS_BY_PRECISION(SNRT_VS_STREAM_ASM, "fadd.d", "vfadd.r.s",
                                 "vfadd.r.h", "vfadd.r.b");
            break;
        case SNRT_VS_SUB:
            SNRT_VS_BY_PRECISION(SNRT_VS_STREAM_ASM, "fsub.d", "vfsub.r.s",
                                 "vfsub.r.h", "vfsub.r.b");
            break;
        case SNRT_VS_MUL:
            SNRT_VS_BY_PRECISION(SNRT_VS_STREAM_ASM, "fmul.d", "vfmul.r.s",
                                 "vfmul.r.h", "vfmul.r.b");
            break;
        case SNRT_VS_MIN:
            SNRT_VS_BY_PRECISION(SNRT_VS_STREAM_ASM, "fmin.d", "vfmin.r.s",
                                 "vfmin.r.h", "vfmin.r.b");
            break;
        case SNRT_VS_MAX:
            SNRT_VS_BY_PRECISION(SNRT_VS_STREAM_ASM, "fmax.d", "vfmax.r.s",
                                 "vfmax.r.h", "vfmax.r.b");
            break;
        case SNRT_VS_DIV:
            // No FP8 here: vfdiv.r.b is not supported (see snrt_vs_supported).
            switch (precision) {
                case FP64:
                    SNRT_VS_STREAM_ASM("fdiv.d");
                    break;
                case FP32:
                    SNRT_VS_STREAM_ASM("vfdiv.r.s");
                    break;
                case FP16:
                    SNRT_VS_STREAM_ASM("vfdiv.r.h");
                    break;
                default:
                    break;
            }
            break;
    }
}

/**
 * @brief Apply an operation to one 64-bit word, without streaming.
 * @param in Pointer to the input word (eight-byte aligned).
 * @param out Pointer to the output word (eight-byte aligned, may equal in).
 */
SNRT_VEC_INLINE void snrt_vs_word(snrt_vs_op_t op, precision_t precision,
                                  const void* in, void* out, double scalar) {
    double v, o;
    switch (op) {
        case SNRT_VS_ADD:
            SNRT_VS_BY_PRECISION(SNRT_VS_WORD_ASM, "fadd.d", "vfadd.r.s",
                                 "vfadd.r.h", "vfadd.r.b");
            break;
        case SNRT_VS_SUB:
            SNRT_VS_BY_PRECISION(SNRT_VS_WORD_ASM, "fsub.d", "vfsub.r.s",
                                 "vfsub.r.h", "vfsub.r.b");
            break;
        case SNRT_VS_MUL:
            SNRT_VS_BY_PRECISION(SNRT_VS_WORD_ASM, "fmul.d", "vfmul.r.s",
                                 "vfmul.r.h", "vfmul.r.b");
            break;
        case SNRT_VS_MIN:
            SNRT_VS_BY_PRECISION(SNRT_VS_WORD_ASM, "fmin.d", "vfmin.r.s",
                                 "vfmin.r.h", "vfmin.r.b");
            break;
        case SNRT_VS_MAX:
            SNRT_VS_BY_PRECISION(SNRT_VS_WORD_ASM, "fmax.d", "vfmax.r.s",
                                 "vfmax.r.h", "vfmax.r.b");
            break;
        case SNRT_VS_DIV:
            // No FP8 here: vfdiv.r.b is not supported (see snrt_vs_supported).
            switch (precision) {
                case FP64:
                    SNRT_VS_WORD_ASM("fdiv.d");
                    break;
                case FP32:
                    SNRT_VS_WORD_ASM("vfdiv.r.s");
                    break;
                case FP16:
                    SNRT_VS_WORD_ASM("vfdiv.r.h");
                    break;
                default:
                    break;
            }
            break;
    }
}

/**
 * @brief Implementation of every vector-scalar function of this file.
 * @details The words of the vector are dealt out to 2^cores_log2 cores in
 *          turns: core `core` takes words core, core + cores, core + 2 cores,
 *          and so on (stride cores * 8 bytes). The cores then read and write
 *          different banks of the memory at the same time, and the split costs
 *          shifts and a mask, no division. The elements after the last whole
 *          word (the partial word), if any, go to the core whose turn it is,
 *          through a zero-padded temporary word, so no memory outside the n
 *          elements is read or written.
 *
 *          With padded set, n is a number of WORDS, the last of which may be
 *          partial: the caller has made room for the whole word, so there is
 *          no tail code and the partial word is processed like the others (the
 *          elements after the n-th element of the vector are read and written).
 *
 *          One core on its own: core = 0, cores_log2 = 0. A core with an index
 *          of 2^cores_log2 or more (the DMA core) does nothing.
 * @pre For constants in core, cores_log2 and padded, the code folds to the
 *      specialised one. The calling core supports the instruction.
 */
SNRT_VEC_INLINE bool snrt_vs_impl(snrt_vs_op_t op, const void* a, const void* s,
                                  void* c, size_t n, precision_t precision,
                                  size_t core, size_t cores_log2,
                                  bool padded) {
    if (!snrt_vs_supported(op, precision)) return false;
    if (core >= ((size_t)1 << cores_log2)) return true;

    const size_t lanes_log2 = snrt_vec_lanes_log2(precision);
    const size_t words = padded ? n : n >> lanes_log2;
    const size_t tail = padded ? 0 : n & (((size_t)1 << lanes_log2) - 1);
    const size_t mask = ((size_t)1 << cores_log2) - 1;
    const size_t count = words > core ? ((words - core - 1) >> cores_log2) + 1
                                      : 0;
    const bool tail_here = tail > 0 && (words & mask) == core;
    if (count == 0 && !tail_here) return true;

    const double scalar = snrt_vs_load(s, precision);

    if (count > 0) {
        const size_t offset = core * sizeof(uint64_t);
        const size_t stride = sizeof(uint64_t) << cores_log2;
#ifdef SNRT_SUPPORTS_FREP
        snrt_vec_stream_begin_strided((const uint8_t*)a + offset,
                                      (uint8_t*)c + offset, count, stride);
        snrt_vs_stream(op, precision, count, scalar);
        snrt_vec_stream_end();
#else
        for (size_t i = 0; i < count; ++i)
            snrt_vs_word(op, precision, (const uint8_t*)a + offset + i * stride,
                         (uint8_t*)c + offset + i * stride, scalar);
#endif
    }

    // The partial word is the next one in the turns.
    if (tail_here) {
        uint64_t in, out;
        const size_t bytes = tail * (size_t)precision;
        const size_t offset = words * sizeof(uint64_t);
        snrt_vec_tail_load(&in, (const uint8_t*)a + offset, bytes);
        snrt_vs_word(op, precision, &in, &out, scalar);
        snrt_vec_tail_store((uint8_t*)c + offset, &out, bytes);
    }

    snrt_fpu_fence();
    return true;
}

// For a precision known only at run time: one non-inlined copy per operation,
// holding four copies of snrt_vs_impl() specialised for the precisions, chosen
// by one comparison chain (a switch would load its table from L3).
#define SNRT_VS_DEFINE_DISPATCH(NAME, OP)                                    \
    __attribute__((noinline, unused)) static bool snrt_vs_##NAME##_dispatch( \
        const void* a, const void* s, void* c, size_t n,                     \
        precision_t precision, size_t core, size_t cores_log2,               \
        bool padded) {                                                       \
        SNRT_VEC_BY_PRECISION(                                               \
            P, snrt_vs_impl(OP, a, s, c, n, P, core, cores_log2, padded));   \
    }
SNRT_VS_DEFINE_DISPATCH(add, SNRT_VS_ADD)
SNRT_VS_DEFINE_DISPATCH(sub, SNRT_VS_SUB)
SNRT_VS_DEFINE_DISPATCH(mul, SNRT_VS_MUL)
SNRT_VS_DEFINE_DISPATCH(div, SNRT_VS_DIV)
SNRT_VS_DEFINE_DISPATCH(min, SNRT_VS_MIN)
SNRT_VS_DEFINE_DISPATCH(max, SNRT_VS_MAX)
#undef SNRT_VS_DEFINE_DISPATCH

// Body of a public function: inline for a constant precision, otherwise the
// call of the dispatcher of the operation. Expects a, s, c, n, precision.
#define SNRT_VS_ROUTE(NAME, OP, CORE, LOG2, PADDED)                          \
    do {                                                                     \
        if (__builtin_constant_p(precision))                                 \
            return snrt_vs_impl(OP, a, s, c, n, precision, CORE, LOG2,       \
                                PADDED);                                     \
        return snrt_vs_##NAME##_dispatch(a, s, c, n, precision, CORE, LOG2,  \
                                         PADDED);                            \
    } while (0)

#undef SNRT_VS_LOAD_ASM
#undef SNRT_VS_STREAM_ASM
#undef SNRT_VS_WORD_ASM
#undef SNRT_VS_BY_PRECISION

//==============================================================================
// Public functions
//==============================================================================

//------------------------------------------------------------------------------
// Addition
//------------------------------------------------------------------------------

/**
 * @brief Add a scalar to each element of a vector in L1, on the calling core
 *        only.
 * @details c[i] = a[i] + *s for i in [0, n). Whole 64-bit words are processed
 *          by an SSR/FREP loop (FP64 uses scalar fadd.d; FP32, FP16 and FP8 use
 *          the packed vfadd.r, which uses the low lane of the scalar register
 *          for every lane). Elements that do not fill a whole word are
 *          processed through a zero-padded temporary word, so no memory outside
 *          the n elements is read or written.
 * @param a Pointer to the first input element.
 * @param s Pointer to the scalar: one element of the selected precision.
 * @param c Pointer to the first output element.
 *          Pointer can be equal to the input pointer (c==a); otherwise the
 *          regions must not overlap.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre a and c point into L1 (TCDM), are eight-byte aligned and cover n
 *      elements. s is naturally aligned for its precision and points to one
 *      element. Addresses must be reachable by native CPU loads and stores.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note The FP pipeline is drained before returning, so this core's stores are
 *       complete. The caller synchronizes with other cores and DMA transfers.
 */
SNRT_VEC_INLINE bool snrt_vs_add_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(add, SNRT_VS_ADD, 0, 0, false);
}

/**
 * @brief Add a scalar to each element of a vector in L1, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vs_add_core_l1(). The 64-bit words are dealt
 *          out to the eight compute cores in turns (core c takes words c,
 *          c + 8, c + 16, ...), which avoids most bank conflicts; the
 *          partial last word, if any, goes to the core whose turn it is.
 *          Cores that are given no work return immediately. No core touches
 *          another core's elements.
 * @param a Pointer to the first input element.
 * @param s Pointer to the scalar: one element of the selected precision.
 * @param c Pointer to the first output element. May equal a, as above.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre Every core of the cluster calls this with the same arguments. The DMA
 *      core returns immediately, as it does not run the packed-FP instructions.
 * @pre Same pointer and mode preconditions as snrt_vs_add_core_l1().
 * @note There is no barrier inside. Before reading the result from another core
 *       or by DMA, synchronize with snrt_cluster_hw_barrier().
 */
SNRT_VEC_INLINE bool snrt_vs_add_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(add, SNRT_VS_ADD, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_add_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_add_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_add_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(add, SNRT_VS_ADD, 0, 0, true);
}

/**
 * @brief snrt_vs_add_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_add_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_add_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(add, SNRT_VS_ADD, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

//------------------------------------------------------------------------------
// Subtraction
//------------------------------------------------------------------------------

/**
 * @brief Subtract a scalar from each element of a vector in L1, on the calling
 *        core only.
 * @details c[i] = a[i] - *s for i in [0, n). The vector is the minuend: the
 *          scalar is subtracted from it, not the other way around. FP64 uses
 *          fsub.d; FP32, FP16 and FP8 use vfsub.r.
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_sub_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(sub, SNRT_VS_SUB, 0, 0, false);
}

/**
 * @brief Subtract a scalar from each element of a vector in L1, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vs_sub_core_l1(), with the work split
 *          as in snrt_vs_add_l1().
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_sub_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(sub, SNRT_VS_SUB, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_sub_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_sub_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_sub_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(sub, SNRT_VS_SUB, 0, 0, true);
}

/**
 * @brief snrt_vs_sub_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_sub_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_sub_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(sub, SNRT_VS_SUB, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

//------------------------------------------------------------------------------
// Multiplication
//------------------------------------------------------------------------------

/**
 * @brief Multiply each element of a vector in L1 by a scalar, on the calling
 *        core only.
 * @details c[i] = a[i] * *s for i in [0, n). FP64 uses fmul.d; FP32, FP16 and
 *          FP8 use vfmul.r.
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_mul_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(mul, SNRT_VS_MUL, 0, 0, false);
}

/**
 * @brief Multiply each element of a vector in L1 by a scalar, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vs_mul_core_l1(), with the work split
 *          as in snrt_vs_add_l1().
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_mul_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(mul, SNRT_VS_MUL, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_mul_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_mul_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_mul_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(mul, SNRT_VS_MUL, 0, 0, true);
}

/**
 * @brief snrt_vs_mul_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_mul_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_mul_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(mul, SNRT_VS_MUL, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

//------------------------------------------------------------------------------
// Division
//------------------------------------------------------------------------------

/**
 * @brief Divide each element of a vector in L1 by a scalar, on the calling core
 *        only.
 * @details c[i] = a[i] / *s for i in [0, n). The vector is the dividend. FP64
 *          uses fdiv.d; FP32 and FP16 use vfdiv.r.
 * @return True if the operation exists for the precision; false otherwise,
 *         without accessing the buffers. FP8 is not supported: the FP backend
 *         has no FP8 divider (vfdiv.b, vfdiv.r.b and the scalar fdiv.b are not
 *         usable).
 * @note The last, partial word is padded with zeros, so the unused lanes
 *       compute 0 / *s. This can set the FP exception flags (for example
 *       0 / 0 if *s is zero), but the padding lanes are never stored.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vs_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_div_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(div, SNRT_VS_DIV, 0, 0, false);
}

/**
 * @brief Divide each element of a vector in L1 by a scalar, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vs_div_core_l1(), with the work split
 *          as in snrt_vs_add_l1().
 * @return True if the operation exists for the precision (FP8 does not, see
 *         snrt_vs_div_core_l1()); false otherwise, without accessing
 *         the buffers.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vs_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_div_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(div, SNRT_VS_DIV, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_div_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_div_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_div_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(div, SNRT_VS_DIV, 0, 0, true);
}

/**
 * @brief snrt_vs_div_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_div_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_div_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(div, SNRT_VS_DIV, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

//------------------------------------------------------------------------------
// Minimum
//------------------------------------------------------------------------------

/**
 * @brief Take the minimum of each element of a vector in L1 and a scalar, on
 *        the calling core only.
 * @details c[i] = min(a[i], *s) for i in [0, n). FP64 uses fmin.d; FP32, FP16
 *          and FP8 use vfmin.r. The result is one of the two operands: nothing
 *          is rounded.
 * @note NaN and signed-zero handling follows the FP backend's min/max
 *       semantics. The FP rounding mode does not apply to this operation.
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vs_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_min_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(min, SNRT_VS_MIN, 0, 0, false);
}

/**
 * @brief Take the minimum of each element of a vector in L1 and a scalar,
 *        across the compute cores of the cluster.
 * @details Same operation as snrt_vs_min_core_l1(), with the work split as in
 *          snrt_vs_add_l1().
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_min_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(min, SNRT_VS_MIN, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_min_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_min_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_min_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(min, SNRT_VS_MIN, 0, 0, true);
}

/**
 * @brief snrt_vs_min_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_min_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_min_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(min, SNRT_VS_MIN, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

//------------------------------------------------------------------------------
// Maximum
//------------------------------------------------------------------------------

/**
 * @brief Take the maximum of each element of a vector in L1 and a scalar, on
 *        the calling core only.
 * @details c[i] = max(a[i], *s) for i in [0, n). FP64 uses fmax.d; FP32, FP16
 *          and FP8 use vfmax.r. The result is one of the two operands: nothing
 *          is rounded.
 * @note NaN and signed-zero handling follows the FP backend's min/max
 *       semantics. The FP rounding mode does not apply to this operation.
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vs_add_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_max_core_l1(const void* a, const void* s, void* c,
                                         size_t n, precision_t precision) {
    SNRT_VS_ROUTE(max, SNRT_VS_MAX, 0, 0, false);
}

/**
 * @brief Take the maximum of each element of a vector in L1 and a scalar,
 *        across the compute cores of the cluster.
 * @details Same operation as snrt_vs_max_core_l1(), with the work split as in
 *          snrt_vs_add_l1().
 * @note Parameters, return value, preconditions and notes: as in
 *       snrt_vs_add_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_max_l1(const void* a, const void* s, void* c,
                                    size_t n, precision_t precision) {
    SNRT_VS_ROUTE(max, SNRT_VS_MAX, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vs_max_core_l1() on a vector that fills whole 64-bit
 *        words, on the calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vector has a partial last word, the
 *          caller has made room for all of it: the elements after the vector's
 *          last one are read, and written to c (meaningless values). No tail
 *          code runs, which saves 16 to 95 cycles per call when there is a
 *          partial word.
 * @pre As snrt_vs_max_core_l1(), with a and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vs_max_padded_core_l1(const void* a, const void* s,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    SNRT_VS_ROUTE(max, SNRT_VS_MAX, 0, 0, true);
}

/**
 * @brief snrt_vs_max_l1() on a vector that fills whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vs_max_padded_core_l1().
 */
SNRT_VEC_INLINE bool snrt_vs_max_padded_l1(const void* a, const void* s,
                                            void* c, size_t n,
                                            precision_t precision) {
    SNRT_VS_ROUTE(max, SNRT_VS_MAX, snrt_vec_core_idx(),
                  SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

#undef SNRT_VS_ROUTE
