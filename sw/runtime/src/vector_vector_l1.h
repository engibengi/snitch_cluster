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
 *          Each operation comes in six versions:
 *          - snrt_vv_<op>_core_l1: runs on the calling core only;
 *          - snrt_vv_<op>_l1: every compute core of the cluster calls it, and
 *            the work is split among them (in turns: core c takes words c,
 *            c + 8, ...);
 *          - snrt_vv_<op>_padded_core_l1 and snrt_vv_<op>_padded_l1: the same
 *            for vectors the caller has padded to whole words (the length is
 *            given in words, no tail code runs).
 *          - snrt_vv_<op>_auto_l1 and snrt_vv_<op>_padded_auto_l1: like the
 *            eight-core functions, but a short vector (fewer words than a
 *            threshold, see snrt_vec_min_words_multicore()) is done by core 0
 *            alone: they follow the faster of the one-core and eight-core
 *            versions.
 *
 *          Precision: give it as a compile-time constant (FP64, FP32, FP16
 *          or FP8 written at the call, or the parameter of an always-inlined
 *          caller that receives one). The functions are always inlined, so
 *          the code folds to the instructions of that precision. A precision
 *          known only at run time (read from memory) works, but then the
 *          code of every precision is inlined at the call: larger and slower
 *          (measured 20 to 100 cycles more per call). A caller with a
 *          run-time precision switches on it once, at the top of its kernel,
 *          and calls a version of the kernel with a constant precision.
 *
 *          The code has the same two parts as vector_scalar_l1.h: an
 *          implementation shared by all the operations, which differ only in
 *          the instruction they use, and the public functions, one pair per
 *          operation, each a one-line call into the shared code.
 *
 *          Operations available: add, sub, mul, div (not for FP8), min, max,
 *          and axpy (z = alpha * x + y), at the end of the file.
 *
 *          Memory layout (bank conflicts). The L1 memory has 32 banks of one
 *          64-bit word. The eight-core functions (_l1 and _padded_l1) run the
 *          streams of the cores in lockstep, and streams that hit the same
 *          bank in the same cycle make one core wait. The cost depends only on
 *          the distances between the buffers in words, modulo 32. Measured on
 *          the eight-core vector-vector add (FP64, 2048 and 1000 words) over
 *          all the 1024 combinations of b - a and c - a (a, b: the inputs,
 *          c: the output), and on FP64 axpy over 64 of them: with b - a = 16
 *          (mod 32) there is no extra cost for any c, including c == a and
 *          c == b (in
 *          place); 17 and 19 were also clean. Other values of b - a cost up to
 *          57 % (166 cycles at 2048 words, 79 at 1000) for some c: 22 % of the
 *          combinations cost more than 3 cycles. So: put b 16 words (mod 32)
 *          after a, and c anywhere. FP32, FP16 and FP8 axpy (two instructions
 *          per word, the FPU is the limit) and the vector-scalar functions
 *          (two streams) showed no dependence on the layout. The one-core
 *          functions were not measured. See paper_notes/BANK_CONFLICTS.md,
 *          section 8.
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
 * @brief Apply an operation to the words of a vector slice: whole 64-bit words,
 *        @p stride_bytes apart, streamed by an SSR/FREP loop.
 */
SNRT_VEC_INLINE void snrt_vv_words(snrt_vv_op_t op, precision_t precision,
                                   const void* a, const void* b, void* c,
                                   size_t words, size_t stride_bytes) {
#ifdef SNRT_SUPPORTS_FREP
    snrt_vec_stream_begin_2in_strided(a, b, c, words, stride_bytes);
    snrt_vv_stream(op, precision, words);
    snrt_vec_stream_end();
#else
    for (size_t i = 0; i < words; ++i)
        snrt_vv_word(op, precision, (const uint8_t*)a + i * stride_bytes,
                     (const uint8_t*)b + i * stride_bytes,
                     (uint8_t*)c + i * stride_bytes);
#endif
}

/**
 * @brief Implementation of every vector-vector function of this file.
 * @details The words of the vector are dealt out to 2^cores_log2 cores in
 *          turns: core `core` takes words core, core + cores, core + 2 cores,
 *          and so on (stride cores * 8 bytes). The cores then read and write
 *          different banks of the memory at the same time, and the split costs
 *          shifts and a mask, no division. The elements after the last whole
 *          word (the partial word), if any, go to the core whose turn it is,
 *          through zero-padded temporary words, so no memory outside the n
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
SNRT_VEC_INLINE bool snrt_vv_impl(snrt_vv_op_t op, const void* a, const void* b,
                                  void* c, size_t n, precision_t precision,
                                  size_t core, size_t cores_log2,
                                  bool padded) {
    if (!snrt_vv_supported(op, precision)) return false;
    if (core >= ((size_t)1 << cores_log2)) return true;

    const size_t lanes_log2 = snrt_vec_lanes_log2(precision);
    const size_t words = padded ? n : n >> lanes_log2;
    const size_t tail = padded ? 0 : n & (((size_t)1 << lanes_log2) - 1);
    const size_t mask = ((size_t)1 << cores_log2) - 1;

    // Words core, core + cores, ... below `words`.
    const size_t count = words > core ? ((words - core - 1) >> cores_log2) + 1
                                      : 0;
    if (count > 0) {
        const size_t offset = core * sizeof(uint64_t);
        snrt_vv_words(op, precision, (const uint8_t*)a + offset,
                      (const uint8_t*)b + offset, (uint8_t*)c + offset, count,
                      sizeof(uint64_t) << cores_log2);
    }

    // The partial word is the next one in the turns.
    if (tail > 0 && (words & mask) == core) {
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
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Add two vectors in L1 element by element, across the compute cores of
 *        the cluster.
 * @details Same operation as snrt_vv_add_core_l1(). The 64-bit words are dealt
 *          out to the eight compute cores in turns (core c takes words c,
 *          c + 8, c + 16, ...), which avoids most bank conflicts; the
 *          partial last word, if any, goes to the core whose turn it is.
 *          Cores that are given no work return immediately. No core touches
 *          another core's elements.
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
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_add_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_add_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_add_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_add_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_add_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_add_padded_core_l1().
 * @pre As snrt_vv_add_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_add_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_add_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_add_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_add_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_add_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, false),
                        false);
}

/**
 * @brief snrt_vv_add_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_add_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_add_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_ADD, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, true),
                        true);
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
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Subtract two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_sub_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_sub_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_sub_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_sub_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_sub_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_sub_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_sub_padded_core_l1().
 * @pre As snrt_vv_sub_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_sub_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_sub_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_sub_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_sub_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_sub_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, false),
                        false);
}

/**
 * @brief snrt_vv_sub_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_sub_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_sub_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_SUB, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, true),
                        true);
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
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Multiply two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_mul_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_mul_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_mul_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_mul_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_mul_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_mul_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_mul_padded_core_l1().
 * @pre As snrt_vv_mul_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_mul_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_mul_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_mul_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_mul_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_mul_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, false),
                        false);
}

/**
 * @brief snrt_vv_mul_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_mul_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_mul_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MUL, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, true),
                        true);
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
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Divide two vectors in L1 element by element, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vv_div_core_l1(), with the work split
 *          as in snrt_vv_add_l1().
 * @return As snrt_vv_div_core_l1(): false for FP8.
 * @note Parameters, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_div_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_div_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_div_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_div_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_div_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_div_padded_core_l1().
 * @pre As snrt_vv_div_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_div_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_div_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_div_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_div_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_div_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_DIV, n, precision,
                                                false),
                        false);
}

/**
 * @brief snrt_vv_div_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_div_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_div_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_DIV, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_DIV, n, precision,
                                                true),
                        true);
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
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Take the minimum of two vectors in L1 element by element, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vv_min_core_l1(), with the work split as in
 *          snrt_vv_add_l1().
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_min_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_min_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_min_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_min_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_min_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_min_padded_core_l1().
 * @pre As snrt_vv_min_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_min_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_min_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_min_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_min_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_min_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, false),
                        false);
}

/**
 * @brief snrt_vv_min_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_min_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_min_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MIN, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, true),
                        true);
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
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, 0, 0, false);
}

/**
 * @brief Take the maximum of two vectors in L1 element by element, across the
 *        compute cores of the cluster.
 * @details Same operation as snrt_vv_max_core_l1(), with the work split as in
 *          snrt_vv_add_l1().
 * @note Parameters, return value, preconditions and the other notes: as in
 *       snrt_vv_add_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_max_l1(const void* a, const void* b, void* c,
                                    size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_max_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to c, like the others (the
 *          result of them is meaningless). No tail code runs, which saves 16 to
 *          95 cycles per call when there is a partial word.
 * @pre As snrt_vv_max_core_l1(), with a, b and c covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_max_padded_core_l1(const void* a, const void* b,
                                                 void* c, size_t n,
                                                 precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_max_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant.
 * @details n is a number of WORDS; see snrt_vv_max_padded_core_l1().
 * @pre As snrt_vv_max_l1(), with a, b and c covering n words.
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_max_padded_l1(const void* a, const void* b,
                                            void* c, size_t n,
                                            precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_max_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments, like
 *          snrt_vv_max_l1(). A vector shorter than the threshold of
 *          snrt_vec_min_words_multicore() (about 13 to 16 words for this
 *          operation; the partial last word counts as a word) is done by core 0
 *          alone and the other cores return at once; a longer one is split
 *          among the eight cores in turns. The threshold is where eight cores
 *          start to be faster than one (measured).
 * @pre As snrt_vv_max_l1(). There is no barrier inside: before using
 *      the result on another core, synchronize.
 */
SNRT_VEC_INLINE bool snrt_vv_max_auto_l1(const void* a, const void* b, void* c,
                                         size_t n, precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, false),
                        false);
}

/**
 * @brief snrt_vv_max_auto_l1() on data the caller has padded to whole
 *        64-bit words: n is a number of WORDS. Caller-padded variant.
 * @pre As snrt_vv_max_padded_l1().
 */
SNRT_VEC_INLINE bool snrt_vv_max_padded_auto_l1(const void* a, const void* b,
                                                void* c, size_t n,
                                                precision_t precision) {
    return snrt_vv_impl(SNRT_VV_MAX, a, b, c, n, precision, snrt_vec_core_idx(),
                        SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_SIMPLE, n,
                                                precision, true),
                        true);
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
 * @brief Implementation of the axpy functions: the same split as
 *        snrt_vv_impl() (see there for core, cores_log2 and padded).
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_impl(const void* alpha, const void* x,
                                       const void* y, void* z, size_t n,
                                       precision_t precision, size_t core,
                                       size_t cores_log2, bool padded) {
    if (!snrt_vec_precision_valid(precision)) return false;
    if (core >= ((size_t)1 << cores_log2)) return true;

    const size_t lanes_log2 = snrt_vec_lanes_log2(precision);
    const size_t words = padded ? n : n >> lanes_log2;
    const size_t tail = padded ? 0 : n & (((size_t)1 << lanes_log2) - 1);
    const size_t mask = ((size_t)1 << cores_log2) - 1;
    const size_t count = words > core ? ((words - core - 1) >> cores_log2) + 1
                                      : 0;

    const bool tail_here = tail > 0 && (words & mask) == core;
    if (count == 0 && !tail_here) return true;

    const double scalar = snrt_vs_load(alpha, precision);

    if (count > 0) {
        const size_t offset = core * sizeof(uint64_t);
        const size_t stride = sizeof(uint64_t) << cores_log2;
#ifdef SNRT_SUPPORTS_FREP
        snrt_vec_stream_begin_2in_strided((const uint8_t*)x + offset,
                                          (const uint8_t*)y + offset,
                                          (uint8_t*)z + offset, count, stride);
        snrt_vv_axpy_stream(precision, count, scalar);
        snrt_vec_stream_end();
#else
        for (size_t i = 0; i < count; ++i)
            snrt_vv_axpy_word(precision,
                              (const uint8_t*)x + offset + i * stride,
                              (const uint8_t*)y + offset + i * stride,
                              (uint8_t*)z + offset + i * stride, scalar);
#endif
    }

    if (tail_here) {
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
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, 0, 0, false);
}

/**
 * @brief Compute z = alpha * x + y for vectors in L1, across the compute cores
 *        of the cluster.
 * @details Same operation as snrt_vv_axpy_core_l1(). The 64-bit words are dealt
 *          out to the eight compute cores in turns (core c takes words c,
 *          c + 8, c + 16, ...), which avoids most bank conflicts; the
 *          partial last word, if any, goes to the core whose turn it is.
 *          Cores that are given no work return immediately. No core touches
 *          another core's elements.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre Every core of the cluster calls this with the same arguments. The DMA
 *      core returns immediately, as it does not run the packed-FP
 *      instructions.
 * @pre Same pointer and mode preconditions as snrt_vv_axpy_core_l1().
 * @note There is no barrier inside. Before reading the result from another
 *       core or by DMA, synchronize with snrt_cluster_hw_barrier().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_l1(const void* alpha, const void* x,
                                     const void* y, void* z, size_t n,
                                     precision_t precision) {
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, snrt_vec_core_idx(),
                             SNRT_VEC_COMPUTE_CORES_LOG2, false);
}

/**
 * @brief snrt_vv_axpy_core_l1() on vectors that fill whole 64-bit words, on the
 *        calling core only. Caller-padded variant.
 * @details n is a number of WORDS. If the vectors have a partial last word,
 *          the caller has made room for all of it: the elements after the
 *          vector's last one are read, and written to z (meaningless values).
 *          No tail code runs.
 * @pre As snrt_vv_axpy_core_l1(), with x, y and z covering n words.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_padded_core_l1(const void* alpha,
                                                 const void* x, const void* y,
                                                 void* z, size_t n,
                                                 precision_t precision) {
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, 0, 0, true);
}

/**
 * @brief snrt_vv_axpy_l1() on vectors that fill whole 64-bit words, across the
 *        compute cores. Caller-padded variant: n is a number of WORDS, see
 *        snrt_vv_axpy_padded_core_l1().
 * @note Memory layout: see "Memory layout" in the description of this file;
 *       the distances between a, b and c modulo 32 words matter.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_padded_l1(const void* alpha, const void* x,
                                            const void* y, void* z, size_t n,
                                            precision_t precision) {
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, snrt_vec_core_idx(),
                             SNRT_VEC_COMPUTE_CORES_LOG2, true);
}

/**
 * @brief snrt_vv_axpy_l1() with the number of cores chosen by the length.
 * @details Every compute core calls it, with the same arguments. A vector
 *          shorter than the threshold of snrt_vec_min_words_multicore() (13 to 15
 *          words in FP64, 9 to 13 packed; the partial last word counts as a
 *          word) is done by core 0 alone and the other cores return at once;
 *          a longer one is split among the eight cores in turns.
 * @pre As snrt_vv_axpy_l1(). No barrier inside.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_auto_l1(const void* alpha, const void* x,
                                          const void* y, void* z, size_t n,
                                          precision_t precision) {
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, snrt_vec_core_idx(),
                             SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_AXPY, n,
                                                     precision, false),
                             false);
}

/**
 * @brief snrt_vv_axpy_auto_l1() on data the caller has padded to whole 64-bit
 *        words: n is a number of WORDS. Caller-padded variant.
 */
SNRT_VEC_INLINE bool snrt_vv_axpy_padded_auto_l1(const void* alpha,
                                                 const void* x, const void* y,
                                                 void* z, size_t n,
                                                 precision_t precision) {
    return snrt_vv_axpy_impl(alpha, x, y, z, n, precision, snrt_vec_core_idx(),
                             SNRT_VEC_MULTICORE_LOG2(SNRT_VEC_COST_AXPY, n,
                                                     precision, true),
                             true);
}

