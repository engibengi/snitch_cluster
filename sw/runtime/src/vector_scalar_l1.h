/**
 * @file
 * @brief Vector-scalar operations on data that is already in L1 (TCDM).
 * @details The functions here take pointers into the cluster's L1 memory and do
 *          no DMA transfer. They are the compute building blocks: the default
 *          vector-scalar functions (without the _l1 suffix, in vector_scalar.h)
 *          take pointers anywhere, and run these on tiles they bring into L1.
 */

#pragma once

#include <string.h>

#include "snitch_cluster_cfg.h"
#include "vector_common.h"

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

// One word, without streaming: out = instruction(in, scalar).
#define SNRT_VS_WORD_ASM(instruction)                      \
    asm volatile(                                          \
        "fld %[v], 0(%[in]) \n"                            \
        instruction " %[o], %[v], %[s] \n"                 \
        "fsd %[o], 0(%[out]) \n"                           \
        : [ v ] "=&f"(v), [ o ] "=&f"(o)                   \
        : [ in ] "r"(in), [ out ] "r"(out), [ s ] "f"(scalar) \
        : "memory")

/**
 * @brief Add a scalar to one 64-bit word, without streaming.
 * @param in Pointer to the input word (eight-byte aligned).
 * @param out Pointer to the output word (eight-byte aligned, may equal in).
 * @param scalar FP register holding the scalar in its low lane.
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 */
static inline void snrt_vs_add_word(const void* in, void* out, double scalar,
                                    precision_t precision) {
    double v, o;
    switch (precision) {
        case FP64:
            SNRT_VS_WORD_ASM("fadd.d");
            break;
        case FP32:
            SNRT_VS_WORD_ASM("vfadd.r.s");
            break;
        case FP16:
            SNRT_VS_WORD_ASM("vfadd.r.h");
            break;
        case FP8:
            SNRT_VS_WORD_ASM("vfadd.r.b");
            break;
        default:
            break;
    }
}

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
inline bool snrt_vec_add_scalar_core_l1(const void* a, const void* s, void* c,
                                        size_t n, precision_t precision) {
    if (!snrt_vec_precision_valid(precision)) return false;

    const size_t lanes = snrt_vec_lanes(precision);
    const size_t words = n / lanes;
    const size_t tail = n % lanes;

    // The scalar, in an FP register.
    double scalar;
    switch (precision) {
        case FP64:
            SNRT_VS_LOAD_ASM("fld");
            break;
        case FP32:
            SNRT_VS_LOAD_ASM("flw");
            break;
        case FP16:
            SNRT_VS_LOAD_ASM("flh");
            break;
        case FP8:
            SNRT_VS_LOAD_ASM("flb");
            break;
        default:
            return false;
    }

    // Whole words, streamed.
    if (words > 0) {
#ifdef SNRT_SUPPORTS_FREP
        snrt_vec_stream_begin(a, c, words);
        switch (precision) {
            case FP64:
                SNRT_VS_STREAM_ASM("fadd.d");
                break;
            case FP32:
                SNRT_VS_STREAM_ASM("vfadd.r.s");
                break;
            case FP16:
                SNRT_VS_STREAM_ASM("vfadd.r.h");
                break;
            case FP8:
                SNRT_VS_STREAM_ASM("vfadd.r.b");
                break;
            default:
                break;
        }
        snrt_vec_stream_end();
#else
        for (size_t i = 0; i < words; ++i)
            snrt_vs_add_word((const uint8_t*)a + i * sizeof(uint64_t),
                             (uint8_t*)c + i * sizeof(uint64_t), scalar,
                             precision);
#endif
    }

    // Remaining elements, through a zero-padded word.
    if (tail > 0) {
        __attribute__((aligned(8))) uint8_t in[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        __attribute__((aligned(8))) uint8_t out[8];
        const size_t bytes = tail * (size_t)precision;
        memcpy(in, (const uint8_t*)a + words * sizeof(uint64_t), bytes);
        snrt_vs_add_word(in, out, scalar, precision);
        memcpy((uint8_t*)c + words * sizeof(uint64_t), out, bytes);
    }

    snrt_fpu_fence();
    return true;
}

/**
 * @brief Add a scalar to each element of a vector in L1, across the compute
 *        cores of the cluster.
 * @details Same operation as snrt_vec_add_scalar_core_l1(). The whole 64-bit
 *          words are divided among the compute cores in contiguous ranges (the
 *          first n_words % cores cores get one more); the last core also takes
 *          the elements that do not fill a whole word. Cores that are given no
 *          work return immediately. No core touches another core's elements.
 * @param a Pointer to the first input element.
 * @param s Pointer to the scalar: one element of the selected precision.
 * @param c Pointer to the first output element. May equal a, as above.
 * @param n Number of elements (not bytes).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre Every core of the cluster calls this with the same arguments. The DMA
 *      core returns immediately, as it does not run the packed-FP instructions.
 * @pre Same pointer and mode preconditions as snrt_vec_add_scalar_core_l1().
 * @note There is no barrier inside. Before reading the result from another core
 *       or by DMA, synchronize with snrt_cluster_hw_barrier().
 */
inline bool snrt_vec_add_scalar_l1(const void* a, const void* s, void* c,
                                   size_t n, precision_t precision) {
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
    return snrt_vec_add_scalar_core_l1((const uint8_t*)a + offset, s,
                                       (uint8_t*)c + offset, elems, precision);
}

#undef SNRT_VS_LOAD_ASM
#undef SNRT_VS_STREAM_ASM
#undef SNRT_VS_WORD_ASM
