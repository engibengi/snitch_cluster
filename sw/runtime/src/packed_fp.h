/**
 * @file
 * @brief This file provides helpers for Snitch packed floating-point arithmetic.
 */

#pragma once

#include "types.h"

#define SNRT_VFOP_ASM(instruction)                         \
    asm volatile(                                          \
        "fld %[rs1], 0(%[a])\n"                            \
        "fld %[rs2], 0(%[b])\n"                            \
        instruction " %[rd], %[rs1], %[rs2]\n"             \
        "fsd %[rd], 0(%[c])\n"                             \
        : [rs1] "=f"(lhs), [rs2] "=f"(rhs), [rd] "=f"(res) \
        : [a] "r"(a), [b] "r"(b), [c] "r"(c)               \
        : "memory"                                         \
    )

/**
 * @brief Multiply corresponding floating-point elements in two 64-bit words.
 * @details Loads eight bytes from each input and stores eight result bytes.
 *          FP64 uses scalar fmul.d; FP32, FP16 and FP8 use packed vfmul.
 * @param a Pointer to the first input word.
 * @param b Pointer to the second input word.
 * @param c Pointer to the destination word.
 *          Pointer can be equal to either input pointer (c==a || c==b).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note Inputs are loaded before the output is stored. The caller is responsible
 *       for synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfmul(const void* a, const void* b, void* c, precision_t precision) {
    
    double lhs, rhs, res;

    switch(precision){
        case FP64:
            SNRT_VFOP_ASM("fmul.d");
            break;
        case FP32:
            SNRT_VFOP_ASM("vfmul.s");
            break;
        case FP16:
            SNRT_VFOP_ASM("vfmul.h");
            break;
        case FP8:
            SNRT_VFOP_ASM("vfmul.b");
            break;
        default:
            return false;
    }
    return true;
}


/**
 * @brief Add corresponding floating-point elements in two 64-bit words.
 * @details Loads eight bytes from each input and stores eight result bytes.
 *          FP64 uses scalar fadd.d; FP32, FP16 and FP8 use packed vfadd.
 * @param a Pointer to the first input word.
 * @param b Pointer to the second input word.
 * @param c Pointer to the destination word.
 *          Pointer can be equal to either input pointer (c==a || c==b).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note Inputs are loaded before the output is stored. The caller is responsible
 *       for synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfadd(const void* a, const void* b, void* c, precision_t precision) {
    
    double lhs, rhs, res;

    switch(precision){
        case FP64:
            SNRT_VFOP_ASM("fadd.d");
            break;
        case FP32:
            SNRT_VFOP_ASM("vfadd.s");
            break;
        case FP16:
            SNRT_VFOP_ASM("vfadd.h");
            break;
        case FP8:
            SNRT_VFOP_ASM("vfadd.b");
            break;
        default:
            return false;
    }
    return true;
}


/**
 * @brief Divide corresponding floating-point elements in two 64-bit words.
 * @details Loads eight bytes from each input and stores eight result bytes.
 *          FP64 uses scalar fdiv.d; FP32, FP16 and FP8 use packed vfdiv.
 * @param a Pointer to the first input word.
 * @param b Pointer to the second input word.
 * @param c Pointer to the destination word.
 *          Pointer can be equal to either input pointer (c==a || c==b).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note Inputs are loaded before the output is stored. The caller is responsible
 *       for synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfdiv(const void* a, const void* b, void* c, precision_t precision) {
    
    double lhs, rhs, res;

    switch(precision){
        case FP64:
            SNRT_VFOP_ASM("fdiv.d");
            break;
        case FP32:
            SNRT_VFOP_ASM("vfdiv.s");
            break;
        case FP16:
            SNRT_VFOP_ASM("vfdiv.h");
            break;
        case FP8:
            // TODO: Currently vfdiv.b is unsupported
            // Enable it (uncomment the two following lines) when support comes
            // SNRT_VFOP_ASM("vfdiv.b");
            // break;
        default:
            return false;
    }
    return true;
}


/**
 * @brief Subtract corresponding floating-point elements in two 64-bit words.
 * @details Loads eight bytes from each input and stores eight result bytes.
 *          FP64 uses scalar fsub.d; FP32, FP16 and FP8 use packed vfsub.
 * @param a Pointer to the first input word.
 * @param b Pointer to the second input word.
 * @param c Pointer to the destination word.
 *          Pointer can be equal to either input pointer (c==a || c==b).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note Inputs are loaded before the output is stored. The caller is responsible
 *       for synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfsub(const void* a, const void* b, void* c, precision_t precision) {
    
    double lhs, rhs, res;

    switch(precision){
        case FP64:
            SNRT_VFOP_ASM("fsub.d");
            break;
        case FP32:
            SNRT_VFOP_ASM("vfsub.s");
            break;
        case FP16:
            SNRT_VFOP_ASM("vfsub.h");
            break;
        case FP8:
            SNRT_VFOP_ASM("vfsub.b");
            break;
        default:
            return false;
    }
    return true;
}

#undef SNRT_VFOP_ASM

#define SNRT_VFSQRT_ASM(instruction)                       \
    asm volatile(                                          \
        "fld %[rs1], 0(%[a])\n"                            \
        instruction " %[rd], %[rs1]\n"                     \
        "fsd %[rd], 0(%[b])\n"                             \
        : [rs1] "=f"(val), [rd] "=f"(res)                  \
        : [a] "r"(a), [b] "r"(b)                           \
        : "memory"                                         \
    )



/**
 * @brief Compute the square root of each floating-point element in a 64-bit word.
 * @details Loads eight bytes from the input and stores eight result bytes.
 *          FP64 uses scalar fsqrt.d; FP32, FP16 and FP8 select packed vfsqrt
 *          for two, four and eight lanes respectively. One word is processed.
 * @param a Pointer to the input word.
 * @param b Pointer to the destination word.
 *          Pointer can be equal to the input pointer (b==a).
 * @param precision Element precision: FP64, FP32, FP16 or FP8.
 * @return True if the precision is recognized; false otherwise, without
 *         accessing the buffers. True does not certify hardware support or
 *         report floating-point exceptions.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @warning The default THMULTI div/sqrt backend does not support FP8/FP8ALT.
 *          This wrapper currently still issues vfsqrt.b for FP8; its result
 *          must not be treated as a valid FP8 square root on that backend.
 * @note Negative finite inputs produce NaN and set the invalid-operation flag
 *       on a supported path. The wrapper does not inspect exception flags.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note Inputs are loaded before the output is stored. The caller is responsible
 *       for synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfsqrt(const void* a, void* b, precision_t precision) {

    double val, res;

    switch(precision){
        case FP64:
            SNRT_VFSQRT_ASM("fsqrt.d");
            break;
        case FP32:
            SNRT_VFSQRT_ASM("vfsqrt.s");
            break;
        case FP16:
            SNRT_VFSQRT_ASM("vfsqrt.h");
            break;
        case FP8:
            // TODO: Currently vfsqrt.b is unsupported
            // Enable it (uncomment the two following lines) when support comes
            // Along with the relative tests
            // SNRT_VFSQRT_ASM("vfsqrt.b");
            // break;
        default:
            return false;
    }
    return true;
}

#undef SNRT_VFSQRT_ASM


#define SNRT_VFSUM_ASM(instruction)                        \
    asm volatile(                                          \
        "fld %[rs1], 0(%[a])\n"                            \
        "fld %[rd], 0(%[b])\n"                             \
        instruction " %[rd], %[rs1]\n"                     \
        "fsd %[rd], 0(%[c])\n"                             \
        : [rs1] "=f"(lhs), [rd] "=f"(res) \
        : [a] "r"(a), [b] "r"(b), [c] "r"(c)               \
        : "memory"                                         \
    )

/**
 * @brief Compute fused three-term pairwise sums with an accumulator word.
 * @details For output lane j, computes c[j] = round(b[j] + a[2*j] + a[2*j+1]).
 *          Each result has one final rounding; this is not a serial pair of
 *          rounded additions or a reduction of the entire input to one value.
 *          FP32 produces one result, FP16 two and FP8 four, in the low 32 bits.
 *          The upper 32 bits are all ones: interpreted as FP32/FP16/FP8 lanes,
 *          these are NaNs. FP64 is not supported; there is no fsum.d branch.
 *          Loads eight bytes from each input and stores eight result bytes.
 * @param a Pointer to the packed source word whose adjacent lanes are summed.
 * @param b Pointer to the accumulator word. Only its low 32 bits contribute.
 * @param c Pointer to the destination word. May equal either input pointer.
 * @param precision Element precision: FP32, FP16 or FP8. FP64 is rejected.
 * @return True if the precision is accepted; false otherwise, without accessing
 *         the buffers. True does not certify hardware support or report FP flags.
 * @pre All pointers are eight-byte aligned and refer to accessible regions of
 *      at least eight bytes. Addresses must be reachable by native CPU loads
 *      and stores; wider DMA addresses are not supported by this interface.
 * @pre SSR is disabled, and the calling core supports the selected instruction.
 * @note The current FP rounding mode is used. For FP16 and FP8, the source and
 *       destination format-mode settings must match the intended data formats.
 * @note NaNs, infinities and invalid combinations follow the FP backend's fused
 *       sum semantics. Upper-word NaNs are boxing bits, not computed sum results.
 * @note Both inputs are loaded before the output is stored. The caller handles
 *       synchronization with other cores and outstanding DMA/SSR accesses.
 */
inline bool snrt_vfsum(const void* a, const void* b, void* c, precision_t precision) {

    double lhs, res;

    switch(precision){
        case FP64:
            // There is no implementation of three members reductive sum in FP64
            // For obvious reasons!
            return false;
        case FP32:
            SNRT_VFSUM_ASM("vfsum.s");
            break;
        case FP16:
            SNRT_VFSUM_ASM("vfsum.h");
            break;
        case FP8:
            SNRT_VFSUM_ASM("vfsum.b");
            break;
        default:
            return false;
    }
    return true;
}

#undef SNRT_VFSUM_ASM

//TODO: add vfmac

//TODO: add vfmre
