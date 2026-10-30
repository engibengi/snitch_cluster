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

