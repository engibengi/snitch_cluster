#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

int main() {
    // Startup defaults: ordinary formats, RNE, SSR/scalar chaining disabled.
    if (snrt_cluster_core_idx() != 0) return 0;

    __attribute__((aligned(8))) double   a_f64[] = {2.0};
    __attribute__((aligned(8))) float    a_f32[] = {2.0f, 4.0f};
    __attribute__((aligned(8))) _Float16 a_f16[] = {2.0, 4.0, 6.0, 8.0};
    __attribute__((aligned(8))) uint8_t   a_f8[] =
        {0x40, 0x44, 0x46, 0x48, 0x49, 0x4a, 0x4b, 0x4c};
    // FP8: 2,    4,    6,    8,   10,   12,   14,   16.

    __attribute__((aligned(8))) double   b_f64[] = {2.0};
    __attribute__((aligned(8))) float    b_f32[] = {2.0f, 3.0f};         // Only first contributes
    __attribute__((aligned(8))) _Float16 b_f16[] = {2.0, 3.0, 4.0, 5.0}; // Only first two contribute
    __attribute__((aligned(8))) uint8_t   b_f8[] =
        {0x40, 0x42, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49};
    // FP8: 2,    3,    4,    5,    6,    7,    8,   10.
    // Only the first four contribute!

    __attribute__((aligned(8))) double   c_f64[] = {0.0};
    __attribute__((aligned(8))) float    c_f32[] = {0.0f, 0.0f};
    __attribute__((aligned(8))) _Float16 c_f16[] = {0.0, 0.0, 0.0, 0.0};
    __attribute__((aligned(8))) uint8_t   c_f8[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // FP32 copies of the inputs, used as in-place (aliased) destinations.
    __attribute__((aligned(8))) float     a_alias_f32[] = {2.0f, 4.0f};
    __attribute__((aligned(8))) float     b_alias_f32[] = {2.0f, 3.0f};

    // Raw encodings let us check every bit, including the exact all-ones
    // NaN boxing. NaN != NaN, so ordinary floating-point equality cannot
    // validate those lanes; no FP conversion of their payload is needed.
    const uint32_t expected_f32[] = {0xc0800000, 0xffffffff};
    // Values: -4, NaN. -4 = accumulator 2 - 2 - 4.
    const uint16_t expected_f16[] = {0xc400, 0xc980, 0xffff, 0xffff};
    // Values: -4, -11, NaN, NaN. -11 = accumulator 3 - 6 - 8.
    const uint8_t expected_f8[] = {
        0xc4, 0xca, 0xcc, 0xce, 0xff, 0xff, 0xff, 0xff};
    // Exact accumulator - pair sums: -4, -11, -18, -25.
    // E5M2/RNE values: -4, -12, -16, -24.
    // -11 and -18 are midpoints: ties to even selects -12 and -16.

    unsigned errors = 0;

    // FP64 is rejected and must leave the destination unchanged.
    if (snrt_vfnsum(a_f64, b_f64, c_f64, FP64)) {
        ++errors;
        printf("Unexpected FP64 vfnsum acceptance!\n");
    }
    uint64_t actual_f64 = __builtin_bit_cast(uint64_t, c_f64[0]);
    if (actual_f64 != 0) {
        ++errors;
        printf("Rejected FP64 vfnsum changed the destination!\n");
    }

    if (!snrt_vfnsum(a_f32, b_f32, c_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        uint32_t actual = __builtin_bit_cast(uint32_t, c_f32[i]);
        if (actual != expected_f32[i]) {
            ++errors;
            printf("FP32 vfnsum lane %u: expected 0x%08x, got 0x%08x\n",
                   i, (unsigned)expected_f32[i], (unsigned)actual);
        }
    }

    if (!snrt_vfnsum(a_f16, b_f16, c_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        uint16_t actual = __builtin_bit_cast(uint16_t, c_f16[i]);
        if (actual != expected_f16[i]) {
            ++errors;
            printf("FP16 vfnsum lane %u: expected 0x%04x, got 0x%04x\n",
                   i, (unsigned)expected_f16[i], (unsigned)actual);
        }
    }

    if (!snrt_vfnsum(a_f8, b_f8, c_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (c_f8[i] != expected_f8[i]) {
            ++errors;
            printf("FP8 vfnsum lane %u: expected 0x%02x, got 0x%02x\n",
                   i, (unsigned)expected_f8[i], (unsigned)c_f8[i]);
        }
    }

    // Aliasing test: destination == first source
    if (!snrt_vfnsum(a_alias_f32, b_f32, a_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (__builtin_bit_cast(uint32_t, a_alias_f32[i]) != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfnsum (c==a)!\n", i);
        }
    }

    // Aliasing test: destination == second source (accumulator)
    if (!snrt_vfnsum(a_f32, b_alias_f32, b_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (__builtin_bit_cast(uint32_t, b_alias_f32[i]) != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfnsum (c==b)!\n", i);
        }
    }

    printf("Packed FP three-term difference: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
