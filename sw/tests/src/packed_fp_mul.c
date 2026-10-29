#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

int main() {
    // Test only one core, it's enough
    if (snrt_cluster_core_idx() != 0) return 0;

    // Eight-byte words, need to be aligned for access.
    __attribute__((aligned(8))) double    a_f64[] = {2.0};
    __attribute__((aligned(8))) float     a_f32[] = {2.0f, 4.0f};
    __attribute__((aligned(8))) _Float16  a_f16[] = {2.0, 4.0, 6.0, 8.0};
    __attribute__((aligned(8))) uint8_t    a_f8[] =
        {0x40, 0x44, 0x46, 0x48, 0x49, 0x4a, 0x4b, 0x4c};
    // FP8: 2,    4,    6,    8,   10,   12,   14,   16.

    __attribute__((aligned(8))) double    b_f64[] = {2.0};
    __attribute__((aligned(8))) float     b_f32[] = {2.0f, 3.0f};
    __attribute__((aligned(8))) _Float16  b_f16[] = {2.0, 3.0, 4.0, 5.0};
    __attribute__((aligned(8))) uint8_t    b_f8[] =
        {0x40, 0x42, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49};
    // FP8: 2,    3,    4,    5,    6,    7,    8,   10.

    __attribute__((aligned(8))) double    c_f64[] = {0.0};
    __attribute__((aligned(8))) float     c_f32[] = {0.0f, 0.0f};
    __attribute__((aligned(8))) _Float16  c_f16[] = {0.0, 0.0, 0.0, 0.0};
    __attribute__((aligned(8))) uint8_t    c_f8[] = {0, 0, 0, 0, 0, 0, 0, 0};

    const double    expected_f64[] = {4.0};
    const float     expected_f32[] = {4.0f, 12.0f};
    const _Float16  expected_f16[] = {4.0, 12.0, 24.0, 40.0};
    const uint8_t    expected_f8[] =
                        {0x44, 0x4a, 0x4e, 0x51, 0x54, 0x55, 0x57, 0x59};
    // Exact products:      4,   12,   24,   40,   60,   84,  112,  160.
    // E5M2/RNE rounds 60 to 64 (midpoint, ties to even) and 84 to 80.
    unsigned errors = 0;

    // FP64 test
    if (!snrt_vfmul(a_f64, b_f64, c_f64, FP64)) ++errors;
    if (c_f64[0] != expected_f64[0]) {
        ++errors;
        printf("Failed FP64 fmul!\n");
    }

    // FP32 test
    if (!snrt_vfmul(a_f32, b_f32, c_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (c_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 vfmul!\n", i);
        }
    }

    // FP16 test
    if (!snrt_vfmul(a_f16, b_f16, c_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        if (c_f16[i] != expected_f16[i]) {
            ++errors;
            printf("Failed %uth FP16 vfmul!\n", i);
        }
    }

    //  FP8 test
    if (!snrt_vfmul(a_f8, b_f8, c_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (c_f8[i] != expected_f8[i]) {
            ++errors;
            printf("Failed %uth FP8 vfmul!\n", i);
        }
    }

    printf("Packed FP multiplication: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
