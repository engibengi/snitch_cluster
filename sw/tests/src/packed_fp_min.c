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

    __attribute__((aligned(8))) double    b_f64[] = {3.0};
    __attribute__((aligned(8))) float     b_f32[] = {3.0f, 3.0f};
    __attribute__((aligned(8))) _Float16  b_f16[] = {3.0, 3.0, 7.0, 7.0};
    __attribute__((aligned(8))) uint8_t    b_f8[] =
        {0x42, 0x42, 0x47, 0x47, 0x48, 0x4b, 0x4a, 0x49};
    // FP8: 3,    3,    7,    7,    8,   14,   12,   10.

    __attribute__((aligned(8))) double    c_f64[] = {0.0};
    __attribute__((aligned(8))) float     c_f32[] = {0.0f, 0.0f};
    __attribute__((aligned(8))) _Float16  c_f16[] = {0.0, 0.0, 0.0, 0.0};
    __attribute__((aligned(8))) uint8_t    c_f8[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // FP32 copies of the inputs, used as in-place (aliased) destinations.
    __attribute__((aligned(8))) float     a_alias_f32[] = {2.0f, 4.0f};
    __attribute__((aligned(8))) float     b_alias_f32[] = {3.0f, 3.0f};

    const double    expected_f64[] = {2.0};
    const float     expected_f32[] = {2.0f, 3.0f};
    const _Float16  expected_f16[] = {2.0, 3.0, 6.0, 7.0};
    const uint8_t    expected_f8[] =
        {0x40, 0x42, 0x46, 0x47, 0x48, 0x4a, 0x4a, 0x49};
    // FP8: 2,    3,    6,    7,    8,   12,   12,   10.
    unsigned errors = 0;

    // FP64 test
    if (!snrt_vfmin(a_f64, b_f64, c_f64, FP64)) ++errors;
    if (c_f64[0] != expected_f64[0]) {
        ++errors;
        printf("Failed FP64 fmin!\n");
    }

    // FP32 test
    if (!snrt_vfmin(a_f32, b_f32, c_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (c_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 vfmin!\n", i);
        }
    }

    // FP16 test
    if (!snrt_vfmin(a_f16, b_f16, c_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        if (c_f16[i] != expected_f16[i]) {
            ++errors;
            printf("Failed %uth FP16 vfmin!\n", i);
        }
    }

    //  FP8 test
    if (!snrt_vfmin(a_f8, b_f8, c_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (c_f8[i] != expected_f8[i]) {
            ++errors;
            printf("Failed %uth FP8 vfmin!\n", i);
        }
    }

    // Aliasing test: destination == first source
    if (!snrt_vfmin(a_alias_f32, b_f32, a_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (a_alias_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfmin (c==a)!\n", i);
        }
    }

    // Aliasing test: destination == second source
    if (!snrt_vfmin(a_f32, b_alias_f32, b_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (b_alias_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfmin (c==b)!\n", i);
        }
    }

    printf("Packed FP minimum: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
