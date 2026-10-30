#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

// Assumes startup RNE and ordinary FP16/FP8 mode; no CSR changes here.
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

    const double    expected_f64[] = {1.0};
    // FP32/RNE: 1 and 4/3 rounded once to FP32.
    const float     expected_f32[] = {1.0f, 0x1.555556p+0f};
    // Exact FP16 values after RNE of 1, 4/3, 3/2 and 8/5.
    const _Float16  expected_f16[] = {1.0, 1.3330078125, 1.5, 1.599609375};
    const uint8_t    expected_f8[] =
        {0x3c, 0x3d, 0x3e, 0x3e, 0x3f, 0x3f, 0x3f, 0x3e};
    // FP8: 1,  4/3,  3/2,  8/5,  5/3, 12/7,  7/4,  8/5.
    // RNE: 1, 1.25,  1.5,  1.5, 1.75, 1.75, 1.75,  1.5.
    unsigned errors = 0;

    // FP64 test
    if (!snrt_vfdiv(a_f64, b_f64, c_f64, FP64)) ++errors;
    if (c_f64[0] != expected_f64[0]) {
        ++errors;
        printf("Failed FP64 fdiv!\n");
    }

    // FP32 test
    if (!snrt_vfdiv(a_f32, b_f32, c_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (c_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 vfdiv!\n", i);
        }
    }

    // FP16 test
    if (!snrt_vfdiv(a_f16, b_f16, c_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        if (c_f16[i] != expected_f16[i]) {
            ++errors;
            printf("Failed %uth FP16 vfdiv!\n", i);
        }
    }

    //  FP8 test
    /*  This test is disabled, as vfdiv.b is currently unsupported!
    if (!snrt_vfdiv(a_f8, b_f8, c_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (c_f8[i] != expected_f8[i]) {
            ++errors;
            printf("Failed %uth FP8 vfdiv!\n", i);
            printf("Expected = %u, Returned = %u.\n",\
                   (unsigned int)expected_f8[i],\
                   (unsigned int)c_f8[i]\
                  );
        }
    }
    */


    printf("Packed FP division: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
