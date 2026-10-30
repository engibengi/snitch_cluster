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

    __attribute__((aligned(8))) double    b_f64[] = {0.0};
    __attribute__((aligned(8))) float     b_f32[] = {0.0f, 0.0f};
    __attribute__((aligned(8))) _Float16  b_f16[] = {0.0, 0.0, 0.0, 0.0};
    __attribute__((aligned(8))) uint8_t    b_f8[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Independently rounded square roots, assuming startup RNE.
    const double    expected_f64[] = {0x1.6a09e667f3bcdp+0};
    const float     expected_f32[] = {0x1.6a09e60000000p+0f, 0x1.0000000000000p+1f};
    const _Float16  expected_f16[] = {1.4140625, 2.0, 2.44921875, 2.828125};
    const uint8_t    expected_f8[] =
        {0x3e, 0x40, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44};
    // FP8:1.5,   2,  2.5,    3,    3,  3.5,  3.5,    4.
    unsigned errors = 0;

    // FP64 test
    if (!snrt_vfsqrt(a_f64, b_f64, FP64)) ++errors;
    if (b_f64[0] != expected_f64[0]) {
        ++errors;
        printf("Failed FP64 fsqrt!\n");
    }

    // FP32 test
    if (!snrt_vfsqrt(a_f32, b_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (b_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 vfsqrt!\n", i);
        }
    }

    // FP16 test
    if (!snrt_vfsqrt(a_f16, b_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        if (b_f16[i] != expected_f16[i]) {
            ++errors;
            printf("Failed %uth FP16 vfsqrt!\n", i);
        }
    }

    // FP8 test
    if (snrt_vfsqrt(a_f8, b_f8, FP8)) {
        ++errors;
        printf("Somehow ran FP8 vfsqrt?\n");
    }
    /*  This test is disabled, as vfsqrt.b is currently unsupported!
    if (!snrt_vfsqrt(a_f8, b_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (b_f8[i] != expected_f8[i]) {
            ++errors;
            printf("Failed %uth FP8 vfsqrt!\n", i);
            printf("Expected = %u, Returned = %u.\n",\
                   (unsigned int)expected_f8[i],\
                   (unsigned int)b_f8[i]\
                  );
        }
    }
    */

    printf("Packed FP square root: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
