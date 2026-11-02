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

    __attribute__((aligned(8))) double   c_f64[] = {3.0};
    __attribute__((aligned(8))) float    c_f32[] = {3.0f, 5.0f};
    __attribute__((aligned(8))) _Float16 c_f16[] = {3.0, 5.0, 7.0, 9.0};
    __attribute__((aligned(8))) uint8_t   c_f8[] = 
        {0x42, 0x45, 0x47, 0x49, 0x40, 0x44, 0x46, 0x48};
    // FP8: 3,    5,    7,   10,    2,    4,    6,    8.

    __attribute__((aligned(8))) double   d_f64[] = {0.0};
    __attribute__((aligned(8))) float    d_f32[] = {0.0f, 0.0f};
    __attribute__((aligned(8))) _Float16 d_f16[] = {0.0, 0.0, 0.0, 0.0};
    __attribute__((aligned(8))) uint8_t   d_f8[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // FP32 copies of the inputs, used as in-place (aliased) destinations.
    __attribute__((aligned(8))) float     a_alias_f32[] = {2.0f, 4.0f};
    __attribute__((aligned(8))) float     b_alias_f32[] = {2.0f, 3.0f};
    __attribute__((aligned(8))) float     c_alias_f32[] = {3.0f, 5.0f};

    
    const double   expected_f64[] = {-1.0};
    const float    expected_f32[] = {-1.0f, -7.0f};
    const _Float16 expected_f16[] = {-1.0, -7.0, -17.0, -31.0};
    const uint8_t   expected_f8[] =
                          {0xbc, 0xc7, 0xcc, 0xd0, 0xd3, 0xd5, 0xd7, 0xd9};
    // Exact c - a*b:        -1,   -7,  -17,  -30,  -58,  -80, -106, -152.
    // E5M2/RNE values:      -1,   -7,  -16,  -32,  -56,  -80, -112, -160.
    // 30 is a midpoint: ties to even selects 32 (0xd0).

    unsigned errors = 0;

    // FP64 uses scalar fnmsub.d: 3 - 2*2.
    if (!snrt_vfmre(a_f64, b_f64, c_f64, d_f64, FP64)) ++errors;
    if (d_f64[0] != expected_f64[0]) {
        ++errors;
        printf("Failed FP64 fnmsub!\n");
    }

    if (!snrt_vfmre(a_f32, b_f32, c_f32, d_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (d_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 vfmre!\n", i);
        }
    }

    if (!snrt_vfmre(a_f16, b_f16, c_f16, d_f16, FP16)) ++errors;
    for (unsigned i = 0; i < 4; ++i) {
        if (d_f16[i] != expected_f16[i]) {
            ++errors;
            printf("Failed %uth FP16 vfmre!\n", i);
        }
    }

    if (!snrt_vfmre(a_f8, b_f8, c_f8, d_f8, FP8)) ++errors;
    for (unsigned i = 0; i < 8; ++i) {
        if (d_f8[i] != expected_f8[i]) {
            ++errors;
            printf("Failed %uth FP8 vfmre!\n", i);
        }
    }

    // Aliasing test: destination == first source
    if (!snrt_vfmre(a_alias_f32, b_f32, c_f32, a_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (a_alias_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfmre (d==a)!\n", i);
        }
    }

    // Aliasing test: destination == second source
    if (!snrt_vfmre(a_f32, b_alias_f32, c_f32, b_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (b_alias_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfmre (d==b)!\n", i);
        }
    }

    // Aliasing test: destination == accumulator
    if (!snrt_vfmre(a_f32, b_f32, c_alias_f32, c_alias_f32, FP32)) ++errors;
    for (unsigned i = 0; i < 2; ++i) {
        if (c_alias_f32[i] != expected_f32[i]) {
            ++errors;
            printf("Failed %uth FP32 in-place vfmre (d==c)!\n", i);
        }
    }

    printf("Packed FP MRE: %s (%u errors)\n",
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
