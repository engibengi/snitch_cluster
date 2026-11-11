#include "snrt.h"
#include "vector_scalar_l1.h"
#include <stdint.h>
#include <stdio.h>

// 75 elements. With 8 compute cores, every core gets at least one whole word
// in every precision (FP8: 9 words, FP16: 18, FP32: 37, FP64: 75), the split is
// uneven, and the last core also takes a partial word (3 elements for FP8 and
// FP16, 1 for FP32), so the tail is exercised too.
#define N 75
// Sentinel bytes after each buffer, to catch writes past the last element.
#define PAD 8
#define SENTINEL 0xA5
// Elements between the starts of the per-core FP16 regions: a multiple of 4
// (eight-byte alignment) with room for the PAD bytes after the N elements.
#define STRIDE16 80

// Every core must make the same allocation calls in the same order.
#define ALLOC(type, count) \
    ((type*)snrt_l1_alloc_cluster_local((count) * sizeof(type) + PAD, 8))

// FP8 (E5M2) encodings of 0, 1, 2, ..., 8. All the sums below are exact.
static const uint8_t enc[9] = {0x00, 0x3c, 0x40, 0x42, 0x44,
                               0x45, 0x46, 0x47, 0x48};

// Half-precision arithmetic lives in its own function, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts. The compiler
// would otherwise be free to run FP16 setup code on every core.
__attribute__((noinline)) static void init_f16(_Float16* a, _Float16* expected,
                                               uint16_t scalar_bits,
                                               unsigned first, unsigned step) {
    _Float16 scalar;
    memcpy(&scalar, &scalar_bits, sizeof(scalar));
    for (unsigned i = first; i < N; i += step) {
        a[i] = (_Float16)i;
        expected[i] = (_Float16)(a[i] + scalar);
    }
}

static void fill(volatile uint8_t* p, unsigned bytes, uint8_t value) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = value;
}

// Check elements first, first + step, ... of a buffer, byte by byte.
static unsigned check_share(const char* name, unsigned core, unsigned first,
                            unsigned step, const volatile uint8_t* got,
                            const uint8_t* want, unsigned elem_size) {
    unsigned errors = 0;
    for (unsigned i = first; i < N; i += step) {
        for (unsigned b = 0; b < elem_size; ++b) {
            const unsigned k = i * elem_size + b;
            if (got[k] != want[k]) {
                ++errors;
                printf("core %u: %s element %u byte %u: expected 0x%02x, "
                       "got 0x%02x\n",
                       core, name, i, b, (unsigned)want[k], (unsigned)got[k]);
            }
        }
    }
    return errors;
}

// Check the sentinel bytes after the N elements.
static unsigned check_pad(const char* name, unsigned core,
                          const volatile uint8_t* got, unsigned elem_size) {
    unsigned errors = 0;
    for (unsigned k = N * elem_size; k < N * elem_size + PAD; ++k) {
        if (got[k] != SENTINEL) {
            ++errors;
            printf("core %u: %s wrote past the end (byte %u)\n", core, name,
                   k);
        }
    }
    return errors;
}

int main() {
    double*   a64 = ALLOC(double, N);
    double*   c64 = ALLOC(double, N);
    double*   e64 = ALLOC(double, N);
    float*    a32 = ALLOC(float, N);
    float*    c32 = ALLOC(float, N);
    float*    e32 = ALLOC(float, N);
    _Float16* a16 = ALLOC(_Float16, N);
    _Float16* c16 = ALLOC(_Float16, N);
    _Float16* e16 = ALLOC(_Float16, N);
    uint8_t*  a8  = ALLOC(uint8_t, N);
    uint8_t*  c8  = ALLOC(uint8_t, N);
    uint8_t*  e8  = ALLOC(uint8_t, N);
    // One private FP16 output region per compute core, for the single-core test.
    _Float16* p16 = ALLOC(_Float16, STRIDE16 * snrt_cluster_compute_core_num());
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    // One scalar element per precision. FP8: 1.0, the others: 0.5. The FP16
    // scalar is kept as raw bits, so no FP16 instruction runs outside the
    // compute cores.
    double   s64 = 0.5;
    float    s32 = 0.5f;
    uint16_t s16 = 0x3800;
    uint8_t  s8  = 0x3c;

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs and the expected
    // results (elements core, core + cores, ...). The DMA core does not
    // compute, it only takes part in the barriers.
    if (compute) {
        for (unsigned i = core; i < N; i += cores) {
            a64[i] = i;
            a32[i] = i;
            a8[i] = enc[i % 8];
            e64[i] = a64[i] + s64;
            e32[i] = a32[i] + s32;
            e8[i] = enc[i % 8 + 1];
        }
        init_f16(a16, e16, s16, core, cores);
    }
    // Core 0 sets the sentinels after the buffers.
    if (core == 0) {
        fill((uint8_t*)a64 + N * 8, PAD, SENTINEL);
        fill((uint8_t*)a32 + N * 4, PAD, SENTINEL);
        fill((uint8_t*)a16 + N * 2, PAD, SENTINEL);
        fill((uint8_t*)a8 + N * 1, PAD, SENTINEL);
        fill((uint8_t*)c64, N * 8 + PAD, SENTINEL);
        fill((uint8_t*)c32, N * 4 + PAD, SENTINEL);
        fill((uint8_t*)c16, N * 2 + PAD, SENTINEL);
        fill((uint8_t*)c8, N * 1 + PAD, SENTINEL);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    // Every core calls the multicore version; the work is split among the
    // compute cores.
    if (!snrt_vec_add_scalar_l1(a64, &s64, c64, N, FP64)) ++errors;
    if (!snrt_vec_add_scalar_l1(a32, &s32, c32, N, FP32)) ++errors;
    if (!snrt_vec_add_scalar_l1(a16, &s16, c16, N, FP16)) ++errors;
    if (!snrt_vec_add_scalar_l1(a8, &s8, c8, N, FP8)) ++errors;
    snrt_cluster_hw_barrier();

    // Each compute core checks a share of the result (interleaved, so it is
    // mostly data written by other cores).
    if (compute) {
        errors += check_share("FP64 vec_add_scalar", core, core, cores,
                              (volatile uint8_t*)c64, (const uint8_t*)e64, 8);
        errors += check_share("FP32 vec_add_scalar", core, core, cores,
                              (volatile uint8_t*)c32, (const uint8_t*)e32, 4);
        errors += check_share("FP16 vec_add_scalar", core, core, cores,
                              (volatile uint8_t*)c16, (const uint8_t*)e16, 2);
        errors += check_share("FP8 vec_add_scalar", core, core, cores,
                              (volatile uint8_t*)c8, e8, 1);
    }
    if (core == 0) {
        errors += check_pad("FP64 vec_add_scalar", core, (volatile uint8_t*)c64, 8);
        errors += check_pad("FP32 vec_add_scalar", core, (volatile uint8_t*)c32, 4);
        errors += check_pad("FP16 vec_add_scalar", core, (volatile uint8_t*)c16, 2);
        errors += check_pad("FP8 vec_add_scalar", core, (volatile uint8_t*)c8, 1);
    }

    // In-place: the destination is the input (FP32 data).
    if (!snrt_vec_add_scalar_l1(a32, &s32, a32, N, FP32)) ++errors;
    snrt_cluster_hw_barrier();

    if (compute)
        errors += check_share("FP32 in-place vec_add_scalar", core, core, cores,
                              (volatile uint8_t*)a32, (const uint8_t*)e32, 4);
    if (core == 0)
        errors += check_pad("FP32 in-place vec_add_scalar", core,
                            (volatile uint8_t*)a32, 4);

    // Single-core version: every compute core runs it on its own, on the whole
    // vector, into its own output region (FP16 data).
    if (compute) {
        _Float16* mine = p16 + core * STRIDE16;
        fill((uint8_t*)mine, STRIDE16 * 2, SENTINEL);
        if (!snrt_vec_add_scalar_core_l1(a16, &s16, mine, N, FP16)) ++errors;
        errors += check_share("FP16 vec_add_scalar_core", core, 0, 1,
                              (volatile uint8_t*)mine, (const uint8_t*)e16, 2);
        errors += check_pad("FP16 vec_add_scalar_core", core,
                            (volatile uint8_t*)mine, 2);
    }

    // Core 0 collects the error counts and reports.
    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();

    if (core == 0) {
        unsigned total = 0;
        for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
            total += ((volatile unsigned*)core_errors)[c];
        printf("Vector-scalar addition (L1): %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
