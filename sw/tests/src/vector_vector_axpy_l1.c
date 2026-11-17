#include "snrt.h"
#include "vector_vector_l1.h"
#include <stdint.h>
#include <stdio.h>

// 75 elements. With 8 compute cores, every core gets at least one whole word
// in every precision (FP8: 9 words, FP16: 18, FP32: 37, FP64: 75), the split is
// uneven, and the last core also takes a partial word (3 elements for FP8 and
// FP16, 1 for FP32), so the tail is exercised too. The packed loop does four
// words per iteration (and the words that do not fill a group one at a time):
// with 8 cores, FP32 and FP64 use both loops, FP16 and FP8 only the
// one-word-at-a-time loop. The single-core runs (18 words of FP16) use both.
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

// z = alpha * x + y. FP64, FP32, FP16: alpha = 3, x[i] = i and y[i] = i % 7, so
// the results are 3 * i + i % 7, all exact. FP8 (E5M2): alpha = 2,
// x[i] = i % 8 and y[i] = 2 * ((i / 8) % 2), so the results are the even
// numbers 0, ..., 16, also exact. enc holds the encodings of 0, ..., 7 and
// enc_even those of 0, 2, ..., 16.
static const uint8_t enc[8] = {0x00, 0x3c, 0x40, 0x42, 0x44, 0x45, 0x46, 0x47};
static const uint8_t enc_even[9] = {0x00, 0x40, 0x44, 0x46, 0x48,
                                    0x49, 0x4a, 0x4b, 0x4c};

// Half-precision arithmetic lives in its own function, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts. The compiler
// would otherwise be free to run FP16 setup code on every core.
__attribute__((noinline)) static void init_f16(_Float16* x, _Float16* y,
                                               _Float16* expected,
                                               unsigned first, unsigned step) {
    for (unsigned i = first; i < N; i += step) {
        x[i] = (_Float16)i;
        y[i] = (_Float16)(i % 7);
        expected[i] = (_Float16)(3 * i + i % 7);
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
    double*   x64 = ALLOC(double, N);
    double*   y64 = ALLOC(double, N);
    double*   z64 = ALLOC(double, N);
    double*   e64 = ALLOC(double, N);
    float*    x32 = ALLOC(float, N);
    float*    y32 = ALLOC(float, N);
    float*    z32 = ALLOC(float, N);
    float*    e32 = ALLOC(float, N);
    _Float16* x16 = ALLOC(_Float16, N);
    _Float16* y16 = ALLOC(_Float16, N);
    _Float16* z16 = ALLOC(_Float16, N);
    _Float16* e16 = ALLOC(_Float16, N);
    uint8_t*  x8  = ALLOC(uint8_t, N);
    uint8_t*  y8  = ALLOC(uint8_t, N);
    uint8_t*  z8  = ALLOC(uint8_t, N);
    uint8_t*  e8  = ALLOC(uint8_t, N);
    // One private FP16 output region per compute core, for the single-core
    // test.
    _Float16* p16 = ALLOC(_Float16, STRIDE16 * snrt_cluster_compute_core_num());
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    // One alpha element per precision. FP8: 2.0, the others: 3.0. The FP16
    // alpha is kept as raw bits, so no FP16 instruction runs outside the
    // compute cores.
    double   a64 = 3.0;
    float    a32 = 3.0f;
    uint16_t a16 = 0x4200;
    uint8_t  a8  = 0x40;

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs and the expected
    // results (elements core, core + cores, ...). The DMA core does not
    // compute, it only takes part in the barriers.
    if (compute) {
        for (unsigned i = core; i < N; i += cores) {
            x64[i] = i;
            y64[i] = i % 7;
            e64[i] = 3 * i + i % 7;
            x32[i] = i;
            y32[i] = i % 7;
            e32[i] = 3 * i + i % 7;
            x8[i] = enc[i % 8];
            y8[i] = enc_even[((i / 8) % 2)];
            e8[i] = enc_even[i % 8 + (i / 8) % 2];
        }
        init_f16(x16, y16, e16, core, cores);
    }
    // Core 0 sets the sentinels after the buffers.
    if (core == 0) {
        fill((uint8_t*)x64 + N * 8, PAD, SENTINEL);
        fill((uint8_t*)y64 + N * 8, PAD, SENTINEL);
        fill((uint8_t*)x32 + N * 4, PAD, SENTINEL);
        fill((uint8_t*)y32 + N * 4, PAD, SENTINEL);
        fill((uint8_t*)x16 + N * 2, PAD, SENTINEL);
        fill((uint8_t*)y16 + N * 2, PAD, SENTINEL);
        fill((uint8_t*)x8 + N * 1, PAD, SENTINEL);
        fill((uint8_t*)y8 + N * 1, PAD, SENTINEL);
        fill((uint8_t*)z64, N * 8 + PAD, SENTINEL);
        fill((uint8_t*)z32, N * 4 + PAD, SENTINEL);
        fill((uint8_t*)z16, N * 2 + PAD, SENTINEL);
        fill((uint8_t*)z8, N * 1 + PAD, SENTINEL);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    // Every core calls the multicore version; the work is split among the
    // compute cores. Out of place: z = alpha * x + y, y is kept.
    if (!snrt_vv_axpy_l1(&a64, x64, y64, z64, N, FP64)) ++errors;
    if (!snrt_vv_axpy_l1(&a32, x32, y32, z32, N, FP32)) ++errors;
    if (!snrt_vv_axpy_l1(&a16, x16, y16, z16, N, FP16)) ++errors;
    if (!snrt_vv_axpy_l1(&a8, x8, y8, z8, N, FP8)) ++errors;
    snrt_cluster_hw_barrier();

    // Each compute core checks a share of the result (interleaved, so it is
    // mostly data written by other cores).
    if (compute) {
        errors += check_share("FP64 vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)z64, (const uint8_t*)e64, 8);
        errors += check_share("FP32 vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)z32, (const uint8_t*)e32, 4);
        errors += check_share("FP16 vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)z16, (const uint8_t*)e16, 2);
        errors += check_share("FP8 vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)z8, e8, 1);
    }
    if (core == 0) {
        errors += check_pad("FP64 vv_axpy_l1", core, (volatile uint8_t*)z64, 8);
        errors += check_pad("FP32 vv_axpy_l1", core, (volatile uint8_t*)z32, 4);
        errors += check_pad("FP16 vv_axpy_l1", core, (volatile uint8_t*)z16, 2);
        errors += check_pad("FP8 vv_axpy_l1", core, (volatile uint8_t*)z8, 1);
    }

    // Single-core version: every compute core runs it on its own, on the whole
    // vectors, into its own output region (FP16 data). y is still intact here.
    if (compute) {
        _Float16* mine = p16 + core * STRIDE16;
        fill((uint8_t*)mine, STRIDE16 * 2, SENTINEL);
        if (!snrt_vv_axpy_core_l1(&a16, x16, y16, mine, N, FP16)) ++errors;
        errors += check_share("FP16 vv_axpy_core_l1", core, 0, 1,
                              (volatile uint8_t*)mine, (const uint8_t*)e16, 2);
        errors += check_pad("FP16 vv_axpy_core_l1", core,
                            (volatile uint8_t*)mine, 2);
    }
    snrt_cluster_hw_barrier();

    // In place: z is y (the usual BLAS axpy). This also shows that the
    // out-of-place runs above left y intact: the results are computed from it.
    if (!snrt_vv_axpy_l1(&a64, x64, y64, y64, N, FP64)) ++errors;
    if (!snrt_vv_axpy_l1(&a32, x32, y32, y32, N, FP32)) ++errors;
    if (!snrt_vv_axpy_l1(&a16, x16, y16, y16, N, FP16)) ++errors;
    if (!snrt_vv_axpy_l1(&a8, x8, y8, y8, N, FP8)) ++errors;
    snrt_cluster_hw_barrier();

    if (compute) {
        errors += check_share("FP64 in-place vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)y64, (const uint8_t*)e64, 8);
        errors += check_share("FP32 in-place vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)y32, (const uint8_t*)e32, 4);
        errors += check_share("FP16 in-place vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)y16, (const uint8_t*)e16, 2);
        errors += check_share("FP8 in-place vv_axpy_l1", core, core, cores,
                              (volatile uint8_t*)y8, e8, 1);
    }
    if (core == 0) {
        errors += check_pad("FP64 in-place vv_axpy_l1", core,
                            (volatile uint8_t*)y64, 8);
        errors += check_pad("FP32 in-place vv_axpy_l1", core,
                            (volatile uint8_t*)y32, 4);
        errors += check_pad("FP16 in-place vv_axpy_l1", core,
                            (volatile uint8_t*)y16, 2);
        errors += check_pad("FP8 in-place vv_axpy_l1", core,
                            (volatile uint8_t*)y8, 1);
    }

    // Core 0 collects the error counts and reports.
    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();

    if (core == 0) {
        unsigned total = 0;
        for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
            total += ((volatile unsigned*)core_errors)[c];
        printf("Vector-vector axpy (L1): %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
