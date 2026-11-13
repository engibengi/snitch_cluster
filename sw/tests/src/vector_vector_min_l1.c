#include "snrt.h"
#include "vector_vector_l1.h"
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

// The inputs are a[i] = i % 8 - 4 and b[i] = (i / 3) % 8 - 4 (FP64, FP32,
// FP16), integers from -4 to 3, so that either one can be the minimum.
// FP8 (E5M2) uses the same values; enc holds the encodings of -4, ..., 3,
// in order. The results are exact: they are one of the two operands.
static const uint8_t enc[8] = {0xc4, 0xc2, 0xc0, 0xbc,
                               0x00, 0x3c, 0x40, 0x42};

// The smaller of two values.
static unsigned pick(unsigned x, unsigned y) { return x < y ? x : y; }

// Half-precision arithmetic lives in its own function, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts. The compiler
// would otherwise be free to run FP16 setup code on every core.
__attribute__((noinline)) static void init_f16(_Float16* a, _Float16* b,
                                               _Float16* expected,
                                               unsigned first, unsigned step) {
    for (unsigned i = first; i < N; i += step) {
        a[i] = (_Float16)((int)(i % 8) - 4);
        b[i] = (_Float16)((int)((i / 3) % 8) - 4);
        expected[i] = (_Float16)((int)pick(i % 8, (i / 3) % 8) - 4);
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
    double*   b64 = ALLOC(double, N);
    double*   c64 = ALLOC(double, N);
    double*   e64 = ALLOC(double, N);
    float*    a32 = ALLOC(float, N);
    float*    b32 = ALLOC(float, N);
    float*    c32 = ALLOC(float, N);
    float*    e32 = ALLOC(float, N);
    _Float16* a16 = ALLOC(_Float16, N);
    _Float16* b16 = ALLOC(_Float16, N);
    _Float16* c16 = ALLOC(_Float16, N);
    _Float16* e16 = ALLOC(_Float16, N);
    uint8_t*  a8  = ALLOC(uint8_t, N);
    uint8_t*  b8  = ALLOC(uint8_t, N);
    uint8_t*  c8  = ALLOC(uint8_t, N);
    uint8_t*  e8  = ALLOC(uint8_t, N);
    // One private FP16 output region per compute core, for the single-core
    // test.
    _Float16* p16 = ALLOC(_Float16, STRIDE16 * snrt_cluster_compute_core_num());
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs and the expected
    // results (elements core, core + cores, ...). The DMA core does not
    // compute, it only takes part in the barriers.
    if (compute) {
        for (unsigned i = core; i < N; i += cores) {
            a64[i] = (int)(i % 8) - 4;
            b64[i] = (int)((i / 3) % 8) - 4;
            e64[i] = (int)pick(i % 8, (i / 3) % 8) - 4;
            a32[i] = (int)(i % 8) - 4;
            b32[i] = (int)((i / 3) % 8) - 4;
            e32[i] = (int)pick(i % 8, (i / 3) % 8) - 4;
            a8[i] = enc[(i % 8)];
            b8[i] = enc[((i / 3) % 8)];
            e8[i] = enc[pick(i % 8, (i / 3) % 8)];
        }
        init_f16(a16, b16, e16, core, cores);
    }
    // Core 0 sets the sentinels after the buffers.
    if (core == 0) {
        fill((uint8_t*)a64 + N * 8, PAD, SENTINEL);
        fill((uint8_t*)b64 + N * 8, PAD, SENTINEL);
        fill((uint8_t*)a32 + N * 4, PAD, SENTINEL);
        fill((uint8_t*)b32 + N * 4, PAD, SENTINEL);
        fill((uint8_t*)a16 + N * 2, PAD, SENTINEL);
        fill((uint8_t*)b16 + N * 2, PAD, SENTINEL);
        fill((uint8_t*)a8 + N * 1, PAD, SENTINEL);
        fill((uint8_t*)b8 + N * 1, PAD, SENTINEL);
        fill((uint8_t*)c64, N * 8 + PAD, SENTINEL);
        fill((uint8_t*)c32, N * 4 + PAD, SENTINEL);
        fill((uint8_t*)c16, N * 2 + PAD, SENTINEL);
        fill((uint8_t*)c8, N * 1 + PAD, SENTINEL);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    // Every core calls the multicore version; the work is split among the
    // compute cores.
    if (!snrt_vv_min_l1(a64, b64, c64, N, FP64)) ++errors;
    if (!snrt_vv_min_l1(a32, b32, c32, N, FP32)) ++errors;
    if (!snrt_vv_min_l1(a16, b16, c16, N, FP16)) ++errors;
    if (!snrt_vv_min_l1(a8, b8, c8, N, FP8)) ++errors;
    snrt_cluster_hw_barrier();

    // Each compute core checks a share of the result (interleaved, so it is
    // mostly data written by other cores).
    if (compute) {
        errors += check_share("FP64 vv_min_l1", core, core, cores,
                              (volatile uint8_t*)c64, (const uint8_t*)e64, 8);
        errors += check_share("FP32 vv_min_l1", core, core, cores,
                              (volatile uint8_t*)c32, (const uint8_t*)e32, 4);
        errors += check_share("FP16 vv_min_l1", core, core, cores,
                              (volatile uint8_t*)c16, (const uint8_t*)e16, 2);
        errors += check_share("FP8 vv_min_l1", core, core, cores,
                              (volatile uint8_t*)c8, e8, 1);
    }
    if (core == 0) {
        errors += check_pad("FP64 vv_min_l1", core, (volatile uint8_t*)c64, 8);
        errors += check_pad("FP32 vv_min_l1", core, (volatile uint8_t*)c32, 4);
        errors += check_pad("FP16 vv_min_l1", core, (volatile uint8_t*)c16, 2);
        errors += check_pad("FP8 vv_min_l1", core, (volatile uint8_t*)c8, 1);
    }

    // In-place, the destination is the first input (c == a), FP32 data.
    if (!snrt_vv_min_l1(a32, b32, a32, N, FP32)) ++errors;
    snrt_cluster_hw_barrier();
    if (compute) {
        errors += check_share("FP32 in-place (c==a) vv_min_l1", core, core,
                              cores, (volatile uint8_t*)a32,
                              (const uint8_t*)e32, 4);
        // Restore this core's share of a32 for the next test. Each core only
        // touches the elements it has just checked.
        for (unsigned i = core; i < N; i += cores) a32[i] = (int)(i % 8) - 4;
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    // In-place, the destination is the second input (c == b), FP32 data.
    if (!snrt_vv_min_l1(a32, b32, b32, N, FP32)) ++errors;
    snrt_cluster_hw_barrier();
    if (compute)
        errors += check_share("FP32 in-place (c==b) vv_min_l1", core, core,
                              cores, (volatile uint8_t*)b32,
                              (const uint8_t*)e32, 4);
    if (core == 0) {
        errors += check_pad("FP32 in-place (c==a) vv_min_l1", core,
                            (volatile uint8_t*)a32, 4);
        errors += check_pad("FP32 in-place (c==b) vv_min_l1", core,
                            (volatile uint8_t*)b32, 4);
    }

    // Single-core version: every compute core runs it on its own, on the whole
    // vectors, into its own output region (FP16 data).
    if (compute) {
        _Float16* mine = p16 + core * STRIDE16;
        fill((uint8_t*)mine, STRIDE16 * 2, SENTINEL);
        if (!snrt_vv_min_core_l1(a16, b16, mine, N, FP16)) ++errors;
        errors += check_share("FP16 vv_min_core_l1", core, 0, 1,
                              (volatile uint8_t*)mine, (const uint8_t*)e16, 2);
        errors += check_pad("FP16 vv_min_core_l1", core,
                            (volatile uint8_t*)mine, 2);
    }

    // Core 0 collects the error counts and reports.
    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();

    if (core == 0) {
        unsigned total = 0;
        for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
            total += ((volatile unsigned*)core_errors)[c];
        printf("Vector-vector minimum (L1): %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
