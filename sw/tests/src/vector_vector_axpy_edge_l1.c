// Edge cases of the vector-vector axpy (z = alpha * x + y) on data in L1:
//  1. Only a subset of the compute cores runs (snrt_vv_axpy_core_l1 on a slice
//     of the vectors, the others stay idle). Each core alone, and groups.
//  2. Many vector lengths through the multicore snrt_vv_axpy_l1: empty,
//     shorter than a word, exactly one word per core, one element more, ...
// Every precision is tested, out of place (z is a separate output). The DMA
// core only takes part in the barriers.

#include "snrt.h"
#include "vector_vector_l1.h"
#include <stdint.h>
#include <stdio.h>

// Capacity of every buffer, in elements.
#define NMAX 75
// Sentinel bytes after each output buffer, to catch writes past the end.
#define PAD 8
#define SENTINEL 0xA5

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

// Core subsets (bit i = compute core i): each core alone, then groups.
static const unsigned masks[] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
                                 0x40, 0x80, 0x0F, 0xF0, 0xAA, 0xFE};
// Vector lengths for the subsets: 75 is a few words with a tail for every
// precision, 13 is less than two words for FP8.
static const unsigned subset_lengths[] = {75, 13};
// Vector lengths for the multicore function. Together with the subsets they
// cover words per core below, equal to and above the four-word group of the
// packed loop, with and without a remainder.
static const unsigned lengths[] = {0,  1,  2,  3,  4,  5,  7,  8,  9,  15,
                                   16, 17, 31, 32, 33, 63, 64, 65, 75};

// One precision: its buffers and alpha.
typedef struct {
    precision_t prec;
    const char* name;
    const void* x;         // first input, NMAX elements
    const void* y;         // second input, NMAX elements
    void* z;               // output, NMAX elements, then PAD bytes
    const uint8_t* e;      // expected output, NMAX elements
    const void* alpha;     // the scalar, one element
} Case;

// Half-precision arithmetic lives in its own function, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts.
__attribute__((noinline)) static void init_f16(_Float16* x, _Float16* y,
                                               _Float16* expected,
                                               unsigned first, unsigned step) {
    for (unsigned i = first; i < NMAX; i += step) {
        x[i] = (_Float16)i;
        y[i] = (_Float16)(i % 7);
        expected[i] = (_Float16)(3 * i + i % 7);
    }
}

static void fill(volatile uint8_t* p, unsigned bytes, uint8_t value) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = value;
}

static unsigned count_bits(unsigned x) {
    unsigned n = 0;
    for (; x; x >>= 1) n += x & 1;
    return n;
}

// Check elements core, core + cores, ... (below n) of y.
static unsigned check_share(const Case* k, unsigned mask, unsigned n,
                            unsigned core, unsigned cores) {
    const volatile uint8_t* got = (const volatile uint8_t*)k->z;
    const unsigned es = (unsigned)k->prec;
    unsigned errors = 0;
    for (unsigned i = core; i < n; i += cores) {
        for (unsigned b = 0; b < es; ++b) {
            const unsigned j = i * es + b;
            if (got[j] != k->e[j]) {
                ++errors;
                printf("%s mask 0x%02x n=%u core %u: element %u byte %u: "
                       "expected 0x%02x, got 0x%02x\n",
                       k->name, mask, n, core, i, b, (unsigned)k->e[j],
                       (unsigned)got[j]);
            }
        }
    }
    return errors;
}

// Check that nothing after the first n elements was written.
static unsigned check_untouched(const Case* k, unsigned mask, unsigned n) {
    const volatile uint8_t* got = (const volatile uint8_t*)k->z;
    const unsigned es = (unsigned)k->prec;
    unsigned errors = 0;
    for (unsigned j = n * es; j < NMAX * es + PAD; ++j) {
        if (got[j] != SENTINEL) {
            ++errors;
            printf("%s mask 0x%02x n=%u: wrote past the end (byte %u)\n",
                   k->name, mask, n, j);
        }
    }
    return errors;
}

// The cores in `mask` share the vectors (words split in order among them, the
// last one also takes the partial word); each runs the single-core kernel on
// its slice. The other cores do nothing.
static unsigned run_subset(const Case* k, unsigned mask, unsigned n,
                           unsigned core) {
    if (!((mask >> core) & 1)) return 0;
    const unsigned active = count_bits(mask);
    const unsigned rank = count_bits(mask & ((1u << core) - 1));
    const size_t lanes = snrt_vec_lanes(k->prec);

    size_t first, count;
    snrt_vec_split(n / lanes, rank, active, &first, &count);
    size_t elems = count * lanes;
    if (rank == active - 1) elems += n % lanes;

    const size_t offset = first * sizeof(uint64_t);
    return snrt_vv_axpy_core_l1(k->alpha, (const uint8_t*)k->x + offset,
                                (const uint8_t*)k->y + offset,
                                (uint8_t*)k->z + offset, elems, k->prec)
               ? 0
               : 1;
}

int main() {
    double*   x64 = ALLOC(double, NMAX);
    double*   y64 = ALLOC(double, NMAX);
    double*   z64 = ALLOC(double, NMAX);
    double*   e64 = ALLOC(double, NMAX);
    float*    x32 = ALLOC(float, NMAX);
    float*    y32 = ALLOC(float, NMAX);
    float*    z32 = ALLOC(float, NMAX);
    float*    e32 = ALLOC(float, NMAX);
    _Float16* x16 = ALLOC(_Float16, NMAX);
    _Float16* y16 = ALLOC(_Float16, NMAX);
    _Float16* z16 = ALLOC(_Float16, NMAX);
    _Float16* e16 = ALLOC(_Float16, NMAX);
    uint8_t*  x8  = ALLOC(uint8_t, NMAX);
    uint8_t*  y8  = ALLOC(uint8_t, NMAX);
    uint8_t*  z8  = ALLOC(uint8_t, NMAX);
    uint8_t*  e8  = ALLOC(uint8_t, NMAX);
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    // One alpha element per precision. FP8: 2.0, the others: 3.0. The FP16
    // alpha is kept as raw bits, so no FP16 instruction runs outside the
    // compute cores.
    double   a64 = 3.0;
    float    a32 = 3.0f;
    uint16_t a16 = 0x4200;
    uint8_t  a8  = 0x40;

    const Case cases[4] = {
        {FP64, "FP64", x64, y64, z64, (const uint8_t*)e64, &a64},
        {FP32, "FP32", x32, y32, z32, (const uint8_t*)e32, &a32},
        {FP16, "FP16", x16, y16, z16, (const uint8_t*)e16, &a16},
        {FP8, "FP8", x8, y8, z8, e8, &a8},
    };

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs and expected results.
    if (compute) {
        for (unsigned i = core; i < NMAX; i += cores) {
            x64[i] = i;
            y64[i] = i % 7;
            e64[i] = 3 * i + i % 7;
            x32[i] = i;
            y32[i] = i % 7;
            e32[i] = 3 * i + i % 7;
            x8[i] = enc[i % 8];
            y8[i] = enc_even[(i / 8) % 2];
            e8[i] = enc_even[i % 8 + (i / 8) % 2];
        }
        init_f16(x16, y16, e16, core, cores);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    for (unsigned p = 0; p < 4; ++p) {
        const Case* k = &cases[p];
        const unsigned out_bytes = NMAX * (unsigned)k->prec + PAD;

        // 1. Subsets of the compute cores.
        for (unsigned m = 0; m < sizeof(masks) / sizeof(masks[0]); ++m) {
            const unsigned mask = masks[m];
            if (mask >> cores) continue;  // needs more cores than there are
            for (unsigned l = 0;
                 l < sizeof(subset_lengths) / sizeof(subset_lengths[0]); ++l) {
                const unsigned n = subset_lengths[l];
                if (core == 0) fill((uint8_t*)k->z, out_bytes, SENTINEL);
                snrt_cluster_hw_barrier();
                if (compute) errors += run_subset(k, mask, n, core);
                snrt_cluster_hw_barrier();
                if (compute) errors += check_share(k, mask, n, core, cores);
                if (core == 0) errors += check_untouched(k, mask, n);
                snrt_cluster_hw_barrier();
            }
        }

        // 2. Many lengths through the multicore function (mask 0xff: all).
        for (unsigned l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
            const unsigned n = lengths[l];
            if (core == 0) fill((uint8_t*)k->z, out_bytes, SENTINEL);
            snrt_cluster_hw_barrier();
            if (!snrt_vv_axpy_l1(k->alpha, k->x, k->y, k->z, n, k->prec))
                ++errors;
            snrt_cluster_hw_barrier();
            if (compute) errors += check_share(k, 0xff, n, core, cores);
            if (core == 0) errors += check_untouched(k, 0xff, n);
            snrt_cluster_hw_barrier();
        }
    }

    // Core 0 collects the error counts and reports.
    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();

    if (core == 0) {
        unsigned total = 0;
        for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
            total += ((volatile unsigned*)core_errors)[c];
        printf("Vector-vector axpy (L1) edge cases: %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
