// Edge cases of the vector-scalar addition on data in L1:
//  1. Only a subset of the compute cores runs (snrt_vs_add_core_l1 on a
//     slice of the vector, the others stay idle). Each core alone, and groups.
//  2. Many vector lengths through the multicore snrt_vs_add_l1: empty,
//     shorter than a word, exactly one word per core, one element more, ...
// Every precision is tested. The DMA core only takes part in the barriers.

#include "snrt.h"
#include "vector_scalar_l1.h"
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

// FP8 (E5M2) encodings of 0, 1, 2, ..., 8. All the sums below are exact.
static const uint8_t enc[9] = {0x00, 0x3c, 0x40, 0x42, 0x44,
                               0x45, 0x46, 0x47, 0x48};

// Core subsets (bit i = compute core i): each core alone, then groups.
static const unsigned masks[] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
                                 0x40, 0x80, 0x0F, 0xF0, 0xAA, 0xFE};
// Vector lengths for the subsets: 75 is a few words with a tail for every
// precision, 13 is less than two words for FP8.
static const unsigned subset_lengths[] = {75, 13};
// Vector lengths for the multicore function.
static const unsigned lengths[] = {0,  1,  2,  3,  4,  5,  7,  8,  9,  15,
                                   16, 17, 31, 32, 33, 63, 64, 65, 75};

// One precision: its buffers and the scalar.
typedef struct {
    precision_t prec;
    const char* name;
    const void* a;     // input, NMAX elements
    void* c;           // output, NMAX elements, then PAD bytes
    const uint8_t* e;  // expected output, NMAX elements
    const void* s;     // the scalar, one element
} Case;

// Half-precision arithmetic lives in its own function, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts.
__attribute__((noinline)) static void init_f16(_Float16* a, _Float16* expected,
                                               uint16_t scalar_bits,
                                               unsigned first, unsigned step) {
    _Float16 scalar;
    memcpy(&scalar, &scalar_bits, sizeof(scalar));
    for (unsigned i = first; i < NMAX; i += step) {
        a[i] = (_Float16)i;
        expected[i] = (_Float16)(a[i] + scalar);
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

// Check elements core, core + cores, ... (below n) of the output.
static unsigned check_share(const Case* k, unsigned mask, unsigned n,
                            unsigned core, unsigned cores) {
    const volatile uint8_t* got = (const volatile uint8_t*)k->c;
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
    const volatile uint8_t* got = (const volatile uint8_t*)k->c;
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

// The cores in `mask` share the vector (words split in order among them, the
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
    return snrt_vs_add_core_l1((const uint8_t*)k->a + offset, k->s,
                                    (uint8_t*)k->c + offset, elems, k->prec)
               ? 0
               : 1;
}

int main() {
    double*   a64 = ALLOC(double, NMAX);
    double*   c64 = ALLOC(double, NMAX);
    double*   e64 = ALLOC(double, NMAX);
    float*    a32 = ALLOC(float, NMAX);
    float*    c32 = ALLOC(float, NMAX);
    float*    e32 = ALLOC(float, NMAX);
    _Float16* a16 = ALLOC(_Float16, NMAX);
    _Float16* c16 = ALLOC(_Float16, NMAX);
    _Float16* e16 = ALLOC(_Float16, NMAX);
    uint8_t*  a8  = ALLOC(uint8_t, NMAX);
    uint8_t*  c8  = ALLOC(uint8_t, NMAX);
    uint8_t*  e8  = ALLOC(uint8_t, NMAX);
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    // One scalar element per precision. FP8: 1.0, the others: 0.5. The FP16
    // scalar is kept as raw bits, so no FP16 instruction runs outside the
    // compute cores.
    double   s64 = 0.5;
    float    s32 = 0.5f;
    uint16_t s16 = 0x3800;
    uint8_t  s8  = 0x3c;

    const Case cases[4] = {
        {FP64, "FP64", a64, c64, (const uint8_t*)e64, &s64},
        {FP32, "FP32", a32, c32, (const uint8_t*)e32, &s32},
        {FP16, "FP16", a16, c16, (const uint8_t*)e16, &s16},
        {FP8, "FP8", a8, c8, e8, &s8},
    };

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs and expected results.
    if (compute) {
        for (unsigned i = core; i < NMAX; i += cores) {
            a64[i] = i;
            a32[i] = i;
            a8[i] = enc[i % 8];
            e64[i] = a64[i] + s64;
            e32[i] = a32[i] + s32;
            e8[i] = enc[i % 8 + 1];
        }
        init_f16(a16, e16, s16, core, cores);
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
                if (core == 0) fill((uint8_t*)k->c, out_bytes, SENTINEL);
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
            if (core == 0) fill((uint8_t*)k->c, out_bytes, SENTINEL);
            snrt_cluster_hw_barrier();
            if (!snrt_vs_add_l1(k->a, k->s, k->c, n, k->prec)) ++errors;
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
        printf("Vector-scalar addition (L1) edge cases: %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
