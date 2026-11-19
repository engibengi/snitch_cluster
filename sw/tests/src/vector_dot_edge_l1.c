// Edge cases of the dot product on data in L1:
//  1. Only a subset of the compute cores runs (snrt_vec_dot_core_l1 on a slice
//     of the vectors, the others stay idle): each core checks its own partial
//     result. Each core alone, and groups; slices can be empty.
//  2. Many vector lengths through the multicore snrt_vec_dot_l1: empty,
//     shorter than a word, exactly one word per core, one element more, ...
// Every precision and accumulation mode is tested. The DMA core only takes part
// in the barriers.

#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

// Capacity of every buffer, in elements.
#define NMAX 75
// Bytes of a result slot: the element, then sentinel bytes, to catch writes
// outside the element.
#define SLOT 16
#define SENTINEL 0xA5

// Every core must make the same allocation calls in the same order.
#define ALLOC(type, count) \
    ((type*)snrt_l1_alloc_cluster_local((count) * sizeof(type), 8))

// FP8 (E5M2) encodings of 0, 1, ..., 8.
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

// Test data, all small integers, so that every sum is exact whatever the order
// of the additions. Three kinds:
//  0: a = i % 5, b = i % 3 + 1 (FP64, FP32, FP16; sums up to 900).
//  1: a = i % 4, b = 1 + i % 2 (FP8 with wide accumulation: products up to 6,
//     accumulated in FP16, sums up to 450).
//  2: a = 1 at i % 11 == 0 and 0 elsewhere, b = 1 except b[22] = 2 (FP8 with
//     accumulation in FP8, which only holds the integers up to 8 exactly: the
//     products are 1 or 2, there are seven, and they add up to 8).
static int av(int kind, unsigned i) {
    if (kind == 0) return i % 5;
    if (kind == 1) return i % 4;
    return i % 11 == 0 ? 1 : 0;
}
static int bv(int kind, unsigned i) {
    if (kind == 0) return i % 3 + 1;
    if (kind == 1) return 1 + i % 2;
    return i == 22 ? 2 : 1;
}
// Reference dot product of elements [from, to).
static int dot_ref(int kind, unsigned from, unsigned to) {
    int s = 0;
    for (unsigned i = from; i < to; ++i) s += av(kind, i) * bv(kind, i);
    return s;
}

// Half-precision arithmetic lives in its own functions, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts.
__attribute__((noinline)) static void init_f16(_Float16* a, _Float16* b,
                                               unsigned first, unsigned step) {
    for (unsigned i = first; i < NMAX; i += step) {
        a[i] = (_Float16)av(0, i);
        b[i] = (_Float16)bv(0, i);
    }
}
__attribute__((noinline)) static uint16_t fp16_bits(int v) {
    _Float16 h = (_Float16)v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// The bytes of an integer value as an element of a precision.
static uint64_t encode(precision_t p, int v) {
    uint64_t bits = 0;
    if (p == FP64) {
        double d = v;
        memcpy(&bits, &d, sizeof(d));
    } else if (p == FP32) {
        float f = v;
        memcpy(&bits, &f, sizeof(f));
    } else if (p == FP16) {
        bits = fp16_bits(v);
    } else {
        bits = enc[v];
    }
    return bits;
}

static void fill(volatile uint8_t* p, unsigned bytes, uint8_t value) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = value;
}

static unsigned count_bits(unsigned x) {
    unsigned n = 0;
    for (; x; x >>= 1) n += x & 1;
    return n;
}

// One configuration: data precision, accumulation mode and data.
typedef struct {
    precision_t prec;
    snrt_acc_t acc;
    int kind;
    const char* name;
    const void* a;
    const void* b;
} Cfg;

// Check a result slot (the element, then sentinel bytes) against the dot
// product of elements [from, to).
static unsigned check_slot(const Cfg* k, unsigned mask, unsigned n,
                           unsigned core, const volatile uint8_t* got,
                           unsigned from, unsigned to) {
    const precision_t rp = snrt_vec_dot_result_precision(k->prec, k->acc);
    const unsigned es = (unsigned)rp;
    const uint64_t want = encode(rp, dot_ref(k->kind, from, to));
    unsigned errors = 0;
    for (unsigned b = 0; b < SLOT; ++b) {
        const uint8_t w = b < es ? (uint8_t)(want >> (8 * b)) : SENTINEL;
        if (got[b] != w) {
            ++errors;
            printf("%s mask 0x%02x n=%u core %u byte %u: expected 0x%02x, "
                   "got 0x%02x\n",
                   k->name, mask, n, core, b, (unsigned)w, (unsigned)got[b]);
        }
    }
    return errors;
}

// The cores in `mask` share the vectors (words split in order among them, the
// last one also takes the partial word); each runs the single-core kernel on
// its slice, into its own slot, and checks it. The other cores do nothing.
static unsigned run_subset(const Cfg* k, unsigned mask, unsigned n,
                           unsigned core, uint8_t* slot) {
    if (!((mask >> core) & 1)) return 0;
    const unsigned active = count_bits(mask);
    const unsigned rank = count_bits(mask & ((1u << core) - 1));
    const size_t lanes = snrt_vec_lanes(k->prec);

    size_t first, count;
    snrt_vec_split(n / lanes, rank, active, &first, &count);
    size_t elems = count * lanes;
    if (rank == active - 1) elems += n % lanes;

    const size_t start = first * lanes;
    const size_t offset = first * sizeof(uint64_t);
    unsigned errors = 0;
    fill(slot, SLOT, SENTINEL);
    if (!snrt_vec_dot_core_l1((const uint8_t*)k->a + offset,
                              (const uint8_t*)k->b + offset, slot, elems,
                              k->prec, k->acc))
        ++errors;
    return errors + check_slot(k, mask, n, core, (volatile uint8_t*)slot, start,
                               start + elems);
}

int main() {
    double*   a64 = ALLOC(double, NMAX);
    double*   b64 = ALLOC(double, NMAX);
    float*    a32 = ALLOC(float, NMAX);
    float*    b32 = ALLOC(float, NMAX);
    _Float16* a16 = ALLOC(_Float16, NMAX);
    _Float16* b16 = ALLOC(_Float16, NMAX);
    uint8_t*  a8s = ALLOC(uint8_t, NMAX + 8);  // kind 2 (FP8 in FP8)
    uint8_t*  b8s = ALLOC(uint8_t, NMAX + 8);
    uint8_t*  a8d = ALLOC(uint8_t, NMAX + 8);  // kind 1 (FP8, wide)
    uint8_t*  b8d = ALLOC(uint8_t, NMAX + 8);
    // Partial results of the cores, for the multicore function.
    void* scratch = snrt_l1_alloc_cluster_local(
        snrt_vec_reduce_scratch_bytes(), 8);
    // Result slot of the multicore function, and one per core for the subsets.
    uint8_t* res = ALLOC(uint8_t, SLOT);
    uint8_t* res_core = ALLOC(uint8_t, snrt_cluster_compute_core_num() * SLOT);
    // Error count of every core, collected by core 0 at the end.
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    const Cfg cfgs[6] = {
        {FP64, SNRT_ACC_SAME, 0, "FP64", a64, b64},
        {FP32, SNRT_ACC_SAME, 0, "FP32", a32, b32},
        {FP16, SNRT_ACC_SAME, 0, "FP16", a16, b16},
        {FP16, SNRT_ACC_WIDE, 0, "FP16 wide", a16, b16},
        {FP8, SNRT_ACC_SAME, 2, "FP8", a8s, b8s},
        {FP8, SNRT_ACC_WIDE, 1, "FP8 wide", a8d, b8d},
    };

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    // Every compute core prepares its share of the inputs.
    if (compute) {
        for (unsigned i = core; i < NMAX; i += cores) {
            a64[i] = av(0, i);
            b64[i] = bv(0, i);
            a32[i] = av(0, i);
            b32[i] = bv(0, i);
            a8s[i] = enc[av(2, i)];
            b8s[i] = enc[bv(2, i)];
            a8d[i] = enc[av(1, i)];
            b8d[i] = enc[bv(1, i)];
        }
        init_f16(a16, b16, core, cores);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    for (unsigned p = 0; p < 6; ++p) {
        const Cfg* k = &cfgs[p];

        // 1. Subsets of the compute cores (no barrier: each core only uses its
        // own slot).
        if (compute)
            for (unsigned m = 0; m < sizeof(masks) / sizeof(masks[0]); ++m) {
                const unsigned mask = masks[m];
                if (mask >> cores) continue;  // needs more cores than there are
                for (unsigned l = 0;
                     l < sizeof(subset_lengths) / sizeof(subset_lengths[0]);
                     ++l)
                    errors += run_subset(k, mask, subset_lengths[l], core,
                                         res_core + core * SLOT);
            }

        // 2. Many lengths through the multicore function (mask 0xff: all).
        for (unsigned l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
            const unsigned n = lengths[l];
            if (core == 0) fill(res, SLOT, SENTINEL);
            snrt_cluster_hw_barrier();
            if (!snrt_vec_dot_l1(k->a, k->b, res, scratch, n, k->prec, k->acc))
                ++errors;
            // The result is available to every core: each checks it.
            if (compute)
                errors += check_slot(k, 0xff, n, core, (volatile uint8_t*)res,
                                     0, n);
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
        printf("Vector dot product (L1) edge cases: %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
