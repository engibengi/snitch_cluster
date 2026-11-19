#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

// 75 elements. With 8 compute cores, every core gets at least one whole word
// in every precision except FP8 (FP8: 9 words, FP16: 18, FP32: 37, FP64: 75),
// the split is uneven (interleaved: core c takes the words c, c + 8, ...), and
// the core whose turn it is takes the partial word (3 elements for FP8 and
// FP16, 1 for FP32), so the tail is exercised too. FP32 and FP64
// cores use both the group-of-four loop and the one-word loop.
#define N 75
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
// Reference dot product of elements [0, n).
static int dot_ref(int kind, unsigned n) {
    int s = 0;
    for (unsigned i = 0; i < n; ++i) s += av(kind, i) * bv(kind, i);
    return s;
}

// Half-precision arithmetic lives in its own functions, called by compute cores
// only. The DMA core (core 8) has no FP16 support: any FP16 instruction
// executed there is an illegal instruction, and the core restarts.
__attribute__((noinline)) static void init_f16(_Float16* a, _Float16* b,
                                               unsigned first, unsigned step) {
    for (unsigned i = first; i < N; i += step) {
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

// One configuration: data precision, accumulation mode and data.
typedef struct {
    precision_t prec;
    snrt_acc_t acc;
    int kind;
    const char* name;
    const void* a;
    const void* b;
} Cfg;

// Check a result slot: the element, then sentinel bytes.
static unsigned check_slot(const char* what, const Cfg* k, unsigned core,
                           const volatile uint8_t* got, unsigned n) {
    const precision_t rp = snrt_vec_dot_result_precision(k->prec, k->acc);
    const unsigned es = (unsigned)rp;
    const uint64_t want = encode(rp, dot_ref(k->kind, n));
    unsigned errors = 0;
    for (unsigned b = 0; b < SLOT; ++b) {
        const uint8_t w = b < es ? (uint8_t)(want >> (8 * b)) : SENTINEL;
        if (got[b] != w) {
            ++errors;
            printf("core %u: %s %s byte %u: expected 0x%02x, got 0x%02x\n",
                   core, what, k->name, b, (unsigned)w, (unsigned)got[b]);
        }
    }
    return errors;
}

int main() {
    double*   a64 = ALLOC(double, N);
    double*   b64 = ALLOC(double, N);
    float*    a32 = ALLOC(float, N);
    float*    b32 = ALLOC(float, N);
    _Float16* a16 = ALLOC(_Float16, N);
    _Float16* b16 = ALLOC(_Float16, N);
    uint8_t*  a8s = ALLOC(uint8_t, N + 8);  // kind 2 (FP8, accumulate in FP8)
    uint8_t*  b8s = ALLOC(uint8_t, N + 8);
    uint8_t*  a8d = ALLOC(uint8_t, N + 8);  // kind 1 (FP8, wide accumulation)
    uint8_t*  b8d = ALLOC(uint8_t, N + 8);
    // Partial results of the cores, for the multicore function.
    void* scratch = snrt_l1_alloc_cluster_local(
        snrt_vec_reduce_scratch_bytes(), 8);
    // One result slot per configuration (multicore), and one per core and
    // configuration (single-core).
    uint8_t* res = ALLOC(uint8_t, 6 * SLOT);
    uint8_t* res_unsupported = ALLOC(uint8_t, SLOT);
    uint8_t* res_core =
        ALLOC(uint8_t, snrt_cluster_compute_core_num() * 6 * SLOT);
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

    // Every compute core prepares its share of the inputs (elements core,
    // core + cores, ...). The DMA core does not compute, it only takes part in
    // the barriers.
    if (compute) {
        for (unsigned i = core; i < N; i += cores) {
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
    // Core 0 sets the sentinels in the result slots.
    if (core == 0) {
        fill(res, 6 * SLOT, SENTINEL);
        fill(res_unsupported, SLOT, SENTINEL);
    }
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();

    // Every core calls the multicore version (it contains barriers); the work
    // is split among the compute cores.
    for (unsigned c = 0; c < 6; ++c)
        if (!snrt_vec_dot_l1(cfgs[c].a, cfgs[c].b, res + c * SLOT, scratch, N,
                             cfgs[c].prec, cfgs[c].acc))
            ++errors;

    // Wide accumulation does not exist for FP32 and FP64: false, and nothing
    // is written.
    if (snrt_vec_dot_l1(a64, b64, res_unsupported, scratch, N, FP64,
                        SNRT_ACC_WIDE))
        ++errors;
    if (snrt_vec_dot_l1(a32, b32, res_unsupported, scratch, N, FP32,
                        SNRT_ACC_WIDE))
        ++errors;

    // The result is available to every core: each compute core checks all of
    // them.
    if (compute) {
        for (unsigned c = 0; c < 6; ++c)
            errors += check_slot("vec_dot_l1", &cfgs[c], core,
                                 (volatile uint8_t*)res + c * SLOT, N);
        for (unsigned b = 0; b < SLOT; ++b)
            if (((volatile uint8_t*)res_unsupported)[b] != SENTINEL) {
                ++errors;
                printf("core %u: unsupported call wrote byte %u\n", core, b);
            }
    }

    // Single-core version: every compute core runs it on its own, on the whole
    // vectors, into its own result slot.
    if (compute) {
        for (unsigned c = 0; c < 6; ++c) {
            uint8_t* mine = res_core + (core * 6 + c) * SLOT;
            fill(mine, SLOT, SENTINEL);
            if (!snrt_vec_dot_core_l1(cfgs[c].a, cfgs[c].b, mine, N,
                                      cfgs[c].prec, cfgs[c].acc))
                ++errors;
            errors += check_slot("vec_dot_core_l1", &cfgs[c], core,
                                 (volatile uint8_t*)mine, N);
        }
    }

    // Core 0 collects the error counts and reports.
    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();

    if (core == 0) {
        unsigned total = 0;
        for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
            total += ((volatile unsigned*)core_errors)[c];
        printf("Vector dot product (L1): %s (%u errors)\n",
               total ? "FAIL" : "PASS", total);
        return total ? 1 : 0;
    }
    return 0;
}
