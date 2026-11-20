// Cycle counts of the dot product on data in L1 (not a pass/fail test).
//
// Measures how the time grows with the vector length, to see the fixed cost of
// a call (function prologue and epilogue, SSR and FREP setup, fences, the
// reduction of the accumulators and lanes, and for the multicore version the
// two barriers and the combine) and the cost per word.
//
// It also compares the library function with the same computation written by
// hand as one single function, with the precision and the accumulation mode
// fixed in the code (no run-time choice, no helper functions, no partial-word
// path). The difference between the two is what the generic structure of the
// library costs: the run-time dispatch on the precision and the mode, the
// helpers (if the compiler does not inline them) and the generic function
// prologue. It is measured for every configuration.
//
// Lengths are given in 64-bit words, so the same row means the same amount of
// data in every precision (n = words * elements per word). A few rows add a
// partial word (the tail) of one element or of the largest partial size.
// For every configuration the table has the columns:
//   tail:   elements in the partial last word (0 if none);
//   single: cycles of snrt_vec_dot_core_l1 on one core (timed on core 0);
//   hand:   cycles of the hand-written single function (whole words only);
//   diff:   single minus hand: the cost of the library's generic structure;
//   extra:  single minus the words, that is, the cycles beyond one cycle per
//           word, which is the best that the two streams allow;
//   multi:  cycles of snrt_vec_dot_l1 as seen by core 0, with all the compute
//           cores working, including the barriers and the combine.
//
// The data is not initialized: the values do not change the timing.

#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

// Words of the longest vector. The two vectors take 2 * 4096 * 8 bytes = 64 kB
// of the 128 kB of L1, which is as much as fits: 16384 words would need 128 kB
// for each vector.
#define MAXW 4096

// A row: whole words, plus a partial word (the tail) of a kind: 0 = none,
// 1 = one element, 2 = the most elements that still leave the word partial
// (elements per word minus one). The tail goes through the zero-padded path of
// the library, which the hand-written functions do not have. Tails that do not
// exist for a precision (any tail for FP64, whose word is one element, and kind
// 2 for FP32, which is the same as kind 1) are skipped.
typedef struct {
    unsigned words;
    unsigned tail_kind;
} Row;

static const Row rows_list[] = {
    {0, 0},   {1, 0},   {2, 0},   {4, 0},    {8, 0},    {16, 0},   {32, 0},
    {64, 0},  {128, 0}, {256, 0}, {512, 0},  {1024, 0}, {4096, 0},
    {16, 1},  {16, 2},  {1024, 1}, {1024, 2},
};
#define NROWS (sizeof(rows_list) / sizeof(rows_list[0]))

// Number of elements of the tail of a row, 0 if the row is skipped for this
// number of elements per word. Sets *valid.
static unsigned tail_elems(unsigned kind, unsigned lanes, int* valid) {
    *valid = 1;
    if (kind == 0) return 0;
    if (kind == 1) {
        if (lanes < 2) *valid = 0;
        return 1;
    }
    if (lanes < 3) *valid = 0;
    return lanes - 1;
}

typedef struct {
    precision_t prec;
    snrt_acc_t acc;
    const char* name;
} Cfg;


//==============================================================================
// The same computation, written by hand as one single function per
// configuration. Whole words only; the instructions are fixed in the code.
// Same steps as the library: the streams, four accumulators in a hardware loop,
// the words that do not fill a group of four, the accumulators combined, the
// lanes summed, and one element stored.
//==============================================================================

// FP64: fused multiply-add.
__attribute__((noinline)) static void hand_fp64(const void* a, const void* b,
                                                void* res, size_t words) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    const size_t groups = words / 4, rem = words % 4;
    if (words > 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, words, 8);
        snrt_ssr_loop_1d(SNRT_SSR_DM1, words, 8);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)a);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)b);
        snrt_ssr_enable();
        if (groups > 0)
            asm volatile(
                "frep.o %[n], 4, 0, 0 \n"
                "fmadd.d %[a0], ft0, ft1, %[a0] \n"
                "fmadd.d %[a1], ft0, ft1, %[a1] \n"
                "fmadd.d %[a2], ft0, ft1, %[a2] \n"
                "fmadd.d %[a3], ft0, ft1, %[a3] \n"
                : [ a0 ] "+f"(a0), [ a1 ] "+f"(a1), [ a2 ] "+f"(a2),
                  [ a3 ] "+f"(a3)
                : [ n ] "r"(groups - 1)
                : "ft0", "ft1", "ft2", "memory");
        if (rem > 0)
            asm volatile(
                "frep.o %[n], 1, 0, 0 \n"
                "fmadd.d %[a0], ft0, ft1, %[a0] \n"
                : [ a0 ] "+f"(a0)
                : [ n ] "r"(rem - 1)
                : "ft0", "ft1", "ft2", "memory");
        snrt_fpu_fence();
        snrt_ssr_disable();
    }
    asm volatile(
        "fadd.d %[a0], %[a0], %[a1] \n"
        "fadd.d %[a2], %[a2], %[a3] \n"
        "fadd.d %[a0], %[a0], %[a2] \n"
        "fsd %[a0], 0(%[r]) \n"
        : [ a0 ] "+f"(a0), [ a2 ] "+f"(a2)
        : [ a1 ] "f"(a1), [ a3 ] "f"(a3), [ r ] "r"(res)
        : "memory");
    snrt_fpu_fence();
}

// One macro for the three packed configurations: the accumulate instruction,
// the instruction that adds two accumulators, the lane reduction and the store.
#define HAND_PACKED(NAME, MAC, ADD, LANES, STORE)                           \
    __attribute__((noinline)) static void NAME(const void* a, const void* b, \
                                               void* res, size_t words) {   \
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, d = 0.0, d2 = 0.0,   \
                d3 = 0.0;                                                   \
        const size_t groups = words / 4, rem = words % 4;                   \
        if (words > 0) {                                                    \
            snrt_ssr_loop_1d(SNRT_SSR_DM0, words, 8);                       \
            snrt_ssr_loop_1d(SNRT_SSR_DM1, words, 8);                       \
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)a);             \
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)b);             \
            snrt_ssr_enable();                                              \
            if (groups > 0)                                                 \
                asm volatile(                                               \
                    "frep.o %[n], 4, 0, 0 \n" MAC " %[a0], ft0, ft1 \n"     \
                    MAC " %[a1], ft0, ft1 \n" MAC " %[a2], ft0, ft1 \n"     \
                    MAC " %[a3], ft0, ft1 \n"                               \
                    : [ a0 ] "+f"(a0), [ a1 ] "+f"(a1), [ a2 ] "+f"(a2),   \
                      [ a3 ] "+f"(a3)                                      \
                    : [ n ] "r"(groups - 1)                                \
                    : "ft0", "ft1", "ft2", "memory");                      \
            if (rem > 0)                                                    \
                asm volatile("frep.o %[n], 1, 0, 0 \n" MAC                  \
                             " %[a0], ft0, ft1 \n"                          \
                             : [ a0 ] "+f"(a0)                             \
                             : [ n ] "r"(rem - 1)                          \
                             : "ft0", "ft1", "ft2", "memory");             \
            snrt_fpu_fence();                                               \
            snrt_ssr_disable();                                             \
        }                                                                   \
        asm volatile(ADD " %[a0], %[a0], %[a1] \n" ADD                      \
                     " %[a2], %[a2], %[a3] \n" ADD " %[a0], %[a0], %[a2] \n" \
                     LANES STORE                                           \
                     : [ a0 ] "+f"(a0), [ a2 ] "+f"(a2), [ d ] "+f"(d),    \
                       [ d2 ] "+f"(d2), [ d3 ] "+f"(d3)                     \
                     : [ a1 ] "f"(a1), [ a3 ] "f"(a3), [ r ] "r"(res)      \
                     : "memory");                                          \
        snrt_fpu_fence();                                                   \
    }

// FP32: vfmac.s, two lanes: one vfsum.s.
HAND_PACKED(hand_fp32, "vfmac.s", "vfadd.s",
            "vfsum.s %[d], %[a0] \n", "fsw %[d], 0(%[r]) \n")
// FP16 with wide accumulation: vfdotpex.s.h into FP32 lanes: one vfsum.s.
HAND_PACKED(hand_fp16_wide, "vfdotpex.s.h", "vfadd.s",
            "vfsum.s %[d], %[a0] \n", "fsw %[d], 0(%[r]) \n")
// FP16, accumulated in FP16: vfmac.h, four lanes: two vfsum.h steps.
HAND_PACKED(hand_fp16, "vfmac.h", "vfadd.h",
            "vfsum.h %[d], %[a0] \n vfsum.h %[d2], %[d] \n",
            "fsh %[d2], 0(%[r]) \n")
// FP8, accumulated in FP8: vfmac.b, eight lanes: three vfsum.b steps.
HAND_PACKED(hand_fp8, "vfmac.b", "vfadd.b",
            "vfsum.b %[d], %[a0] \n vfsum.b %[d2], %[d] \n"
            "vfsum.b %[d3], %[d2] \n",
            "fsb %[d3], 0(%[r]) \n")
// FP8 with wide accumulation: vfdotpex.h.b into FP16 lanes: vfsumex.s.h, then
// vfsum.s, stored as FP32.
HAND_PACKED(hand_fp8_wide, "vfdotpex.h.b", "vfadd.h",
            "vfsumex.s.h %[d], %[a0] \n vfsum.s %[d2], %[d] \n",
            "fsw %[d2], 0(%[r]) \n")

// Every configuration has a hand-written version (the index is the one of the
// table in main()).
static int has_hand(unsigned c) { return c < 6; }

// Cycles of one call of the hand-written function of configuration c.
static uint32_t time_hand(unsigned c, const void* a, const void* b, void* res,
                          size_t words) {
    uint32_t t0, t1;
    for (int pass = 0; pass < 2; ++pass) {  // the first pass is the warm-up
        t0 = snrt_mcycle();
        switch (c) {
            case 0:
                hand_fp64(a, b, res, words);
                break;
            case 1:
                hand_fp32(a, b, res, words);
                break;
            case 2:
                hand_fp16(a, b, res, words);
                break;
            case 3:
                hand_fp16_wide(a, b, res, words);
                break;
            case 4:
                hand_fp8(a, b, res, words);
                break;
            default:
                hand_fp8_wide(a, b, res, words);
                break;
        }
        t1 = snrt_mcycle();
    }
    return t1 - t0;
}

// Cycles of one call of the single-core function (after a warm-up call that
// fills the instruction cache).
static uint32_t time_single(const void* a, const void* b, void* res, size_t n,
                            precision_t prec, snrt_acc_t acc) {
    snrt_vec_dot_core_l1(a, b, res, n, prec, acc);
    const uint32_t t0 = snrt_mcycle();
    snrt_vec_dot_core_l1(a, b, res, n, prec, acc);
    const uint32_t t1 = snrt_mcycle();
    return t1 - t0;
}

// Cycles of one call of the multicore function, seen by core 0. Every core
// calls it, and they start together.
static uint32_t time_multi(const void* a, const void* b, void* res,
                           void* scratch, size_t n, precision_t prec,
                           snrt_acc_t acc) {
    snrt_vec_dot_l1(a, b, res, scratch, n, prec, acc);  // warm-up
    snrt_cluster_hw_barrier();
    const uint32_t t0 = snrt_mcycle();
    snrt_vec_dot_l1(a, b, res, scratch, n, prec, acc);
    const uint32_t t1 = snrt_mcycle();
    return t1 - t0;
}

int main() {
    // Every core must make the same allocation calls in the same order.
    void* a = snrt_l1_alloc_cluster_local(MAXW * sizeof(uint64_t) + 8, 8);
    void* b = snrt_l1_alloc_cluster_local(MAXW * sizeof(uint64_t) + 8, 8);
    void* res = snrt_l1_alloc_cluster_local(16, 8);
    void* scratch =
        snrt_l1_alloc_cluster_local(snrt_vec_reduce_scratch_bytes(), 8);

    const Cfg cfgs[6] = {
        {FP64, SNRT_ACC_SAME, "FP64"},
        {FP32, SNRT_ACC_SAME, "FP32"},
        {FP16, SNRT_ACC_SAME, "FP16"},
        {FP16, SNRT_ACC_WIDE, "FP16 wide"},
        {FP8, SNRT_ACC_SAME, "FP8"},
        {FP8, SNRT_ACC_WIDE, "FP8 wide"},
    };

    const int core0 = snrt_cluster_core_idx() == 0;

    for (unsigned c = 0; c < 6; ++c) {
        const Cfg* k = &cfgs[c];
        const unsigned lanes = (unsigned)snrt_vec_lanes(k->prec);
        uint32_t single[NROWS];
        uint32_t hand[NROWS];
        uint32_t multi[NROWS];

        for (unsigned r = 0; r < NROWS; ++r) {
            int valid;
            const unsigned tail = tail_elems(rows_list[r].tail_kind, lanes,
                                             &valid);
            if (!valid) continue;  // the same on every core
            const size_t n = (size_t)rows_list[r].words * lanes + tail;

            // Single core: core 0 only; the others wait.
            snrt_cluster_hw_barrier();
            if (core0) {
                single[r] = time_single(a, b, res, n, k->prec, k->acc);
                hand[r] = (has_hand(c) && tail == 0)
                              ? time_hand(c, a, b, res, rows_list[r].words)
                              : 0;
            }
            snrt_cluster_hw_barrier();

            // Multicore: every core calls it.
            multi[r] = time_multi(a, b, res, scratch, n, k->prec, k->acc);
        }

        if (core0) {
            printf("\n%s\n", k->name);
            printf("  words  tail  single   hand   diff  extra   multi\n");
            for (unsigned r = 0; r < NROWS; ++r) {
                int valid;
                const unsigned tail = tail_elems(rows_list[r].tail_kind, lanes,
                                                 &valid);
                if (!valid) continue;
                printf("  %5u  %4u  %6u", rows_list[r].words, tail,
                       (unsigned)single[r]);
                if (has_hand(c) && tail == 0)
                    printf("  %5u  %5d", (unsigned)hand[r],
                           (int)single[r] - (int)hand[r]);
                else
                    printf("      -      -");
                printf("  %5d  %6u\n",
                       (int)single[r] - (int)rows_list[r].words,
                       (unsigned)multi[r]);
            }
        }
        snrt_cluster_hw_barrier();
    }
    return 0;
}
