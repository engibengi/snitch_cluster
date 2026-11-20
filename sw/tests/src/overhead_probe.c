// Probes of the fixed costs of the vector library (not a pass/fail test).
//
// vector_dot_latency_l1 showed that a call of the library dot product costs
// 70-95 cycles more than the same computation written as one fixed function,
// that the multicore version costs 250-300 cycles more than the single-core one
// at zero length, that a partial word costs about 70 + 12 cycles per element,
// and that eight cores stream at 71-93% of the ideal rate. This file measures
// the possible causes one by one, so that the fixes can be chosen by impact.
//
// Everything is timed with the core cycle counter. The sections:
//   1. Primitives: the cost of one execution of the small things the library
//      does (a call, saving registers, a fence, an integer divide, ...).
//   2. Ablation: a fixed-function dot product (FP32) and then the same function
//      with one feature of the library added at a time, to see what each adds.
//   3. The multicore call, broken down: the barrier, the split, the combine,
//      and how much of the measured total is left unexplained.
//   4. The partial word: memcpy of a few bytes, against straight-line copies.
//   5. Eight cores streaming at once, with different slice lengths and offsets,
//      to look for L1 bank conflicts.
//
// The data is not initialized: the values do not change the timing.
// Only compute cores run FP instructions; the DMA core only takes part in the
// barriers.

#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

//==============================================================================
// Hand-written dot products: the instructions are fixed in the code, whole
// words only, same steps as the library.
//==============================================================================

#define HAND_PACKED(NAME, MAC, ADD, LANES, STORE)                           \
    static inline __attribute__((always_inline)) void NAME##_body(          \
        const void* a, const void* b, void* res, size_t words) {            \
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, d = 0.0, d2 = 0.0,   \
               d3 = 0.0;                                                    \
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
    }                                                                       \
    __attribute__((noinline)) static void NAME(const void* a, const void* b, \
                                               void* res, size_t words) {   \
        NAME##_body(a, b, res, words);                                      \
    }

HAND_PACKED(hand_fp32, "vfmac.s", "vfadd.s", "vfsum.s %[d], %[a0] \n",
            "fsw %[d], 0(%[r]) \n")
HAND_PACKED(hand_fp16, "vfmac.h", "vfadd.h",
            "vfsum.h %[d], %[a0] \n vfsum.h %[d2], %[d] \n",
            "fsh %[d2], 0(%[r]) \n")
HAND_PACKED(hand_fp16_wide, "vfdotpex.s.h", "vfadd.s",
            "vfsum.s %[d], %[a0] \n", "fsw %[d], 0(%[r]) \n")
HAND_PACKED(hand_fp8, "vfmac.b", "vfadd.b",
            "vfsum.b %[d], %[a0] \n vfsum.b %[d2], %[d] \n"
            "vfsum.b %[d3], %[d2] \n",
            "fsb %[d3], 0(%[r]) \n")
HAND_PACKED(hand_fp8_wide, "vfdotpex.h.b", "vfadd.h",
            "vfsumex.s.h %[d], %[a0] \n vfsum.s %[d2], %[d] \n",
            "fsw %[d2], 0(%[r]) \n")

static inline __attribute__((always_inline)) void hand_fp64_body(
    const void* a, const void* b, void* res, size_t words) {
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
__attribute__((noinline)) static void hand_fp64(const void* a, const void* b,
                                                void* res, size_t words) {
    hand_fp64_body(a, b, res, words);
}

// Ablation 1: the fixed FP32 function, preceded by the run-time computation of
// the words from the length: lanes = 8 / precision and words = n / lanes (two
// integer divisions, as in the library).
__attribute__((noinline)) static void abl_div(const void* a, const void* b,
                                              void* res, size_t n,
                                              precision_t prec) {
    const size_t lanes = snrt_vec_lanes(prec);
    const size_t words = n / lanes;
    hand_fp32_body(a, b, res, words);
}

// Ablation 2: the same, and the run-time choice of the configuration among the
// six (the dispatch of the library), still without the partial-word path.
__attribute__((noinline)) static void abl_dispatch(const void* a, const void* b,
                                                   void* res, size_t n,
                                                   precision_t prec,
                                                   snrt_acc_t acc) {
    if (!snrt_vec_dot_supported(prec, acc)) return;
    const size_t words = n >> snrt_vec_lanes_log2(prec);  // no division
    if (prec == FP64)
        hand_fp64_body(a, b, res, words);
    else if (acc == SNRT_ACC_WIDE && prec == FP16)
        hand_fp16_wide_body(a, b, res, words);
    else if (acc == SNRT_ACC_WIDE && prec == FP8)
        hand_fp8_wide_body(a, b, res, words);
    else if (prec == FP32)
        hand_fp32_body(a, b, res, words);
    else if (prec == FP16)
        hand_fp16_body(a, b, res, words);
    else
        hand_fp8_body(a, b, res, words);
}

//==============================================================================
// Measuring helpers
//==============================================================================

#define REPS 32

// Cycles x 100 of one repetition of `body`, from REPS repetitions after a
// warm-up one, minus `base` (the cost of the empty loop).
#define BENCH(var, body)                                                    \
    do {                                                                    \
        {body;}                                                             \
        const uint32_t t0_ = snrt_mcycle();                                 \
        for (int i_ = 0; i_ < REPS; ++i_) {body;}                           \
        const uint32_t t1_ = snrt_mcycle();                                 \
        (var) = (int)(((t1_ - t0_) * 100u) / REPS);                         \
    } while (0)

// Prints a value in hundredths as a decimal number, with its sign.
static void print_x100(int v) {
    printf("%s%d.%02d", v < 0 ? "-" : "", (v < 0 ? -v : v) / 100,
           (v < 0 ? -v : v) % 100);
}

static void show(const char* name, int x100, int base100) {
    printf("  %-48s ", name);
    print_x100(x100 - base100);
    printf("\n");
}

// Keeping the compiler from folding or removing the measured code, without
// touching memory. Everything that a timed region uses is in L1 (the vectors,
// the results, the scratch memory, the stack) or in a register: the values that
// come from tables or global variables (which are linked into L3, far slower
// than L1) are copied into local variables before the timed region starts. The
// run-time values are made opaque in a register, and the results are consumed
// by an empty asm statement.
#define OPAQUE(x) asm volatile("" : "+r"(x))
#define USE(x) asm volatile("" : : "r"(x))

__attribute__((noinline)) static void f_empty() {
    asm volatile("" ::: "memory");
}
__attribute__((noinline)) static void f_save_fp() {
    asm volatile("" ::: "fs0", "fs1", "fs2", "fs3", "memory");
}
__attribute__((noinline)) static void f_save_int() {
    asm volatile("" ::: "s1", "s2", "s3", "s4", "s5", "s6", "s7", "memory");
}

// The library calls through functions that cannot be inlined into the
// measuring loop. The entry points are force-inlined, so called directly from a
// loop of identical calls the compiler could hoist work out of the loop, and
// the measurement would be better than a real isolated call. Behind a wrapper,
// the body is the one that a call site gets, and the arguments are run-time
// values.
__attribute__((noinline)) static void w_lib(const void* a, const void* b,
                                            void* res, size_t n, unsigned prec,
                                            unsigned acc) {
    snrt_vec_dot_core_l1(a, b, res, n, (precision_t)prec, (snrt_acc_t)acc);
}
#define W_CONST(NAME, PREC, ACC)                                            \
    __attribute__((noinline)) static void NAME(const void* a, const void* b, \
                                               void* res, size_t n) {       \
        snrt_vec_dot_core_l1(a, b, res, n, PREC, ACC);                      \
    }
W_CONST(w_const_fp64, FP64, SNRT_ACC_SAME)
W_CONST(w_const_fp32, FP32, SNRT_ACC_SAME)
W_CONST(w_const_fp16, FP16, SNRT_ACC_SAME)
W_CONST(w_const_fp16_wide, FP16, SNRT_ACC_WIDE)
W_CONST(w_const_fp8, FP8, SNRT_ACC_SAME)
W_CONST(w_const_fp8_wide, FP8, SNRT_ACC_WIDE)
__attribute__((noinline)) static void w_multi_fp32(const void* a, const void* b,
                                                   void* res, void* scratch,
                                                   size_t n) {
    snrt_vec_dot_l1(a, b, res, scratch, n, FP32, SNRT_ACC_SAME);
}

// Wrappers that the compiler cannot inline, so that every repetition of a
// measurement really executes the library helper (the helpers are pure, and
// with constant arguments the compiler would run them once, outside the loop).
// Their call cost is the "call of an empty function" line and is subtracted.
__attribute__((noinline)) static unsigned w_lanes(unsigned p) {
    return (unsigned)snrt_vec_lanes((precision_t)p);
}
// The number of compute cores is a constant in the library (8), so the
// divisions of the split become shifts.
__attribute__((noinline)) static unsigned w_split(unsigned words,
                                                  unsigned core) {
    size_t first, count;
    snrt_vec_split(words, core, snrt_cluster_compute_core_num(), &first,
                   &count);
    return (unsigned)(first + count);
}
__attribute__((noinline)) static unsigned w_ids(unsigned x) {
    return x + snrt_cluster_core_idx() + snrt_is_compute_core() +
           snrt_cluster_compute_core_num();
}

// The same, with the library's own helpers (no function calls).
__attribute__((noinline)) static unsigned w_ids_vec(unsigned x) {
    return x + (unsigned)snrt_vec_core_idx() + snrt_vec_is_compute_core() +
           SNRT_VEC_COMPUTE_CORES;
}

// Copies of 1 to 7 bytes for the partial word, three ways. `src` and `dst` are
// eight-byte aligned.
__attribute__((noinline)) static void copy_memcpy(void* dst, const void* src,
                                                  size_t bytes) {
    memcpy(dst, src, bytes);
}
__attribute__((noinline)) static void copy_straight(void* dst, const void* src,
                                                    size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (bytes & 4) {
        *(volatile uint32_t*)d = *(const volatile uint32_t*)s;
        d += 4;
        s += 4;
    }
    if (bytes & 2) {
        *(volatile uint16_t*)d = *(const volatile uint16_t*)s;
        d += 2;
        s += 2;
    }
    if (bytes & 1) *(volatile uint8_t*)d = *(const volatile uint8_t*)s;
}
__attribute__((noinline)) static void copy_loop(void* dst, const void* src,
                                                size_t bytes) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const volatile uint8_t* s = (const volatile uint8_t*)src;
    for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
}

typedef struct {
    precision_t prec;
    snrt_acc_t acc;
    const char* name;
} Cfg;

// Words of the buffers. Section 5 needs 8 cores times 513 words, plus offsets.
#define BUFW 4224

int main() {
    // Every core must make the same allocation calls in the same order.
    void* a = snrt_l1_alloc_cluster_local(BUFW * sizeof(uint64_t), 8);
    void* b = snrt_l1_alloc_cluster_local(BUFW * sizeof(uint64_t), 8);
    void* res = snrt_l1_alloc_cluster_local(16, 8);
    void* scratch =
        snrt_l1_alloc_cluster_local(snrt_vec_reduce_scratch_bytes(), 8);
    uint32_t* cycles = (uint32_t*)snrt_l1_alloc_cluster_local(
        snrt_cluster_core_num() * sizeof(uint32_t), 8);
    uint32_t* phase = (uint32_t*)snrt_l1_alloc_cluster_local(
        snrt_cluster_core_num() * 5 * sizeof(uint32_t), 8);
    uint8_t* tmp_in = (uint8_t*)snrt_l1_alloc_cluster_local(16, 8);
    uint8_t* tmp_out = (uint8_t*)snrt_l1_alloc_cluster_local(16, 8);

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int core0 = core == 0;
    const int compute = snrt_is_compute_core();

    const Cfg cfgs[6] = {
        {FP64, SNRT_ACC_SAME, "FP64"},
        {FP32, SNRT_ACC_SAME, "FP32"},
        {FP16, SNRT_ACC_SAME, "FP16"},
        {FP16, SNRT_ACC_WIDE, "FP16 wide"},
        {FP8, SNRT_ACC_SAME, "FP8"},
        {FP8, SNRT_ACC_WIDE, "FP8 wide"},
    };

    // Values measured in section 1 and reused in section 3.
    int base = 0, t_barrier = 0, t_split = 0, t_ids = 0, t_combine = 0;

    //--------------------------------------------------------------------------
    // 1. Primitives (core 0 only; the others wait)
    //--------------------------------------------------------------------------
    if (core0) {
        printf("1. Primitives: cycles per execution (the empty loop is "
               "subtracted)\n");
        int x;
        BENCH(base, asm volatile("" ::: "memory"));
        show("empty loop (per iteration, not subtracted)", base, 0);

        BENCH(x, f_empty());
        show("call of an empty function", x, base);
        BENCH(x, f_save_fp());
        show("call, saving 4 FP registers (fs0-fs3)", x, base);
        BENCH(x, f_save_int());
        show("call, saving 7 integer registers (s1-s7)", x, base);

        BENCH(x, snrt_fpu_fence());
        show("snrt_fpu_fence, FPU idle", x, base);
        BENCH(x, {
            asm volatile("vfadd.s ft3, ft3, ft3" ::: "ft3");
            snrt_fpu_fence();
        });
        show("one vfadd.s, then snrt_fpu_fence", x, base);

        {
            unsigned r = 0, d = 4, big = 0x7fff0000u;
            OPAQUE(big);
            BENCH(x, {
                asm volatile("divu %[r], %[a], %[b]"
                             : [ r ] "=r"(r)
                             : [ a ] "r"(75u), [ b ] "r"(d));
                d = (r & 3) + 2;  // the next divisor depends on the quotient
            });
            USE(r);
            show("divu, dependent chain (75 / small)", x, base);
            BENCH(x, {
                asm volatile("divu %[r], %[a], %[b]"
                             : [ r ] "=r"(r)
                             : [ a ] "r"(big), [ b ] "r"(d));
                d = (r & 3) + 2;
            });
            USE(r);
            show("divu, dependent chain (large / small)", x, base);
            BENCH(x, {
                asm volatile("mul %[r], %[a], %[b]"
                             : [ r ] "=r"(r)
                             : [ a ] "r"(r | 3u), [ b ] "r"(d));
            });
            USE(r);
            show("mul, dependent chain", x, base);
        }
        int call_cost;
        BENCH(call_cost, f_empty());
        unsigned prec_rt = FP16, words_rt = 75, core_rt = 3;
        OPAQUE(prec_rt);
        OPAQUE(words_rt);
        OPAQUE(core_rt);
        BENCH(x, {
            unsigned p_ = prec_rt;
            OPAQUE(p_);  // a new value each time: the call cannot be hoisted
            USE(w_lanes(p_));
        });
        show("snrt_vec_lanes (8 / precision), call subtracted", x, call_cost);
        BENCH(x, {
            unsigned w_ = words_rt;
            unsigned c_ = core_rt;
            OPAQUE(w_);
            OPAQUE(c_);
            USE(w_split(w_, c_));
        });
        t_split = x - call_cost;
        show("snrt_vec_split (75 words, core 3, 8 cores)", x, call_cost);
        BENCH(x, {
            unsigned c_ = core_rt;
            OPAQUE(c_);
            USE(w_ids(c_));
        });
        show("runtime calls: core_idx, is_compute, compute_num", x, call_cost);
        BENCH(x, {
            unsigned c_ = core_rt;
            OPAQUE(c_);
            USE(w_ids_vec(c_));
        });
        t_ids = x - call_cost;
        show("library helpers (no calls), same three values", x, call_cost);
        {
            // The final combine of the single-core function (FP32).
            double a0 = 1, a1 = 1, a2 = 1, a3 = 1, d = 0;
            BENCH(x, {
                asm volatile(
                    "vfadd.s %[a0], %[a0], %[a1] \n"
                    "vfadd.s %[a2], %[a2], %[a3] \n"
                    "vfadd.s %[a0], %[a0], %[a2] \n"
                    "vfsum.s %[d], %[a0] \n"
                    "fsw %[d], 0(%[r]) \n"
                    : [ a0 ] "+f"(a0), [ a2 ] "+f"(a2), [ d ] "+f"(d)
                    : [ a1 ] "f"(a1), [ a3 ] "f"(a3), [ r ] "r"(res)
                    : "memory");
                snrt_fpu_fence();
            });
            show("combine 4 accumulators + lanes + store + fence", x, base);
        }
        {
            // The combine of the multicore function on core 0 (8 partials).
            BENCH(x, snrt_dot_combine8((const uint64_t*)scratch, FP32, res));
            t_combine = x - base;
            show("multicore combine (8 partials, core 0)", x, base);
        }
        printf("\n");
    }
    snrt_cluster_hw_barrier();

    // The cost of a barrier: every core calls it in step (core 0 times it).
    {
        const int reps = 16;
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        uint32_t t0 = snrt_mcycle();
        for (int i = 0; i < reps; ++i) snrt_cluster_hw_barrier();
        uint32_t t1 = snrt_mcycle();
        if (core0) {
            t_barrier = (int)(((t1 - t0) * 100u) / reps);
            printf("   barrier, all 9 cores, back to back        %5d.%02d\n\n",
                   t_barrier / 100, t_barrier % 100);
        }
    }

    //--------------------------------------------------------------------------
    // 2. Ablation: fixed function, then one library feature added at a time
    //--------------------------------------------------------------------------
    {
        static const unsigned words_list[] = {0, 16, 256};
        if (core0) {
            printf("2. Ablation (cycles, one call, whole words only)\n");
            printf("   library  = snrt_vec_dot_core_l1\n");
            printf("   hand     = fixed function\n");
            printf("   +div     = hand, with lanes = 8 / precision and words = "
                   "n / lanes at run time (FP32 only)\n");
            printf("   +switch  = hand, with the words taken by a shift (no "
                   "division) and the run-time choice among the six "
                   "configurations (so +div is not part of it)\n");
            printf("   (library - +switch is what is left: the partial-word "
                   "path, the supported check and the call of the "
                   "dispatcher)\n");
            printf("   const    = snrt_vec_dot_core_l1 called with the "
                   "precision and the mode as constants,\n"
                   "              so that the choices on them are removed at "
                   "compile time (the length is still a run-time value)\n");
        }
        for (unsigned c = 0; c < 6; ++c) {
            const Cfg* k = &cfgs[c];
            const unsigned lanes = (unsigned)snrt_vec_lanes(k->prec);
            if (core0) printf("\n   %s\n   words  library   const    hand"
                              "    +div  +switch\n", k->name);
            for (unsigned r = 0; r < 3; ++r) {
                const unsigned words = words_list[r];
                const size_t n = (size_t)words * lanes;
                snrt_cluster_hw_barrier();
                if (core0) {
                    int lib, cst, hand, div = -1, sw;
                    unsigned prec_v = (unsigned)k->prec;
                    unsigned acc_v = (unsigned)k->acc;
                    OPAQUE(prec_v);
                    OPAQUE(acc_v);
                    BENCH(lib, w_lib(a, b, res, n, prec_v, acc_v));
                    switch (c) {
                        case 0: BENCH(cst, w_const_fp64(a, b, res, n)); break;
                        case 1: BENCH(cst, w_const_fp32(a, b, res, n)); break;
                        case 2: BENCH(cst, w_const_fp16(a, b, res, n)); break;
                        case 3:
                            BENCH(cst, w_const_fp16_wide(a, b, res, n));
                            break;
                        case 4: BENCH(cst, w_const_fp8(a, b, res, n)); break;
                        default:
                            BENCH(cst, w_const_fp8_wide(a, b, res, n));
                            break;
                    }
                    switch (c) {
                        case 0: BENCH(hand, hand_fp64(a, b, res, words)); break;
                        case 1: BENCH(hand, hand_fp32(a, b, res, words)); break;
                        case 2: BENCH(hand, hand_fp16(a, b, res, words)); break;
                        case 3:
                            BENCH(hand, hand_fp16_wide(a, b, res, words));
                            break;
                        case 4: BENCH(hand, hand_fp8(a, b, res, words)); break;
                        default:
                            BENCH(hand, hand_fp8_wide(a, b, res, words));
                            break;
                    }
                    if (c == 1)
                        BENCH(div,
                              abl_div(a, b, res, n, (precision_t)prec_v));
                    BENCH(sw, abl_dispatch(a, b, res, n, (precision_t)prec_v,
                                           (snrt_acc_t)acc_v));
                    printf("   %5u  %6d  %6d  %6d  ", words, lib / 100,
                           cst / 100, hand / 100);
                    if (c == 1)
                        printf("%6d  ", div / 100);
                    else
                        printf("     -  ");
                    printf("%6d\n", sw / 100);
                }
                snrt_cluster_hw_barrier();
            }
        }
        if (core0) printf("\n");
    }
    snrt_cluster_hw_barrier();

    //--------------------------------------------------------------------------
    // 3. The multicore call, broken down (FP32, zero words)
    //--------------------------------------------------------------------------
    {
        int lib0 = 0;
        if (core0) {
            int x;
            BENCH(x, w_const_fp32(a, b, res, 0));
            lib0 = x;
        }
        snrt_cluster_hw_barrier();
        // Warm-up, then time the multicore call with all cores in step. The
        // length is made opaque: w_multi_fp32 is only called with this one
        // value, and the compiler would otherwise propagate the constant into
        // it and remove the work (which an earlier version of this section did,
        // measuring about half of the real cost).
        size_t zero_rt = 0;
        OPAQUE(zero_rt);
        w_multi_fp32(a, b, res, scratch, zero_rt);
        uint32_t sum = 0, lo = 0xffffffffu, hi = 0;
        for (int i = 0; i < 8; ++i) {
            snrt_cluster_hw_barrier();
            const uint32_t t0 = snrt_mcycle();
            w_multi_fp32(a, b, res, scratch, zero_rt);
            const uint32_t t1 = snrt_mcycle();
            sum += t1 - t0;
            if (t1 - t0 < lo) lo = t1 - t0;
            if (t1 - t0 > hi) hi = t1 - t0;
        }
        if (core0) {
            const int measured = (int)(sum * 100u / 8);
            const int predicted =
                t_ids + t_split + lib0 + 2 * t_barrier + t_combine;
            printf("3. Multicore call, FP32, zero words (cycles)\n");
            printf("   measured (snrt_vec_dot_l1, mean of 8)   %6d.%02d  "
                   "(min %u, max %u)\n",
                   measured / 100, measured % 100, (unsigned)lo, (unsigned)hi);
            printf("   core ids and counts                     %6d.%02d\n",
                   t_ids / 100, t_ids % 100);
            printf("   snrt_vec_split                          %6d.%02d\n",
                   t_split / 100, t_split % 100);
            printf("   single-core function, zero words        %6d.%02d\n",
                   lib0 / 100, lib0 % 100);
            printf("   2 barriers                              %6d.%02d\n",
                   2 * t_barrier / 100, 2 * t_barrier % 100);
            printf("   combine of 8 partials                   %6d.%02d\n",
                   t_combine / 100, t_combine % 100);
            printf("   sum of the parts                        %6d.%02d\n",
                   predicted / 100, predicted % 100);
            printf("   unexplained (measured - parts)          ");
            print_x100(measured - predicted);
            printf("\n\n");
        }
    }
    snrt_cluster_hw_barrier();

    //--------------------------------------------------------------------------
    // 3b. The phases of the multicore call, per core (FP32, zero words)
    //     The steps of snrt_vec_dot_l1, written out, with a timestamp after
    //     each:
    //     (1) the single-core call on the core's own slice; (2) the wait at the
    //     first barrier; (3) the combine of the partials, on core 0; (4) the
    //     wait at the second barrier. Run with constants (the code of one
    //     configuration) and with a run-time precision (the dispatcher), to
    //     see where the cost of a multicore call is, and whether it is the
    //     same when eight cores run the same code at once.
    //--------------------------------------------------------------------------
    {
        if (core0) {
            printf("3b. Phases of the multicore call, FP32, zero words "
                   "(cycles, mean of 8 calls)\n");
            printf("                              core 0   slowest   "
                   "fastest\n");
        }
        for (int variant = 0; variant < 2; ++variant) {
            unsigned prec_v = FP32, acc_v = SNRT_ACC_SAME;
            OPAQUE(prec_v);
            OPAQUE(acc_v);
            uint32_t sum0[5] = {0}, sumhi[5] = {0}, sumlo[5] = {0};
            for (int rep = 0; rep < 8; ++rep) {
                snrt_cluster_hw_barrier();
                uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
                if (compute) {
                    uint64_t* slot = (uint64_t*)scratch + core;
                    t0 = snrt_mcycle();
                    ((volatile uint32_t*)slot)[0] = 0;
                    ((volatile uint32_t*)slot)[1] = 0;
                    if (variant == 0)
                        w_const_fp32(a, b, slot, 0);
                    else
                        w_lib(a, b, slot, 0, prec_v, acc_v);
                    t1 = snrt_mcycle();
                }
                snrt_cluster_hw_barrier();
                if (compute) {
                    t2 = snrt_mcycle();
                    if (core0)
                        snrt_dot_combine8((const uint64_t*)scratch, FP32, res);
                    t3 = snrt_mcycle();
                }
                snrt_cluster_hw_barrier();
                if (compute) {
                    t4 = snrt_mcycle();
                    uint32_t* ph = phase + core * 5;
                    ph[0] = t1 - t0;
                    ph[1] = t2 - t1;
                    ph[2] = t3 - t2;
                    ph[3] = t4 - t3;
                    ph[4] = t4 - t0;
                }
                snrt_cluster_hw_barrier();
                if (core0) {
                    for (int k = 0; k < 5; ++k) {
                        uint32_t hi = 0, lo = 0xffffffffu;
                        for (unsigned c = 0; c < cores; ++c) {
                            const uint32_t v =
                                ((volatile uint32_t*)phase)[c * 5 + k];
                            if (v > hi) hi = v;
                            if (v < lo) lo = v;
                        }
                        sum0[k] += ((volatile uint32_t*)phase)[k];
                        sumhi[k] += hi;
                        sumlo[k] += lo;
                    }
                }
            }
            if (core0) {
                static const char* names[5] = {
                    "single-core call (own slice)", "wait, first barrier",
                    "combine (core 0)", "wait, second barrier", "total"};
                printf("   %s\n", variant == 0
                                       ? "constants (code of one configuration)"
                                       : "run-time precision (dispatcher)");
                for (int k = 0; k < 5; ++k)
                    printf("     %-26s %6u  %8u  %8u\n", names[k],
                           (unsigned)(sum0[k] / 8), (unsigned)(sumhi[k] / 8),
                           (unsigned)(sumlo[k] / 8));
            }
        }
        if (core0) printf("\n");
    }
    snrt_cluster_hw_barrier();

    //--------------------------------------------------------------------------
    // 4. The partial word: copying 1 to 7 bytes (core 0 only)
    //--------------------------------------------------------------------------
    if (core0) {
        int x;
        int base2;
        BENCH(base2, asm volatile("" ::: "memory"));
        printf("4. Copy of the bytes of a partial word (cycles per call)\n");
        printf("   bytes   memcpy  straight   loop\n");
        for (unsigned bytes = 1; bytes <= 7; bytes += (bytes == 1 ? 2 : 4)) {
            int m, s, l;
            BENCH(x, copy_memcpy(tmp_out, tmp_in, bytes));
            m = x - base2;
            BENCH(x, copy_straight(tmp_out, tmp_in, bytes));
            s = x - base2;
            BENCH(x, copy_loop(tmp_out, tmp_in, bytes));
            l = x - base2;
            printf("   %5u  %7d  %8d  %5d\n", bytes, m / 100, s / 100,
                   l / 100);
        }
        printf("\n");
    }
    snrt_cluster_hw_barrier();

    //--------------------------------------------------------------------------
    // 5. Eight cores streaming at once (FP32 hand function, whole words)
    //    Every compute core runs its own slice. Cycles per call, and the
    //    efficiency: one core alone / the slowest of the eight.
    //--------------------------------------------------------------------------
    {
        typedef struct {
            const char* name;
            unsigned w;       // words per core
            unsigned a_step;  // words between the starts of the slices of a
            unsigned b_off;   // words added to the start of every slice of b
            unsigned a_core;  // words added per core number to the start of a
        } Scen;
        static const Scen scens[] = {
            {"slices of 512 words (a multiple of 32)", 512, 512, 0, 0},
            {"slices of 513 words", 513, 513, 0, 0},
            {"512 words, each core starts one word later", 512, 512, 0, 1},
            {"512 words, b one word after a", 512, 512, 1, 0},
            {"512 words, b sixteen words after a", 512, 512, 16, 0},
            {"slices of 64 words (a multiple of 32)", 64, 64, 0, 0},
            {"slices of 65 words", 65, 65, 0, 0},
        };
        if (core0) {
            printf("5. Eight cores streaming at once (cycles per call)\n");
            printf("   %-46s  single    max    mean   efficiency\n", "");
        }
        for (unsigned s = 0; s < sizeof(scens) / sizeof(scens[0]); ++s) {
            const Scen* sc = &scens[s];
            const unsigned slice_words = sc->w;  // before any timing
            const size_t off =
                (size_t)core * sc->a_step + (size_t)core * sc->a_core;
            const void* pa = (const uint8_t*)a + off * 8;
            const void* pb = (const uint8_t*)b + (off + sc->b_off) * 8;

            // One core alone (core 0), as the reference.
            uint32_t single = 0;
            snrt_cluster_hw_barrier();
            if (core0) {
                hand_fp32(a, b, res, slice_words);
                const uint32_t t0 = snrt_mcycle();
                hand_fp32(a, b, res, slice_words);
                single = snrt_mcycle() - t0;
            }
            snrt_cluster_hw_barrier();

            // All the compute cores at once, each on its own slice. The dot
            // result of each core goes to its own word of the scratch memory,
            // and its cycles to its own entry of the cycles array.
            if (compute) {
                uint64_t* mine = (uint64_t*)scratch + core;
                hand_fp32(pa, pb, mine, slice_words);  // warm-up
            }
            snrt_cluster_hw_barrier();
            if (compute) {
                uint64_t* mine = (uint64_t*)scratch + core;
                const uint32_t t0 = snrt_mcycle();
                hand_fp32(pa, pb, mine, slice_words);
                cycles[core] = snrt_mcycle() - t0;
            }
            snrt_cluster_hw_barrier();
            if (core0) {
                uint32_t hi = 0, total = 0;
                for (unsigned c = 0; c < cores; ++c) {
                    const uint32_t v = ((volatile uint32_t*)cycles)[c];
                    if (v > hi) hi = v;
                    total += v;
                }
                const unsigned eff = single * 1000u / hi;
                printf("   %-46s  %6u  %6u  %6u   %3u.%u%%\n", sc->name,
                       (unsigned)single, (unsigned)hi,
                       (unsigned)(total / cores), eff / 10, eff % 10);
            }
            snrt_cluster_hw_barrier();
        }
    }
    return 0;
}
