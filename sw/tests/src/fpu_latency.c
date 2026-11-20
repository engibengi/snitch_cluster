// FPU latency / throughput microbenchmark (not a pass/fail test).
//
// Runs on compute core 0 only; the other cores just wait at the barrier. No
// SSR and no memory traffic in the measured loops: only FREP-repeated FP
// instructions on registers, so the numbers are the FPU's own.
//
// 1. Dependent chains. A hardware loop (frep) whose body has K instructions,
//    each updating its own accumulator: ft3 = op(ft3, ft11), ft4 = op(ft4, ft11),
//    ... Instructions of one iteration are independent of each other, but each
//    one depends on the same accumulator of the previous iteration.
//      K = 1  -> a pure dependency chain: cycles per instruction = latency.
//      K grows -> the FPU pipeline fills; cycles per instruction falls to 1.0
//                 (one instruction per cycle). The K where it reaches 1.0 is
//                 the number of independent accumulators needed.
// 2. The axpy shape: K multiplies into temporaries, then K adds that use them
//    (vfmul.r then vfadd, as in the packed axpy). Cycles per (mul + add) pair.
//
// Cycles per instruction are computed from the difference between two run
// lengths, so loop and fence overheads cancel out.
//
// Output: one line per instruction, cycles per instruction for K = 1..8.

#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

#define R1 64   // iterations, short run
#define R2 192  // iterations, long run

// Bodies of K instructions I, each on its own accumulator (ft3 ... ft10), with
// ft11 as the second operand.
#define B1(I) I " ft3, ft3, ft11 \n"
#define B2(I) B1(I) I " ft4, ft4, ft11 \n"
#define B3(I) B2(I) I " ft5, ft5, ft11 \n"
#define B4(I) B3(I) I " ft6, ft6, ft11 \n"
#define B5(I) B4(I) I " ft7, ft7, ft11 \n"
#define B6(I) B5(I) I " ft8, ft8, ft11 \n"
#define B7(I) B6(I) I " ft9, ft9, ft11 \n"
#define B8(I) B7(I) I " ft10, ft10, ft11 \n"

// Same, for the fused multiply-add with four operands: ft{3+j} = ft11 * ft11
// + ft{3+j} (the accumulator is the addend).
#define F1(I) I " ft3, ft11, ft11, ft3 \n"
#define F2(I) F1(I) I " ft4, ft11, ft11, ft4 \n"
#define F3(I) F2(I) I " ft5, ft11, ft11, ft5 \n"
#define F4(I) F3(I) I " ft6, ft11, ft11, ft6 \n"
#define F5(I) F4(I) I " ft7, ft11, ft11, ft7 \n"
#define F6(I) F5(I) I " ft8, ft11, ft11, ft8 \n"
#define F7(I) F6(I) I " ft9, ft11, ft11, ft9 \n"
#define F8(I) F7(I) I " ft10, ft11, ft11, ft10 \n"

// Same, for the two-operand reductions (vfsum, vfsumex): rd += sum of the
// lanes of ft11.
#define S1(I) I " ft3, ft11 \n"
#define S2(I) S1(I) I " ft4, ft11 \n"
#define S3(I) S2(I) I " ft5, ft11 \n"
#define S4(I) S3(I) I " ft6, ft11 \n"
#define S5(I) S4(I) I " ft7, ft11 \n"
#define S6(I) S5(I) I " ft8, ft11 \n"
#define S7(I) S6(I) I " ft9, ft11 \n"
#define S8(I) S7(I) I " ft10, ft11 \n"

// axpy shape: K multiplies ft{3+j} = ft11 * fs2, then K adds
// ft{3+j} = ft{3+j} + fs3.
#define M1(MUL, ADD) MUL " ft3, ft11, fs2 \n"
#define M2(MUL, ADD) M1(MUL, ADD) MUL " ft4, ft11, fs2 \n"
#define M3(MUL, ADD) M2(MUL, ADD) MUL " ft5, ft11, fs2 \n"
#define M4(MUL, ADD) M3(MUL, ADD) MUL " ft6, ft11, fs2 \n"
#define A1(ADD) ADD " ft3, ft3, fs3 \n"
#define A2(ADD) A1(ADD) ADD " ft4, ft4, fs3 \n"
#define A3(ADD) A2(ADD) ADD " ft5, ft5, fs3 \n"
#define A4(ADD) A3(ADD) ADD " ft6, ft6, fs3 \n"

#define STR2(x) #x
#define STR(x) STR2(x)
// 2 * K, as a literal (the body has K multiplies and K adds).
#define D1 2
#define D2 4
#define D3 6
#define D4 8
#define DBL(K) D##K

#define CLOBBERS                                                           \
    "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",       \
        "fs2", "fs3", "memory"

// Run `reps` iterations of a K-instruction chain body and return the cycles.
#define CHAINB(BODY, I, K, reps)                                           \
    ({                                                                     \
        uint32_t t0, t1;                                                   \
        snrt_fpu_fence();                                                  \
        t0 = snrt_mcycle();                                                \
        asm volatile("frep.o %[n], " STR(K) ", 0, 0 \n" BODY##K(I)         \
                     :                                                     \
                     : [ n ] "r"((reps)-1)                                 \
                     : CLOBBERS);                                          \
        snrt_fpu_fence();                                                  \
        t1 = snrt_mcycle();                                                \
        t1 - t0;                                                           \
    })

#define CHAIN(I, K, reps) CHAINB(B, I, K, reps)

// Run `reps` iterations of the K-pair multiply-then-add body.
#define MULADD(MUL, ADD, K, reps)                                          \
    ({                                                                     \
        uint32_t t0, t1;                                                   \
        snrt_fpu_fence();                                                  \
        t0 = snrt_mcycle();                                                \
        asm volatile("frep.o %[n], " STR(DBL(K)) ", 0, 0 \n"         \
                         M##K(MUL, ADD) A##K(ADD)                          \
                     :                                                     \
                     : [ n ] "r"((reps)-1)                                 \
                     : CLOBBERS);                                          \
        snrt_fpu_fence();                                                  \
        t1 = snrt_mcycle();                                                \
        t1 - t0;                                                           \
    })

// Cycles per iteration x 100, from the two run lengths, then per instruction.
static unsigned per_instr_x100(uint32_t c1, uint32_t c2, unsigned k) {
    return (unsigned)(((c2 - c1) * 100u) / ((R2 - R1) * k));
}

static void print_row(const char* name, const unsigned* v, unsigned n) {
    printf("%-14s", name);
    for (unsigned i = 0; i < n; ++i) printf(" %2u.%02u", v[i] / 100, v[i] % 100);
    printf("\n");
}

// One chain row: cycles per instruction for K = 1 ... 8. BODY is B (instr rd,
// rd, ft11), F (fused multiply-add, four operands) or S (instr rd, ft11).
#define CHAIN_ROWB(BODY, name, instr)                                      \
    do {                                                                   \
        unsigned v[8];                                                     \
        v[0] = per_instr_x100(CHAINB(BODY, instr, 1, R1),            \
                                  CHAINB(BODY, instr, 1, R2), 1);   \
        v[1] = per_instr_x100(CHAINB(BODY, instr, 2, R1),            \
                                  CHAINB(BODY, instr, 2, R2), 2);   \
        v[2] = per_instr_x100(CHAINB(BODY, instr, 3, R1),            \
                                  CHAINB(BODY, instr, 3, R2), 3);   \
        v[3] = per_instr_x100(CHAINB(BODY, instr, 4, R1),            \
                                  CHAINB(BODY, instr, 4, R2), 4);   \
        v[4] = per_instr_x100(CHAINB(BODY, instr, 5, R1),            \
                                  CHAINB(BODY, instr, 5, R2), 5);   \
        v[5] = per_instr_x100(CHAINB(BODY, instr, 6, R1),            \
                                  CHAINB(BODY, instr, 6, R2), 6);   \
        v[6] = per_instr_x100(CHAINB(BODY, instr, 7, R1),            \
                                  CHAINB(BODY, instr, 7, R2), 7);   \
        v[7] = per_instr_x100(CHAINB(BODY, instr, 8, R1),            \
                                  CHAINB(BODY, instr, 8, R2), 8);   \
        print_row(name, v, 8);                                             \
    } while (0)
#define CHAIN_ROW(name, instr) CHAIN_ROWB(B, name, instr)

// One multiply+add row: cycles per (multiply + add) pair for K = 1 ... 4.
#define MULADD_ROW(name, mul, add)                                         \
    do {                                                                   \
        unsigned v[4];                                                     \
        v[0] = per_instr_x100(MULADD(mul, add, 1, R1),                     \
                              MULADD(mul, add, 1, R2), 1);                 \
        v[1] = per_instr_x100(MULADD(mul, add, 2, R1),                     \
                              MULADD(mul, add, 2, R2), 2);                 \
        v[2] = per_instr_x100(MULADD(mul, add, 3, R1),                     \
                              MULADD(mul, add, 3, R2), 3);                 \
        v[3] = per_instr_x100(MULADD(mul, add, 4, R1),                     \
                              MULADD(mul, add, 4, R2), 4);                 \
        print_row(name, v, 4);                                             \
    } while (0)

// Initialise the registers the loops use, from L1 (no FP16 instruction).
// The values do not matter for timing: FP exceptions only set flags.
static void init_regs(const double* one) {
    asm volatile(
        "fld ft3, 0(%0) \n fld ft4, 0(%0) \n fld ft5, 0(%0) \n"
        "fld ft6, 0(%0) \n fld ft7, 0(%0) \n fld ft8, 0(%0) \n"
        "fld ft9, 0(%0) \n fld ft10, 0(%0) \n fld ft11, 0(%0) \n"
        "fld fs2, 0(%0) \n fld fs3, 0(%0) \n"
        :
        : "r"(one)
        : "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
          "fs2", "fs3", "memory");
}

// The benchmark runs in functions only compute core 0 calls: the DMA core has
// no FP16/FP8/FREP support and must never execute these instructions.
__attribute__((noinline)) static void run_benchmark(const double* one) {
    init_regs(one);

    printf("Cycles per instruction, K independent accumulators (K = 1..8)\n");
    printf("%-14s    K=1   K=2   K=3   K=4   K=5   K=6   K=7   K=8\n", "");
    CHAIN_ROW("fadd.d", "fadd.d");
    CHAIN_ROW("fmul.d", "fmul.d");
    CHAIN_ROW("fdiv.d", "fdiv.d");
    CHAIN_ROW("fmin.d", "fmin.d");
    CHAIN_ROW("fmax.d", "fmax.d");
    CHAIN_ROWB(F, "fmadd.d", "fmadd.d");
    CHAIN_ROW("vfadd.s", "vfadd.s");
    CHAIN_ROW("vfadd.h", "vfadd.h");
    CHAIN_ROW("vfadd.b", "vfadd.b");
    CHAIN_ROW("vfmul.s", "vfmul.s");
    CHAIN_ROW("vfmul.h", "vfmul.h");
    CHAIN_ROW("vfmul.b", "vfmul.b");
    CHAIN_ROW("vfdiv.s", "vfdiv.s");
    CHAIN_ROW("vfdiv.h", "vfdiv.h");
    CHAIN_ROW("vfmin.s", "vfmin.s");
    CHAIN_ROW("vfmin.h", "vfmin.h");
    CHAIN_ROW("vfmin.b", "vfmin.b");
    CHAIN_ROW("vfmax.s", "vfmax.s");
    CHAIN_ROW("vfmax.h", "vfmax.h");
    CHAIN_ROW("vfmax.b", "vfmax.b");
    CHAIN_ROW("vfmac.s", "vfmac.s");
    CHAIN_ROW("vfmac.h", "vfmac.h");
    CHAIN_ROW("vfmac.b", "vfmac.b");
    CHAIN_ROW("vfdotpex.s.h", "vfdotpex.s.h");
    CHAIN_ROW("vfdotpex.h.b", "vfdotpex.h.b");
    CHAIN_ROWB(S, "vfsum.s", "vfsum.s");
    CHAIN_ROWB(S, "vfsum.h", "vfsum.h");
    CHAIN_ROWB(S, "vfsum.b", "vfsum.b");
    CHAIN_ROWB(S, "vfsumex.s.h", "vfsumex.s.h");
    CHAIN_ROWB(S, "vfsumex.h.b", "vfsumex.h.b");

    printf("\nCycles per (multiply + add) pair, axpy shape (K = 1..4)\n");
    printf("(add = fadd.d, vfadd.s, vfadd.h, vfadd.b respectively)\n");
    printf("%-14s    K=1   K=2   K=3   K=4\n", "");
    MULADD_ROW("fmul.d+fadd.d", "fmul.d", "fadd.d");
    MULADD_ROW("vfmul.r.s+add", "vfmul.r.s", "vfadd.s");
    MULADD_ROW("vfmul.r.h+add", "vfmul.r.h", "vfadd.h");
    MULADD_ROW("vfmul.r.b+add", "vfmul.r.b", "vfadd.b");
}

int main() {
    // 1.0 in double precision. The bit pattern is not meaningful for the packed
    // formats, which does not matter for timing.
    double* one = (double*)snrt_l1_alloc_cluster_local(sizeof(double), 8);

    if (snrt_cluster_core_idx() == 0) {
        *one = 1.0;
        run_benchmark(one);
    }
    snrt_cluster_hw_barrier();
    return 0;
}
