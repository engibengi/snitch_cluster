// Latency of the basic components of a function call on one compute core (not a
// pass/fail test). Together with fpu_latency.c (the FP instructions) and
// overhead_probe.c (calls, saved registers, fences, divisions, the barrier, the
// combine of the dot product and the partial-word copy), it measures every
// component that a library function is made of, so that the cost of a function
// can be calculated from its instructions and then compared with a measurement.
//
// What this file measures:
//   1. The measurement itself: the cost of reading the cycle counter.
//   2. Integer: an ALU instruction, a taken and a not-taken branch.
//   3. Memory in L1: a load that is used at once, a store, and the same in FP.
//   4. SSR and FREP: one configuration write, the setup of two streams, the
//      cost of a stream of W words (start-up plus one cycle per word), and the
//      launch of a hardware loop.
//   5. The wait for the FPU: the cost of the fence (the integer core waits for
//      the FP pipeline) after a chain of d dependent FP instructions.
//   6. The instruction cache: the first call of a function against the next
//      ones, for functions of 16 to 128 instructions.
//
// Everything that a timed region uses is in L1 or in a register (the buffers
// are allocated in L1; run-time values are made opaque in registers). The
// functions that are measured are called once before the timing starts, except
// in section 6, which measures that first call. The code itself is fetched from
// L3 through the instruction cache.
//
// Only core 0 runs; the other cores wait at the barrier at the end.

#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

#define REPS 32

// Cycles x 100 of one repetition of `body` (after one untimed repetition).
#define BENCH(var, body)                                                    \
    do {                                                                    \
        {body;}                                                             \
        const uint32_t t0_ = snrt_mcycle();                                 \
        for (int i_ = 0; i_ < REPS; ++i_) {body;}                           \
        const uint32_t t1_ = snrt_mcycle();                                 \
        (var) = (int)(((t1_ - t0_) * 100u) / REPS);                         \
    } while (0)

#define OPAQUE(x) asm volatile("" : "+r"(x))
// Consume a value without moving it: an integer in an integer register, a
// double in an FP register (USE on a double would copy it through the stack,
// which costs cycles and is not what is measured).
#define USE(x) asm volatile("" : : "r"(x))
#define USEF(x) asm volatile("" : : "f"(x))

static void print_x100(int v) {
    printf("%s%d.%02d", v < 0 ? "-" : "", (v < 0 ? -v : v) / 100,
           (v < 0 ? -v : v) % 100);
}

static void show(const char* name, int x100, int base100) {
    printf("  %-52s ", name);
    print_x100(x100 - base100);
    printf("\n");
}

//==============================================================================
// Functions for section 6: straight-line code of a given number of
// instructions, in separate functions that are called for the first time in
// the order of the table. Every instruction is a nop; the call and the return
// are not included in the count.
//==============================================================================

#define N4 "nop\nnop\nnop\nnop\n"
#define N16 N4 N4 N4 N4
#define N32 N16 N16
#define N64 N32 N32
#define N128 N64 N64

__attribute__((noinline)) static void code16() { asm volatile(N16); }
__attribute__((noinline)) static void code32() { asm volatile(N32); }
__attribute__((noinline)) static void code64() { asm volatile(N64); }
__attribute__((noinline)) static void code128() { asm volatile(N128); }

// The same sizes, with a taken jump every four instructions, so the fetch is
// not sequential (sequential code is prefetched, and its first call is hardly
// slower than the next ones).
#define J4 "nop\nnop\nnop\nj 1f\n1:\n"
#define J16 J4 J4 J4 J4
#define J32 J16 J16
#define J64 J32 J32
#define J128 J64 J64
__attribute__((noinline)) static void jumpy16() { asm volatile(J16); }
__attribute__((noinline)) static void jumpy32() { asm volatile(J32); }
__attribute__((noinline)) static void jumpy64() { asm volatile(J64); }
__attribute__((noinline)) static void jumpy128() { asm volatile(J128); }

// A stream of `words` words (a multiple of 4) that are summed, the least that
// one stream of FP work needs: the two streams, one hardware loop, the fence
// and the disable. Four accumulators, one per instruction of the loop body, so
// that the loop is not limited by the latency of the multiply-add (one
// accumulator would give one word every four cycles). Used to separate the
// start-up cost from the cost per word.
__attribute__((noinline)) static void stream_words(const void* a,
                                                   const void* b,
                                                   size_t words) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    snrt_ssr_loop_1d(SNRT_SSR_DM0, words, 8);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, words, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)a);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)b);
    snrt_ssr_enable();
    asm volatile(
        "frep.o %[n], 4, 0, 0 \n"
        "vfmac.s %[a0], ft0, ft1 \n"
        "vfmac.s %[a1], ft0, ft1 \n"
        "vfmac.s %[a2], ft0, ft1 \n"
        "vfmac.s %[a3], ft0, ft1 \n"
        : [ a0 ] "+f"(a0), [ a1 ] "+f"(a1), [ a2 ] "+f"(a2), [ a3 ] "+f"(a3)
        : [ n ] "r"(words / 4 - 1)
        : "ft0", "ft1", "ft2", "memory");
    snrt_fpu_fence();
    snrt_ssr_disable();
    USEF(a0);
    USEF(a1);
    USEF(a2);
    USEF(a3);
}

int main() {
    // Every core must make the same allocation calls in the same order.
    uint8_t* mem = (uint8_t*)snrt_l1_alloc_cluster_local(4096 * 2, 8);
    uint8_t* a = mem;
    uint8_t* b = mem + 4096;

    if (snrt_cluster_core_idx() == 0) {
        int x, base;
        unsigned v = 1, nz = 1, zero = 0;
        OPAQUE(nz);
        OPAQUE(zero);

        printf("Latency of the components of a function call "
               "(cycles per execution)\n\n");

        // 1. The measurement itself.
        printf("1. Measurement\n");
        BENCH(base, asm volatile("" ::: "memory"));
        show("empty loop body (subtracted below)", base, 0);
        BENCH(x, { uint32_t t = snrt_mcycle(); USE(t); });
        show("read of the cycle counter", x, base);

        // 2. Integer instructions.
        printf("\n2. Integer\n");
        BENCH(x, asm volatile("addi %0, %0, 1" : "+r"(v)));
        show("addi, dependent chain", x, base);
        BENCH(x, asm volatile("beqz %0, 1f \n nop \n 1: \n" : : "r"(nz)));
        show("branch not taken, plus a nop that is executed", x, base);
        BENCH(x, asm volatile("bnez %0, 1f \n nop \n 1: \n" : : "r"(nz)));
        show("branch taken, the nop is skipped", x, base);
        BENCH(x, asm volatile("j 1f \n nop \n 1: \n"));
        show("unconditional jump", x, base);
        BENCH(x, asm volatile("bnez %0, 1f \n nop \n 1: \n" : : "r"(zero)));
        show("branch not taken (bnez on zero), plus a nop", x, base);

        // 3. Memory in L1.
        printf("\n3. Memory in L1\n");
        {
            volatile uint32_t* p = (volatile uint32_t*)a;
            p[0] = (uint32_t)(uintptr_t)a;  // points to itself
            p[1] = 0;
            uint32_t q = (uint32_t)(uintptr_t)a;
            BENCH(x, asm volatile("lw %0, 0(%0)" : "+r"(q) : : "memory"));
            show("lw, dependent chain (load-to-use latency)", x, base);
            uint32_t r;
            BENCH(x, {
                asm volatile("lw %0, 4(%1)" : "=r"(r) : "r"(a) : "memory");
                USE(r);
            });
            show("lw, then a use of the value", x, base);
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            BENCH(x, {
                asm volatile(
                    "lw %0, 0(%8) \n lw %1, 4(%8) \n lw %2, 8(%8) \n"
                    "lw %3, 12(%8) \n lw %4, 16(%8) \n lw %5, 20(%8) \n"
                    "lw %6, 24(%8) \n lw %7, 28(%8) \n"
                    : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3), "=&r"(r4),
                      "=&r"(r5), "=&r"(r6), "=&r"(r7)
                    : "r"(a)
                    : "memory");
                USE(r0);
                USE(r1);
                USE(r2);
                USE(r3);
                USE(r4);
                USE(r5);
                USE(r6);
                USE(r7);
            });
            show("8 independent lw, then a use of each (8 loads)", x, base);
            BENCH(x, asm volatile("sw %0, 8(%1)" : : "r"(v), "r"(a)
                                  : "memory"));
            show("sw", x, base);
            double f;
            ((volatile double*)a)[2] = 1.0;
            BENCH(x, {
                asm volatile("fld %0, 16(%1)" : "=f"(f) : "r"(a) : "memory");
                USEF(f);
            });
            show("fld, then a use of the value", x, base);
            BENCH(x, asm volatile("fsd %0, 24(%1)" : : "f"(f), "r"(a)
                                  : "memory"));
            show("fsd", x, base);
            BENCH(x, {
                asm volatile("fld %0, 16(%1)" : "=f"(f) : "r"(a) : "memory");
                asm volatile("fsd %0, 24(%1)" : : "f"(f), "r"(a) : "memory");
            });
            show("fld then fsd of the loaded value", x, base);
        }

        // 4. SSR and FREP.
        printf("\n4. SSR and FREP\n");
        {
            BENCH(x, asm volatile("scfgwi %0, 0 | 2<<5" : : "r"(v)));
            show("one scfgwi (SSR configuration write)", x, base);
            BENCH(x, {
                asm volatile("csrsi ssr, 1");
                asm volatile("csrci ssr, 1");
            });
            show("csrsi ssr + csrci ssr (enable, disable)", x, base);

            printf("   stream of W words (a multiple of 4), two read streams, "
                   "four vfmac per iteration of the hardware loop, from the "
                   "first configuration write to after the disable:\n");
            static const unsigned ws[] = {4, 8, 16, 32, 64, 128, 512};
            for (unsigned i = 0; i < sizeof(ws) / sizeof(ws[0]); ++i) {
                const size_t w = ws[i];
                BENCH(x, stream_words(a, b, w));
                printf("    stream of %4u words                          ",
                       (unsigned)w);
                print_x100(x - base);
                printf("\n");
            }

            // The launch of a hardware loop: n iterations of four independent
            // instructions (distinct destinations, so no dependency between
            // them: one instruction per cycle), then a fence.
            for (unsigned n = 1; n <= 8; n *= 2) {
                unsigned cnt = n - 1;
                OPAQUE(cnt);
                BENCH(x, {
                    asm volatile(
                        "frep.o %[c], 4, 0, 0 \n"
                        "fadd.d ft3, ft8, ft9 \n"
                        "fadd.d ft4, ft8, ft9 \n"
                        "fadd.d ft5, ft8, ft9 \n"
                        "fadd.d ft6, ft8, ft9 \n"
                        :
                        : [ c ] "r"(cnt)
                        : "ft3", "ft4", "ft5", "ft6", "memory");
                    snrt_fpu_fence();
                });
                printf("  frep, %u iteration(s) of 4 fadd.d, then a fence   ",
                       n);
                print_x100(x - base);
                printf("\n");
            }
        }

        // 5. The wait for the FPU.
        printf("\n5. Fence (integer core waits for the FPU)\n");
        BENCH(x, snrt_fpu_fence());
        show("fence, FPU idle", x, base);
#define FENCE_AFTER(D, SEQ)                                                  \
    BENCH(x, {                                                              \
        asm volatile(SEQ ::: "ft3");                                        \
        snrt_fpu_fence();                                                   \
    });                                                                     \
    show("fence after " D, x, base)
        FENCE_AFTER("1 vfadd.s", "vfadd.s ft3, ft3, ft3\n");
        FENCE_AFTER("2 dependent vfadd.s",
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n");
        FENCE_AFTER("3 dependent vfadd.s",
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n"
                    "vfadd.s ft3, ft3, ft3\n");
        FENCE_AFTER("4 dependent vfadd.s",
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n"
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n");
        FENCE_AFTER("8 dependent vfadd.s",
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n"
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n"
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n"
                    "vfadd.s ft3, ft3, ft3\nvfadd.s ft3, ft3, ft3\n");
        FENCE_AFTER("the dot combine: 3 vfadd.s, vfsum.s, fsw",
                    "vfadd.s ft3, ft3, ft4\nvfadd.s ft5, ft5, ft6\n"
                    "vfadd.s ft3, ft3, ft5\nvfsum.s ft7, ft3\n");
        FENCE_AFTER("the zero-initialisation of 4 accumulators",
                    "fcvt.d.w ft3, zero\nfmv.d ft4, ft3\nfmv.d ft5, ft3\n"
                    "fmv.d ft6, ft3\n");
    }
    snrt_cluster_hw_barrier();

    // 6. The instruction cache: the first call of a function against the next
    // ones. Core 0 only; the functions have not been called before.
    if (snrt_cluster_core_idx() == 0) {
        printf("\n6. Instruction cache: first call against later calls\n");
        printf("   (straight-line nops; the cache line is 64 bytes, "
               "16 instructions)\n");
        printf("   code      instructions   first call   later call   "
               "difference\n");
        void (*fns[8])() = {code16,  code32,  code64,  code128,
                            jumpy16, jumpy32, jumpy64, jumpy128};
        const unsigned sizes[8] = {16, 32, 64, 128, 16, 32, 64, 128};
        for (unsigned i = 0; i < 8; ++i) {
            void (*f)() = fns[i];
            const uint32_t t0 = snrt_mcycle();
            f();
            const uint32_t t1 = snrt_mcycle();
            f();
            f();
            const uint32_t t2 = snrt_mcycle();
            f();
            const uint32_t t3 = snrt_mcycle();
            (void)t2;
            printf("   %s %9u   %10u   %10u   %10d\n",
                   i < 4 ? "straight" : "jumpy   ", sizes[i],
                   (unsigned)(t1 - t0), (unsigned)(t3 - t2),
                   (int)(t1 - t0) - (int)(t3 - t2));
        }
    }
    snrt_cluster_hw_barrier();
    return 0;
}
