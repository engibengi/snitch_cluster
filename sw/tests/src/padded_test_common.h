// Shared driver of the tests of the caller-padded vector functions
// (vector_vector_<op>_padded_l1.c, vector_scalar_<op>_padded_l1.c, axpy). Each
// test includes this header after defining:
//   TEST_NAME          string, the operation, for the messages
//   TEST_SKIP_FP8      1 if the operation does not exist in FP8 (division)
//   G_CORE(a,b,c,n,p)  call of the general one-core function (n elements)
//   G_MULTI(...)       call of the general eight-core function
//   P_CORE(a,b,c,w,p)  call of the padded one-core function (w WORDS)
//   P_MULTI(...)       call of the padded eight-core function
// where b is the second vector (or the scalar, in its first element) and p the
// precision. The calls are macros so that the same text is used with a run-time
// precision (the dispatcher) and with a constant one (the inlined code).
//
// The result of the padded function is compared, byte by byte over the n
// elements of the vector, with the result of the general function on the same
// inputs (the general functions are tested by the other tests). Both kinds of
// data are used: lengths that are not a whole number of words, held in buffers
// the caller has made room in for the whole last word (data that needs
// padding), and lengths that are a whole number of words (already padded). The
// elements that are padding are not compared: the padded functions compute on
// them and write them.
//
// The errors are counted per precision, number of cores and kind of data, and
// printed in a table: a failure says which function and which data.

#include "snrt.h"
#include "vector_scalar_l1.h"
#include "vector_vector_l1.h"
#include <stdint.h>
#include <stdio.h>

// Buffers: 75 elements of 8 bytes, plus one word (axpy reads y from b + 8).
#define CAP 608
#define SENTINEL 0xA5

static const unsigned lengths[] = {1, 3, 8, 9, 17, 31, 33, 75};
#define NLEN (sizeof(lengths) / sizeof(lengths[0]))
static const precision_t precs[] = {FP64, FP32, FP16, FP8};
static const char* pnames[] = {"FP64", "FP32", "FP16", "FP8"};

static uint8_t* ba;
static uint8_t* bb;
static uint8_t* bg;  // output of the general function
static uint8_t* bp;  // output of the padded function

// errors[precision][cores 1 or 8][0 = whole words, 1 = partial word]
static unsigned errors[4][2][2];
static unsigned checks[4][2][2];

// Pseudo-random bytes (a fixed generator). Whatever the bits mean in a
// precision (NaN, infinity, ...), the same operation on the same bits gives the
// same bits in both functions.
static void fill(uint8_t* p, unsigned seed) {
    uint32_t x = 0x9E3779B9u * (seed + 1);
    for (unsigned i = 0; i < CAP; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (uint8_t)(x >> 24);
    }
}

static unsigned words_of(unsigned n, precision_t prec) {
    const unsigned lanes = 8 / (unsigned)prec;
    return (n + lanes - 1) / lanes;
}

// One case with a run-time precision (not inlined: the dispatcher is used).
__attribute__((noinline)) static void run_case(precision_t prec, unsigned n,
                                               bool multi) {
    const unsigned core = snrt_cluster_core_idx();
    const bool compute = snrt_is_compute_core();
    const unsigned w = words_of(n, prec);
    if (core == 0)
        for (unsigned i = 0; i < CAP; ++i) bg[i] = bp[i] = SENTINEL;
    snrt_cluster_hw_barrier();
    if (compute) {
        if (multi) {
            G_MULTI(ba, bb, bg, n, prec);
        } else if (core == 0) {
            G_CORE(ba, bb, bg, n, prec);
        }
    }
    snrt_cluster_hw_barrier();
    if (compute) {
        if (multi) {
            P_MULTI(ba, bb, bp, w, prec);
        } else if (core == 0) {
            P_CORE(ba, bb, bp, w, prec);
        }
    }
    snrt_cluster_hw_barrier();
}

static void check(unsigned pi, unsigned n, bool multi, const char* what) {
    const unsigned bytes = n * (unsigned)precs[pi];
    const unsigned part = (n * (unsigned)precs[pi]) % 8 != 0;
    unsigned bad = 0;
    for (unsigned i = 0; i < bytes; ++i)
        if (bg[i] != bp[i]) {
            if (bad < 2)
                printf("  FAIL %s %s n=%u %s: byte %u general %02x padded "
                       "%02x\n",
                       pnames[pi], what, n, multi ? "8 cores" : "1 core", i,
                       bg[i], bp[i]);
            ++bad;
        }
    errors[pi][multi][part] += bad ? 1 : 0;
    ++checks[pi][multi][part];
}

int main() {
    const unsigned core = snrt_cluster_core_idx();
    const bool compute = snrt_is_compute_core();
    // Every core must make the same allocation calls in the same order.
    ba = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bb = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bg = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bp = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    if (core == 0) {
        fill(ba, 1);
        fill(bb, 2);
    }
    snrt_cluster_hw_barrier();

    for (unsigned pi = 0; pi < 4; ++pi) {
        if (TEST_SKIP_FP8 && precs[pi] == FP8) continue;
        for (unsigned li = 0; li < NLEN; ++li)
            for (unsigned m = 0; m < 2; ++m) {
                run_case(precs[pi], lengths[li], m);
                if (core == 0) check(pi, lengths[li], m, "run-time precision");
                snrt_cluster_hw_barrier();
            }
    }

    // The constant-precision path (inlined), FP32: a length with a partial word
    // and one without, on one core and on eight.
    const unsigned cn[] = {13, 16};
    for (unsigned k = 0; k < 2; ++k)
        for (unsigned m = 0; m < 2; ++m) {
            const unsigned n = cn[k];
            const bool multi = m;
            if (core == 0)
                for (unsigned i = 0; i < CAP; ++i) bg[i] = bp[i] = SENTINEL;
            snrt_cluster_hw_barrier();
            if (compute) {
                if (multi) {
                    G_MULTI(ba, bb, bg, n, FP32);
                } else if (core == 0) {
                    G_CORE(ba, bb, bg, n, FP32);
                }
            }
            snrt_cluster_hw_barrier();
            if (compute) {
                if (multi) {
                    P_MULTI(ba, bb, bp, (n + 1) / 2, FP32);
                } else if (core == 0) {
                    P_CORE(ba, bb, bp, (n + 1) / 2, FP32);
                }
            }
            snrt_cluster_hw_barrier();
            if (core == 0) check(1, n, multi, "constant precision");
            snrt_cluster_hw_barrier();
        }

    if (core != 0) return 0;
    unsigned total = 0;
    printf("%s, padded against general (cases with errors / cases)\n",
           TEST_NAME);
    printf("          1 core                    8 cores\n");
    printf("          whole words  partial word  whole words  partial word\n");
    for (unsigned pi = 0; pi < 4; ++pi) {
        if (TEST_SKIP_FP8 && precs[pi] == FP8) continue;
        printf("  %-5s", pnames[pi]);
        for (unsigned m = 0; m < 2; ++m)
            for (unsigned part = 0; part < 2; ++part) {
                if (checks[pi][m][part])
                    printf("   %3u / %-3u   ", errors[pi][m][part],
                           checks[pi][m][part]);
                else
                    printf("       -        ");
                total += errors[pi][m][part];
            }
        printf("\n");
    }
    printf("%s: %s (%u cases with errors)\n", TEST_NAME,
           total ? "FAIL" : "PASS", total);
    return total ? 1 : 0;
}
