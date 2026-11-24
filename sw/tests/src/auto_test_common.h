// Shared driver of the tests of the automatic one-core / eight-core functions
// (snrt_<fam>_<op>_auto_l1 and snrt_<fam>_<op>_padded_auto_l1). Each test
// includes this header after defining:
//   TEST_NAME        string, the operation, for the messages
//   TEST_SKIP_FP8    1 if the operation does not exist in FP8 (division)
//   TEST_CLASS       the snrt_vec_cost_class_t of the operation
//   G_MULTI(a,b,c,n,p)   reference: the general eight-core function
//   A_AUTO(a,b,c,n,p)    the general automatic function (n elements)
//   PA_AUTO(a,b,c,w,p)   the padded automatic function (w WORDS)
// where b is the second vector (or the scalar, in its first element).
//
// Every compute core calls every function with the same arguments. The result
// of the automatic function is compared, byte by byte over the n elements, with
// the reference. Lengths: 0 to 24 words (both sides of every threshold: the
// thresholds are between 4 and 16 words, see words_list), each whole and with the largest
// partial word (data that needs padding; for the padded function the buffers
// have room for the whole word) so both kinds of data are tested. Also checked:
// the general function writes nothing after the n elements, and the padded
// function nothing after the last word. The errors are counted per precision,
// function, path taken (one core or eight, as chosen by the library) and kind
// of data, and printed in a table. The constant-precision (inlined) path is
// tested on FP32 at the end.

#include "snrt.h"
#include "vector_scalar_l1.h"
#include "vector_vector_l1.h"
#include <stdint.h>
#include <stdio.h>

#define CAP 608
#define SENTINEL 0xA5

// Around every threshold: division 4 (padded) and 5, packed axpy 9 and 13,
// the simple operations and FP64 axpy 13 to 16.
static const unsigned words_list[] = {0, 1, 3, 4, 5, 8, 9, 12, 13, 15, 16, 17, 24};
#define NW (sizeof(words_list) / sizeof(words_list[0]))
static const precision_t precs[] = {FP64, FP32, FP16, FP8};
static const char* pnames[] = {"FP64", "FP32", "FP16", "FP8"};

static uint8_t *ba, *bb, *bref, *bout;

// errors/checks[precision][function: 0 general, 1 padded][path: 0 one core, 1
// eight cores][data: 0 whole words, 1 partial word]
static unsigned errors[4][2][2][2];
static unsigned checks[4][2][2][2];

static void fill(uint8_t* p, unsigned seed) {
    uint32_t x = 0x9E3779B9u * (seed + 1);
    for (unsigned i = 0; i < CAP; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (uint8_t)(x >> 24);
    }
}

// CAP is a multiple of 8: word stores (the simulator is slow on byte loops).
static void clear(uint8_t* p) {
    uint64_t* q = (uint64_t*)p;
    for (unsigned i = 0; i < CAP / 8; ++i) q[i] = 0xA5A5A5A5A5A5A5A5ull;
}

// Not inlined: a run-time precision (the dispatcher).
__attribute__((noinline)) static void run_case(precision_t prec, unsigned n,
                                               unsigned w, bool padded) {
    const bool compute = snrt_is_compute_core();
    if (snrt_cluster_core_idx() == 0) {
        clear(bref);
        clear(bout);
    }
    snrt_cluster_hw_barrier();
    if (compute) {
        G_MULTI(ba, bb, bref, n, prec);
    }
    snrt_cluster_hw_barrier();
    if (compute) {
        if (padded) {
            PA_AUTO(ba, bb, bout, w, prec);
        } else {
            A_AUTO(ba, bb, bout, n, prec);
        }
    }
    snrt_cluster_hw_barrier();
}

static void check(unsigned pi, unsigned n, unsigned w, bool padded,
                  const char* what) {
    const unsigned lanes = 8 / (unsigned)precs[pi];
    const unsigned bytes = n * (unsigned)precs[pi];
    const unsigned limit = padded ? w * 8 : bytes;  // nothing is written after
    const unsigned path =
        snrt_vec_use_multicore(TEST_CLASS, padded ? w : n, precs[pi], padded)
            ? 1
            : 0;
    const unsigned data = (n % lanes) ? 1 : 0;
    unsigned bad = 0;
    for (unsigned i = 0; i < bytes; ++i)
        if (bref[i] != bout[i]) {
            if (!bad)
                printf("  FAIL %s %s n=%u (%s): byte %u reference %02x got "
                       "%02x\n",
                       pnames[pi], what, n, path ? "8 cores" : "1 core", i,
                       bref[i], bout[i]);
            ++bad;
        }
    // Nothing may be written after the data: the 64 bytes that follow.
    for (unsigned i = limit; i < limit + 64 && i < CAP; ++i)
        if (bout[i] != SENTINEL) {
            if (!bad)
                printf("  FAIL %s %s n=%u (%s): byte %u after the data was "
                       "written (%02x)\n",
                       pnames[pi], what, n, path ? "8 cores" : "1 core", i,
                       bout[i]);
            ++bad;
            break;
        }
    errors[pi][padded][path][data] += bad ? 1 : 0;
    ++checks[pi][padded][path][data];
}

int main() {
    const unsigned core = snrt_cluster_core_idx();
    const bool compute = snrt_is_compute_core();
    ba = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bb = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bref = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    bout = (uint8_t*)snrt_l1_alloc_cluster_local(CAP, 8);
    if (core == 0) {
        fill(ba, 1);
        fill(bb, 2);
    }
    snrt_cluster_hw_barrier();

    for (unsigned pi = 0; pi < 4; ++pi) {
        if (TEST_SKIP_FP8 && precs[pi] == FP8) continue;
        const unsigned lanes = 8 / (unsigned)precs[pi];
        for (unsigned wi = 0; wi < NW; ++wi)
            for (unsigned kind = 0; kind < 2; ++kind) {
                if (kind == 1 && lanes == 1) continue;  // FP64: no partial word
                const unsigned w = words_list[wi];
                const unsigned partial = kind ? lanes - 1 : 0;
                if (kind == 1 && w == 0) continue;
                const unsigned n = w * lanes + partial;  // elements
                const unsigned pw = w + (partial ? 1 : 0);  // padded words
                for (unsigned padded = 0; padded < 2; ++padded) {
                    run_case(precs[pi], n, pw, padded);
                    if (core == 0)
                        check(pi, n, pw, padded,
                              padded ? "padded auto" : "general auto");
                    snrt_cluster_hw_barrier();
                }
            }
    }

    // Constant precision (inlined), FP32: below and above every threshold.
    const unsigned cn[] = {5, 13, 40};  // elements: 3, 7, 20 words
    for (unsigned k = 0; k < 3; ++k)
        for (unsigned padded = 0; padded < 2; ++padded) {
            const unsigned n = cn[k], pw = (n + 1) / 2;
            if (core == 0) {
                clear(bref);
                clear(bout);
            }
            snrt_cluster_hw_barrier();
            if (compute) G_MULTI(ba, bb, bref, n, FP32);
            snrt_cluster_hw_barrier();
            if (compute) {
                if (padded) {
                    PA_AUTO(ba, bb, bout, pw, FP32);
                } else {
                    A_AUTO(ba, bb, bout, n, FP32);
                }
            }
            snrt_cluster_hw_barrier();
            if (core == 0)
                check(1, n, pw, padded,
                      padded ? "padded auto (constant)"
                             : "general auto (constant)");
            snrt_cluster_hw_barrier();
        }

    if (core != 0) return 0;
    unsigned total = 0;
    printf("%s, automatic functions against the eight-core reference "
           "(cases with errors / cases)\n",
           TEST_NAME);
    printf("               general auto                              padded "
           "auto\n");
    printf("         1 core: whole  partial   8 cores: whole  partial   1 core: "
           "whole  partial   8 cores: whole  partial\n");
    for (unsigned pi = 0; pi < 4; ++pi) {
        if (TEST_SKIP_FP8 && precs[pi] == FP8) continue;
        printf("  %-5s", pnames[pi]);
        for (unsigned f = 0; f < 2; ++f)
            for (unsigned path = 0; path < 2; ++path)
                for (unsigned d = 0; d < 2; ++d) {
                    if (checks[pi][f][path][d])
                        printf("  %3u/%-3u", errors[pi][f][path][d],
                               checks[pi][f][path][d]);
                    else
                        printf("     -   ");
                    total += errors[pi][f][path][d];
                }
        printf("\n");
    }
    printf("%s auto: %s (%u cases with errors)\n", TEST_NAME,
           total ? "FAIL" : "PASS", total);
    return total ? 1 : 0;
}
