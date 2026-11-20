// Dot product on 1 and 8 cores (snrt_vec_dot_cores_l1 and
// snrt_vec_dot_padded_cores_l1, vector_reduce_l1.h): interleaved split of the
// words. The contiguous version this one replaced has its own test, in
// experiments/dot_versions/v2_contiguous/tests/vector_dot_cores_l1.c. See
// paper_notes/BANK_CONFLICTS.md.
//
// Correctness: every precision, accumulation mode, number of cores and length
// (with and without a partial last word) on sparse data: the vectors are zero,
// except for ones at chosen positions of both vectors, so the dot product is
// the number of those positions (a small integer that every format represents
// exactly, whatever the order of the additions). With the interleaved split,
// core c owns the words c, c + C, c + 2C, ...:
//  run A: a one at the first element of the first word of every core (count C);
//  run B: a one at the last element of the last whole word of every core (C);
//  run C: a one at the last element of the vector, if there is a partial word
//         (it is in the partial word, which has its own owner; count 1).
// A core that reads the wrong words, a wrong stride, a lost or double-counted
// word or a lost partial word changes the count.
// Each vector is also run through the padded function: when it is a whole
// number of words (already padded), and when it has a partial word, passed as
// a caller pads it (one word more, the elements after the vector being zero).
//
// The timing of these functions is in vector_dot_cores_latency_l1.c.

#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

// Words of the longest vector: 2096. One more word of room, for the partial
// word of FP8 (7 elements) of the longest vector.
#define MAXW 2097
#define SLOT 16
#define SENTINEL 0xA5

#define ALLOC(type, count) \
    ((type*)snrt_l1_alloc_cluster_local((count) * sizeof(type), 8))

// Lengths of the correctness check, in words: a power of two, the smallest
// length, primes and an odd length, and the longest.
static const unsigned check_words[] = {8, 9, 13, 75, 333, 2096};
#define N_CHECK (sizeof(check_words) / sizeof(check_words[0]))

// The bits of the integers 1 to 8 in every format. FP16 and FP8 (E5M2, the top
// byte of FP16) are listed; FP32 and FP64 are converted.
static const uint16_t half_bits[9] = {0,      0x3c00, 0x4000, 0x4200, 0x4400,
                                      0x4500, 0x4600, 0x4700, 0x4800};

// The bytes of a small integer (1 to 8) as an element of a precision. Run by
// compute cores only (the conversions are FP instructions).
__attribute__((noinline)) static uint64_t encode(precision_t p, unsigned v) {
    uint64_t bits = 0;
    if (p == FP64) {
        double d = (double)v;
        memcpy(&bits, &d, sizeof(d));
    } else if (p == FP32) {
        float f = (float)v;
        memcpy(&bits, &f, sizeof(f));
    } else if (p == FP16) {
        bits = half_bits[v];
    } else {
        bits = half_bits[v] >> 8;
    }
    return bits;
}

static void set_element(uint8_t* base, precision_t p, size_t i,
                        uint64_t bits) {
    volatile uint8_t* d = base + i * (size_t)p;
    for (unsigned k = 0; k < (unsigned)p; ++k)
        d[k] = (uint8_t)(bits >> (8 * k));
}

// Positions (element indices) of the ones of a run, for the interleaved split.
// Fills `pos` and returns how many. Written with divisions on purpose (the
// library uses shifts and masks).
static unsigned run_positions(int run, size_t n, size_t lanes, size_t cores,
                              size_t* pos) {
    const size_t words = n / lanes;
    unsigned k = 0;
    if (run == 2) {
        if (n % lanes) pos[k++] = n - 1;  // the partial word
        return k;
    }
    for (size_t c = 0; c < cores; ++c) {
        // first and last whole word owned by core c (words >= 8 >= cores)
        const size_t first_word = c;
        const size_t last_word = c + ((words - 1 - c) / cores) * cores;
        pos[k++] =
            run == 0 ? first_word * lanes : last_word * lanes + lanes - 1;
    }
    return k;
}

template <precision_t P, snrt_acc_t M>
__attribute__((noinline)) static unsigned check_one(
    const char* name, uint8_t* a, uint8_t* b, void* scratch, uint8_t* res,
    size_t n, unsigned log2) {
    const size_t lanes = 8 / (size_t)P;
    const size_t cores = (size_t)1 << log2;
    const precision_t rp = snrt_vec_dot_result_precision(P, M);
    const unsigned core = snrt_cluster_core_idx();
    const int compute = snrt_is_compute_core();
    unsigned errors = 0;

    for (int run = 0; run < 3; ++run) {
        size_t pos[8];
        const unsigned count = run_positions(run, n, lanes, cores, pos);
        if (count == 0) continue;  // run C without a partial word
        if (core == 0) {
            for (unsigned i = 0; i < SLOT; ++i)
                ((volatile uint8_t*)res)[i] = SENTINEL;
            for (unsigned k = 0; k < count; ++k) {
                set_element(a, P, pos[k], encode(P, 1));
                set_element(b, P, pos[k], encode(P, 1));
            }
        }
        // The same data through the padded function as well: variant 1 when
        // the vector is a whole number of words (already padded), variant 2
        // when it has a partial word, passed the way a caller pads it (one
        // word more; the elements after the vector are zero in the buffer).
        for (int variant = 0; variant < 3; ++variant) {
            if (variant == 1 && n % lanes) continue;
            if (variant == 2 && n % lanes == 0) continue;
            if (core == 0)
                for (unsigned i = 0; i < SLOT; ++i)
                    ((volatile uint8_t*)res)[i] = SENTINEL;
            snrt_fpu_fence();
            snrt_cluster_hw_barrier();

            const char* vname = variant == 1   ? " (padded)"
                                : variant == 2 ? " (padded, caller-padded)"
                                               : "";
            if (variant == 0)
                snrt_vec_dot_cores_l1(a, b, res, scratch, n, P, M, log2);
            else
                snrt_vec_dot_padded_cores_l1(
                    a, b, res, scratch, n / lanes + (variant == 2 ? 1 : 0), P,
                    M, log2);

            if (compute) {
                const uint64_t want = encode(rp, count);
                for (unsigned i = 0; i < SLOT; ++i) {
                    const uint8_t w = i < (unsigned)rp
                                          ? (uint8_t)(want >> (8 * i))
                                          : SENTINEL;
                    const uint8_t g = ((volatile uint8_t*)res)[i];
                    if (g != w) {
                        ++errors;
                        printf("core %u: %s, %u words + %u, %u cores, "
                               "run %c%s, byte %u: expected 0x%02x, "
                               "got 0x%02x\n",
                               core, name, (unsigned)(n / lanes),
                               (unsigned)(n % lanes), (unsigned)cores,
                               'A' + run, vname, i, (unsigned)w,
                               (unsigned)g);
                    }
                }
            }
            snrt_cluster_hw_barrier();
        }  // variants

        // Back to zero, for the next run and the next call.
        if (core == 0) {
            for (unsigned k = 0; k < count; ++k) {
                set_element(a, P, pos[k], 0);
                set_element(b, P, pos[k], 0);
            }
        }
    }
    return errors;
}

template <precision_t P, snrt_acc_t M>
static unsigned check_config(const char* name, uint8_t* a, uint8_t* b,
                             void* scratch, uint8_t* res) {
    const size_t lanes = 8 / (size_t)P;
    unsigned errors = 0;
    for (unsigned w = 0; w < N_CHECK; ++w) {
        // Without a partial word, with one element, and with the most elements
        // that still leave the word partial (for FP64 there is none).
        const size_t tails[3] = {0, 1, lanes - 1};
        for (unsigned t = 0; t < 3; ++t) {
            if (tails[t] >= lanes) continue;
            if (t == 2 && (tails[2] == tails[1] || tails[2] == 0)) continue;
            for (unsigned log2 = 0; log2 < 4; log2 += 3)
                errors += check_one<P, M>(name, a, b, scratch, res,
                                          check_words[w] * lanes + tails[t],
                                          log2);
        }
    }
    return errors;
}

int main() {
    uint8_t* a = ALLOC(uint8_t, MAXW * 8);
    uint8_t* b = ALLOC(uint8_t, MAXW * 8);
    void* scratch =
        snrt_l1_alloc_cluster_local(snrt_vec_reduce_scratch_bytes(), 8);
    uint8_t* res = ALLOC(uint8_t, SLOT);
    unsigned* core_errors = ALLOC(unsigned, snrt_cluster_core_num());

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    const int compute = snrt_is_compute_core();

    // The vectors start as all zeros (integer stores, by the compute cores).
    if (compute) {
        volatile uint64_t* pa = (volatile uint64_t*)a;
        volatile uint64_t* pb = (volatile uint64_t*)b;
        for (unsigned i = core; i < MAXW; i += cores) {
            pa[i] = 0;
            pb[i] = 0;
        }
    }
    snrt_cluster_hw_barrier();

    unsigned errors = 0;
    errors += check_config<FP64, SNRT_ACC_SAME>("FP64", a, b, scratch, res);
    errors += check_config<FP32, SNRT_ACC_SAME>("FP32", a, b, scratch, res);
    errors += check_config<FP16, SNRT_ACC_SAME>("FP16", a, b, scratch, res);
    errors +=
        check_config<FP16, SNRT_ACC_WIDE>("FP16 wide", a, b, scratch, res);
    errors += check_config<FP8, SNRT_ACC_SAME>("FP8", a, b, scratch, res);
    errors += check_config<FP8, SNRT_ACC_WIDE>("FP8 wide", a, b, scratch, res);

    ((volatile unsigned*)core_errors)[core] = errors;
    snrt_cluster_hw_barrier();
    unsigned total = 0;
    for (unsigned c = 0; c < snrt_cluster_core_num(); ++c)
        total += ((volatile unsigned*)core_errors)[c];
    if (core == 0)
        printf("Dot product on 1 and 8 cores, interleaved split: %s (%u "
               "errors)\n",
               total ? "FAIL" : "PASS", total);

    return (core == 0 && total) ? 1 : 0;
}
