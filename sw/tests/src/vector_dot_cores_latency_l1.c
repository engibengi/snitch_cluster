// Cycles of the dot product on 1 and 8 cores, with the interleaved split
// (snrt_vec_dot_cores_l1) and with the padded function
// (snrt_vec_dot_padded_cores_l1), for every precision (timing only, not a
// pass/fail test; the correctness test is vector_dot_cores_l1.c). See
// paper_notes/BANK_CONFLICTS.md section 6d.
//
// Constant precision and mode, data in L1, a warm-up call, the mean of REPS
// calls on core 0, all cores calling. Each table has:
//  - rows of a whole number of words (length in 64-bit words): the general
//    function has no partial word there and only carries the tail code unused,
//    so the padded function's gain on these rows is the cost of that code;
//  - rows with a partial last word (words + 1 element, and words + the most
//    elements that still leave the word partial): the general function gets
//    the vector as it is, the padded function gets it padded by the caller to
//    one more word (words + 1). These are the rows that show what padding
//    saves for a vector that needs it.
// The data is not initialized: the values do not change the timing.

#include "snrt.h"
#include "vector_reduce_l1.h"
#include <stdint.h>
#include <stdio.h>

// Words of the longest vector: 2096, plus one word for the padded vector with
// a partial word.
#define MAXW 2097
#define REPS 8

#define ALLOC(type, count) \
    ((type*)snrt_l1_alloc_cluster_local((count) * sizeof(type), 8))

// Lengths of the timing table, in words.
static const unsigned time_words[] = {
    8,   12,  16,  24,  32,  40,   48,   64,   75,   96,   128, 160,
    192, 256, 333, 384, 512, 700, 768, 1024, 1536, 2048, 2096};
#define N_TIME (sizeof(time_words) / sizeof(time_words[0]))

// Cycles of one call, mean of REPS, as seen by core 0.
template <precision_t P, snrt_acc_t M>
__attribute__((noinline)) static unsigned time_call(const uint8_t* a,
                                                    const uint8_t* b,
                                                    void* scratch, uint8_t* res,
                                                    size_t n, unsigned log2) {
    snrt_vec_dot_cores_l1(a, b, res, scratch, n, P, M, log2);  // warm-up
    snrt_cluster_hw_barrier();
    const uint32_t t0 = snrt_mcycle();
    for (int r = 0; r < REPS; ++r)
        snrt_vec_dot_cores_l1(a, b, res, scratch, n, P, M, log2);
    const uint32_t t1 = snrt_mcycle();
    return (t1 - t0 + REPS / 2) / REPS;
}

// The same for whole words (no partial word): the length is in words.
template <precision_t P, snrt_acc_t M>
__attribute__((noinline)) static unsigned time_call_padded(
    const uint8_t* a, const uint8_t* b, void* scratch, uint8_t* res,
    size_t words, unsigned log2) {
    snrt_vec_dot_padded_cores_l1(a, b, res, scratch, words, P, M, log2);
    snrt_cluster_hw_barrier();
    const uint32_t t0 = snrt_mcycle();
    for (int r = 0; r < REPS; ++r)
        snrt_vec_dot_padded_cores_l1(a, b, res, scratch, words, P, M, log2);
    const uint32_t t1 = snrt_mcycle();
    return (t1 - t0 + REPS / 2) / REPS;
}

template <precision_t P, snrt_acc_t M>
static void time_config(const char* name, uint8_t* a, uint8_t* b,
                        void* scratch, uint8_t* res) {
    const size_t lanes = 8 / (size_t)P;
    const int core0 = snrt_cluster_core_idx() == 0;
    if (core0) {
        printf("%s\n", name);
        printf("  words   1 core   8 cores   1 padded  8 padded\n");
    }
    for (unsigned w = 0; w < N_TIME; ++w) {
        const unsigned t1 =
            time_call<P, M>(a, b, scratch, res, time_words[w] * lanes, 0);
        const unsigned t8 =
            time_call<P, M>(a, b, scratch, res, time_words[w] * lanes, 3);
        // Whole words, through the function for whole words.
        const unsigned p1 =
            time_call_padded<P, M>(a, b, scratch, res, time_words[w], 0);
        const unsigned p8 =
            time_call_padded<P, M>(a, b, scratch, res, time_words[w], 3);
        if (core0)
            printf("  %5u  %7u   %7u   %8u  %8u\n", time_words[w], t1, t8, p1,
                   p8);
    }
    // Vectors with a partial last word: the general function gets `words`
    // words plus 1 element, and plus the most elements that still leave the
    // word partial; the padded function gets the vector as the caller would
    // pad it, one word more (`words + 1`). The rows above have no partial
    // word, so the general function only carries the (unused) tail code there.
    if (lanes > 1) {
        if (core0)
            printf("  with a partial word (words+elements; padded: words+1)\n");
        static const unsigned tail_words[2] = {16, 1024};
        for (unsigned w = 0; w < 2; ++w)
            for (unsigned k = 0; k < 2; ++k) {
                const size_t extra = k == 0 ? 1 : lanes - 1;
                if (k == 1 && extra == 1) continue;  // FP32: same as k = 0
                const size_t n = tail_words[w] * lanes + extra;
                const unsigned t1 = time_call<P, M>(a, b, scratch, res, n, 0);
                const unsigned t8 = time_call<P, M>(a, b, scratch, res, n, 3);
                const unsigned p1 = time_call_padded<P, M>(
                    a, b, scratch, res, tail_words[w] + 1, 0);
                const unsigned p8 = time_call_padded<P, M>(
                    a, b, scratch, res, tail_words[w] + 1, 3);
                if (core0)
                    printf("  %4u+%ue %6u   %7u   %8u  %8u\n", tail_words[w],
                           (unsigned)extra, t1, t8, p1, p8);
            }
    }
    if (core0) printf("\n");
}

int main() {
    uint8_t* a = ALLOC(uint8_t, MAXW * 8);
    uint8_t* b = ALLOC(uint8_t, MAXW * 8);
    void* scratch =
        snrt_l1_alloc_cluster_local(snrt_vec_reduce_scratch_bytes(), 8);
    uint8_t* res = ALLOC(uint8_t, 16);

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();

    if (snrt_is_compute_core()) {
        volatile uint64_t* pa = (volatile uint64_t*)a;
        volatile uint64_t* pb = (volatile uint64_t*)b;
        for (unsigned i = core; i < MAXW; i += cores) {
            pa[i] = 0;
            pb[i] = 0;
        }
    }
    snrt_cluster_hw_barrier();

    if (core == 0) {
        printf("Cycles of one call, all cores calling (timed on core 0), "
               "length in 64-bit words\n");
        // The distance between the vectors changes the bank conflicts (see
        // BANK_CONFLICTS.md): print it, in words and modulo 32 (the banks).
        printf("a at 0x%x (bank %u), b at 0x%x (bank %u), b - a = %d words "
               "(mod 32: %d)\n\n",
               (unsigned)(uintptr_t)a, (unsigned)(((uintptr_t)a >> 3) & 31),
               (unsigned)(uintptr_t)b, (unsigned)(((uintptr_t)b >> 3) & 31),
               (int)(((intptr_t)b - (intptr_t)a) / 8),
               (int)((((intptr_t)b - (intptr_t)a) / 8) & 31));
    }
    time_config<FP64, SNRT_ACC_SAME>("FP64", a, b, scratch, res);
    time_config<FP32, SNRT_ACC_SAME>("FP32", a, b, scratch, res);
    time_config<FP16, SNRT_ACC_SAME>("FP16", a, b, scratch, res);
    time_config<FP16, SNRT_ACC_WIDE>("FP16 wide", a, b, scratch, res);
    time_config<FP8, SNRT_ACC_SAME>("FP8", a, b, scratch, res);
    time_config<FP8, SNRT_ACC_WIDE>("FP8 wide", a, b, scratch, res);
    return 0;
}
