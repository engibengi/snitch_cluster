// Shared code of the layout (bank conflict) latency tests of the eight-core
// element-wise functions: vector_vector_distance_latency_l1.c,
// vector_vector_layout_<n>_latency_l1.c and vector_axpy_layout_latency_l1.c.
// See paper_notes/ELEMENTWISE_LATENCY.md section 7 and BANK_CONFLICTS.md.
//
// The L1 memory has 32 banks of one word (8 bytes): two streams that hit the
// same bank in the same cycle make one wait. Only the distances between the
// buffers modulo 32 words matter. The vectors are S = 2112 words apart (a
// multiple of 32) plus the distance d1 = b - a, d2 = c - a, and do not overlap.
// Each cell is the mean cycles of 4 calls on the eight compute cores, with a
// barrier after each call so that the cores start every call together
// (lockstep, where conflicts show), after one warm-up call.
// Functions: F_VV (vv add), F_AXPY, F_VS (two streams: a control, it does not
// use b), F_CONTIG (vv add with the contiguous split, the old one). All use
// the library's caller-padded functions on whole words, precision a constant.

#pragma once

#include "snrt.h"
#include "vector_vector_l1.h"
#include "vector_scalar_l1.h"
#include <stdint.h>
#include <stdio.h>

#define S 2112      // words between the vectors, a multiple of 32
#define MAXW 2048
#define REPS 4

typedef enum { F_VV, F_AXPY, F_VS, F_CONTIG } fn_t;

template <precision_t P, fn_t F>
__attribute__((noinline)) static void call(const uint8_t* a, const uint8_t* b,
                                           uint8_t* c, const void* alpha,
                                           size_t words) {
    if (F == F_VV) {
        snrt_vv_add_padded_l1(a, b, c, words, P);
    } else if (F == F_AXPY) {
        snrt_vv_axpy_padded_l1(alpha, a, b, c, words, P);
    } else if (F == F_VS) {
        snrt_vs_add_padded_l1(a, alpha, c, words, P);
    } else {
        // Contiguous split: the first words % 8 cores take one word more.
        if (!snrt_vec_is_compute_core()) return;
        size_t first, count;
        snrt_vec_split_log2(words, snrt_vec_core_idx(), 3, &first, &count);
        if (count > 0) {
            const size_t off = first * sizeof(uint64_t);
            snrt_vec_stream_begin_2in(a + off, b + off, c + off, count);
            snrt_vv_stream(SNRT_VV_ADD, P, count);
            snrt_vec_stream_end();
        }
        snrt_fpu_fence();
    }
}

// Mean cycles of REPS calls with all the cores calling, a barrier after each.
template <precision_t P, fn_t F>
__attribute__((noinline)) static unsigned time_call(const uint8_t* a,
                                                    const uint8_t* b,
                                                    uint8_t* c,
                                                    const void* alpha,
                                                    size_t words) {
    call<P, F>(a, b, c, alpha, words);
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    const uint32_t t0 = snrt_mcycle();
    for (int r = 0; r < REPS; ++r) {
        call<P, F>(a, b, c, alpha, words);
        snrt_cluster_hw_barrier();
    }
    const uint32_t t1 = snrt_mcycle();
    return (t1 - t0 + REPS / 2) / REPS;
}

// A grid: rows = b - a = d1[i] (mod 32), columns = c - a = d2[j] (mod 32), then
// c = a (in place on the first input) and, if with_cb, c = b (in place on the
// second input: z = y of axpy). Only the cells' distances modulo 32 matter.
template <precision_t P, fn_t F>
static void grid(const char* pname, const char* fname, uint8_t* arena,
                 const void* alpha, size_t words, const unsigned* d1,
                 unsigned n1, const unsigned* d2, unsigned n2, bool with_cb) {
    const int core0 = snrt_cluster_core_idx() == 0;
    uint8_t* a = arena;
    const unsigned rows = F == F_VS ? 1 : n1;  // vs does not use b
    if (core0) {
        printf("%s %s, %u words: cycles per call\n   b-a \\ c-a:", pname,
               fname, (unsigned)words);
        for (unsigned j = 0; j < n2; ++j) printf(" %4u", d2[j]);
        printf("  c=a%s\n", with_cb ? "  c=b" : "");
    }
    unsigned lo = ~0u, hi = 0;
    for (unsigned i = 0; i < rows; ++i) {
        uint8_t* b = arena + (S + d1[i]) * 8;
        if (core0) printf("      %2u      :", d1[i]);
        for (unsigned j = 0; j < n2 + 1 + (with_cb ? 1 : 0); ++j) {
            uint8_t* c = j < n2 ? arena + (2 * S + d2[j]) * 8
                                : (j == n2 ? a : b);
            const unsigned t = time_call<P, F>(a, b, c, alpha, words);
            if (core0) {
                printf(" %4u", t);
                if (t < lo) lo = t;
                if (t > hi) hi = t;
            }
        }
        if (core0) printf("\n");
    }
    if (core0) printf("   smallest %u, largest %u\n\n", lo, hi);
}

// The arena: a, then b and c at the distances above, then the scalar.
static uint8_t* layout_setup(const char* title, const void** alpha) {
    uint8_t* arena = (uint8_t*)snrt_l1_alloc_cluster_local((3 * S + 40) * 8, 8);
    *alpha = arena + (3 * S + 32) * 8;
    if (snrt_cluster_core_idx() == 0) {
        volatile uint64_t* p = (volatile uint64_t*)arena;
        for (unsigned i = 0; i < 3 * S + 40; ++i) p[i] = 0;
        printf("%s (arena at 0x%x, bank %u)\n\n", title,
               (unsigned)(uintptr_t)arena,
               (unsigned)(((uintptr_t)arena >> 3) & 31));
    }
    snrt_cluster_hw_barrier();
    return arena;
}
