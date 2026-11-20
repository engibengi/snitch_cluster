// Cycle counts of the element-wise operations on data in L1 (not a pass/fail
// test): vector + vector (add, as the representative of add, sub, mul, div,
// min, max), vector + scalar (add) and axpy (z = alpha * x + y), for every
// precision. See paper_notes/PENDING.md ("vs, vv and axpy").
//
// It is the measurement that comes before rebuilding these functions, as it was
// done for the dot product (vector_dot_latency_l1.c): how much of a call is
// fixed cost, how much of it is the generic structure of the library, and what
// the split of the words among the cores costs.
//
// Columns, for every operation and precision, per length (in 64-bit words, so
// a row is the same amount of data in every precision):
//   tail:    elements of a partial last word (the library handles it, the hand
//            functions are for whole words only: marked -);
//   lib1:    the library function on one core (timed on core 0), precision a
//            constant;
//   libR:    the same, with the precision a run-time value;
//   hand1:   the same computation written here as one function, precision and
//            operation fixed (whole words only, no tail code): what the
//            library's generic structure costs is lib1 - hand1;
//   lib8:    the library function on the 8 compute cores (contiguous slices,
//            the library's split), plus a barrier after the call (the library
//            functions have none, so this is the time until all cores are
//            done);
//   hand8c:  hand-written, 8 cores, contiguous slices (shifts, no division);
//   hand8i:  hand-written, 8 cores, interleaved (core c takes the words c,
//            c + 8, ...: a stream with a stride of 8 words), the split that
//            removed the bank conflicts of the dot product.
// A call is a separate non-inlined function in all columns, so the call
// overhead is the same in all of them. Each cell is the mean of REPS calls
// after one warm-up call. The data is not initialized: timing does not depend
// on it. The three vectors are allocated one after the other, 2064 words apart
// (16 modulo 32 banks, the distance that has no bank conflicts between
// streams); the program prints the distances it actually got.

#include "snrt.h"
#include "vector_vector_l1.h"
#include "vector_scalar_l1.h"
#include <stdint.h>
#include <stdio.h>

#define MAXW 2048
#define SPACING (MAXW + 16)  // words between the vectors: 16 modulo 32
#define REPS 4

typedef enum { OP_VV, OP_VS, OP_AXPY } op_t;

//------------------------------------------------------------------------------
// Hand-written loops: precision and operation constant, whole words only.
//------------------------------------------------------------------------------

// ft2 = op(ft0, ft1): the two sources stream in, the result streams out.
#define HAND_VV_ASM(instruction)                           \
    asm volatile(                                          \
        "frep.o %[n_frep], 1, 0, 0 \n"                     \
        instruction " ft2, ft0, ft1 \n"                    \
        :                                                  \
        : [ n_frep ] "r"(count - 1)                        \
        : "ft0", "ft1", "ft2", "memory")

// ft1 = op(ft0, scalar): one source streams in, the result streams out.
#define HAND_VS_ASM(instruction)                           \
    asm volatile(                                          \
        "frep.o %[n_frep], 1, 0, 0 \n"                     \
        instruction " ft1, ft0, %[s] \n"                   \
        :                                                  \
        : [ n_frep ] "r"(count - 1), [ s ] "f"(scalar)     \
        : "ft0", "ft1", "ft2", "memory")

// axpy: ft2 = ft0 * scalar + ft1. FP64: one fused instruction per word.
#define HAND_AXPY_FP64_ASM()                                       \
    asm volatile(                                                  \
        "frep.o %[n_frep], 1, 0, 0 \n"                             \
        "fmadd.d ft2, ft0, %[s], ft1 \n"                           \
        :                                                          \
        : [ n_frep ] "r"(count - 1), [ s ] "f"(scalar)             \
        : "ft0", "ft1", "ft2", "memory")

// axpy, packed: four words per iteration (multiply into ft3 ... ft6, then add).
#define HAND_AXPY_GROUPS_ASM(mul, add)                             \
    asm volatile(                                                  \
        "frep.o %[n_frep], 8, 0, 0 \n"                             \
        mul " ft3, ft0, %[s] \n"                                   \
        mul " ft4, ft0, %[s] \n"                                   \
        mul " ft5, ft0, %[s] \n"                                   \
        mul " ft6, ft0, %[s] \n"                                   \
        add " ft2, ft3, ft1 \n"                                    \
        add " ft2, ft4, ft1 \n"                                    \
        add " ft2, ft5, ft1 \n"                                    \
        add " ft2, ft6, ft1 \n"                                    \
        :                                                          \
        : [ n_frep ] "r"(groups - 1), [ s ] "f"(scalar)            \
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "memory")

#define HAND_AXPY_REST_ASM(mul, add)                               \
    asm volatile(                                                  \
        "frep.o %[n_frep], 2, 0, 0 \n"                             \
        mul " ft3, ft0, %[s] \n"                                   \
        add " ft2, ft3, ft1 \n"                                    \
        :                                                          \
        : [ n_frep ] "r"(rem - 1), [ s ] "f"(scalar)               \
        : "ft0", "ft1", "ft2", "ft3", "memory")

// The streams: `count` words, `stride` bytes apart, on the SSR data movers. Two
// sources and one destination (vv, axpy), or one source and one destination
// (vs).
SNRT_VEC_INLINE void streams_3(const void* x, const void* y, void* z,
                               size_t count, size_t stride) {
    snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, count, stride);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)x);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)y);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, z);
    snrt_ssr_enable();
}

SNRT_VEC_INLINE void streams_2(const void* x, void* z, size_t count,
                               size_t stride) {
    snrt_ssr_loop_1d(SNRT_SSR_DM0, count, stride);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, count, stride);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)x);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, z);
    snrt_ssr_enable();
}

// The work of one core: `count` words (0 = nothing), `stride` bytes apart.
// a, b: the sources (vs: b unused), c: the destination, scalar: the scalar.
template <precision_t P, op_t OP>
SNRT_VEC_INLINE void hand_core(const uint8_t* a, const uint8_t* b, uint8_t* c,
                               double scalar, size_t count, size_t stride) {
    if (count == 0) return;
    if (OP == OP_VV) {
        streams_3(a, b, c, count, stride);
        if (P == FP64) HAND_VV_ASM("fadd.d");
        else if (P == FP32) HAND_VV_ASM("vfadd.s");
        else if (P == FP16) HAND_VV_ASM("vfadd.h");
        else HAND_VV_ASM("vfadd.b");
    } else if (OP == OP_VS) {
        streams_2(a, c, count, stride);
        if (P == FP64) HAND_VS_ASM("fadd.d");
        else if (P == FP32) HAND_VS_ASM("vfadd.r.s");
        else if (P == FP16) HAND_VS_ASM("vfadd.r.h");
        else HAND_VS_ASM("vfadd.r.b");
    } else {
        streams_3(a, b, c, count, stride);
        if (P == FP64) {
            HAND_AXPY_FP64_ASM();
        } else {
            const size_t groups = count / 4;
            const size_t rem = count % 4;
            if (P == FP32) {
                if (groups > 0) HAND_AXPY_GROUPS_ASM("vfmul.r.s", "vfadd.s");
                if (rem > 0) HAND_AXPY_REST_ASM("vfmul.r.s", "vfadd.s");
            } else if (P == FP16) {
                if (groups > 0) HAND_AXPY_GROUPS_ASM("vfmul.r.h", "vfadd.h");
                if (rem > 0) HAND_AXPY_REST_ASM("vfmul.r.h", "vfadd.h");
            } else {
                if (groups > 0) HAND_AXPY_GROUPS_ASM("vfmul.r.b", "vfadd.b");
                if (rem > 0) HAND_AXPY_REST_ASM("vfmul.r.b", "vfadd.b");
            }
        }
    }
    snrt_vec_stream_end();
}

// One core, whole words, contiguous (the same call shape as the library's
// single-core function: a separate function with the scalar loaded inside).
template <precision_t P, op_t OP>
__attribute__((noinline)) static void hand1(const uint8_t* a, const uint8_t* b,
                                            uint8_t* c, const void* s,
                                            size_t words) {
    const double scalar = OP == OP_VV ? 0.0 : snrt_vs_load(s, P);
    hand_core<P, OP>(a, b, c, scalar, words, sizeof(uint64_t));
}

// 8 cores, contiguous slices (shifts, no division).
template <precision_t P, op_t OP>
__attribute__((noinline)) static void hand8c(const uint8_t* a, const uint8_t* b,
                                             uint8_t* c, const void* s,
                                             size_t words) {
    if (!snrt_vec_is_compute_core()) return;
    const double scalar = OP == OP_VV ? 0.0 : snrt_vs_load(s, P);
    size_t first, count;
    snrt_vec_split_log2(words, snrt_vec_core_idx(), 3, &first, &count);
    const size_t off = first * sizeof(uint64_t);
    hand_core<P, OP>(a + off, b + off, c + off, scalar, count,
                     sizeof(uint64_t));
}

// 8 cores, interleaved: core k takes the words k, k + 8, ...
template <precision_t P, op_t OP>
__attribute__((noinline)) static void hand8i(const uint8_t* a, const uint8_t* b,
                                             uint8_t* c, const void* s,
                                             size_t words) {
    if (!snrt_vec_is_compute_core()) return;
    const double scalar = OP == OP_VV ? 0.0 : snrt_vs_load(s, P);
    const size_t core = snrt_vec_core_idx();
    const size_t count = words > core ? (words - core + 7) >> 3 : 0;
    const size_t off = core * sizeof(uint64_t);
    hand_core<P, OP>(a + off, b + off, c + off, scalar, count,
                     8 * sizeof(uint64_t));
}

//------------------------------------------------------------------------------
// The library, called through the same kind of separate function.
//------------------------------------------------------------------------------

// n elements; `s` is the scalar (vs: the scalar, axpy: alpha).
template <precision_t P, op_t OP>
__attribute__((noinline)) static void lib1(const uint8_t* a, const uint8_t* b,
                                           uint8_t* c, const void* s,
                                           size_t n) {
    if (OP == OP_VV) snrt_vv_add_core_l1(a, b, c, n, P);
    else if (OP == OP_VS) snrt_vs_add_core_l1(a, s, c, n, P);
    else snrt_vv_axpy_core_l1(s, a, b, c, n, P);
}

// The same, the precision known only at run time.
template <op_t OP>
__attribute__((noinline)) static void libR(const uint8_t* a, const uint8_t* b,
                                           uint8_t* c, const void* s, size_t n,
                                           precision_t p) {
    if (OP == OP_VV) snrt_vv_add_core_l1(a, b, c, n, p);
    else if (OP == OP_VS) snrt_vs_add_core_l1(a, s, c, n, p);
    else snrt_vv_axpy_core_l1(s, a, b, c, n, p);
}

template <precision_t P, op_t OP>
__attribute__((noinline)) static void lib8(const uint8_t* a, const uint8_t* b,
                                           uint8_t* c, const void* s,
                                           size_t n) {
    if (OP == OP_VV) snrt_vv_add_l1(a, b, c, n, P);
    else if (OP == OP_VS) snrt_vs_add_l1(a, s, c, n, P);
    else snrt_vv_axpy_l1(s, a, b, c, n, P);
}

//------------------------------------------------------------------------------
// Timing
//------------------------------------------------------------------------------

// Mean cycles of `reps` calls of F on core 0 alone (the other cores do not run
// it; they wait at the barrier that the caller places after).
template <typename F>
__attribute__((noinline)) static unsigned time_single(F f) {
    f();  // warm-up
    const uint32_t t0 = snrt_mcycle();
    for (int r = 0; r < REPS; ++r) f();
    const uint32_t t1 = snrt_mcycle();
    return (t1 - t0 + REPS / 2) / REPS;
}

// The same with every core calling F, and a barrier after each call.
template <typename F>
__attribute__((noinline)) static unsigned time_multi(F f) {
    f();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    const uint32_t t0 = snrt_mcycle();
    for (int r = 0; r < REPS; ++r) {
        f();
        snrt_cluster_hw_barrier();
    }
    const uint32_t t1 = snrt_mcycle();
    return (t1 - t0 + REPS / 2) / REPS;
}

typedef struct {
    unsigned words;
    unsigned tail_kind;  // 0 none, 1 one element
} Row;

static const Row rows_list[] = {
    {1, 0},  {2, 0},   {4, 0},   {8, 0},    {16, 0},   {32, 0},   {64, 0},
    {256, 0}, {1024, 0}, {2048, 0}, {16, 1}, {1024, 1}};
#define NROWS (sizeof(rows_list) / sizeof(rows_list[0]))

template <precision_t P, op_t OP>
static void table(const char* pname, const char* opname, const uint8_t* a,
                  const uint8_t* b, uint8_t* c, const void* s) {
    const size_t lanes = 8 / (size_t)P;
    const int core0 = snrt_cluster_core_idx() == 0;
    if (core0)
        printf("%s %s\n  words tail   lib1   libR  hand1 |   lib8 hand8c "
               "hand8i\n",
               pname, opname);
    for (unsigned r = 0; r < NROWS; ++r) {
        const unsigned words = rows_list[r].words;
        const unsigned tail = rows_list[r].tail_kind;
        if (tail && lanes == 1) continue;  // FP64 has no partial word
        const size_t n = words * lanes + tail;
        volatile precision_t pv = P;  // not a constant to the compiler
        const precision_t prun = pv;

        unsigned t_lib1 = 0, t_libR = 0, t_hand1 = 0;
        if (core0) {
            t_lib1 = time_single([&] { lib1<P, OP>(a, b, c, s, n); });
            t_libR = time_single([&] { libR<OP>(a, b, c, s, n, prun); });
            if (!tail)
                t_hand1 = time_single([&] { hand1<P, OP>(a, b, c, s, words); });
        }
        snrt_cluster_hw_barrier();
        const unsigned t_lib8 =
            time_multi([&] { lib8<P, OP>(a, b, c, s, n); });
        unsigned t_h8c = 0, t_h8i = 0;
        if (!tail) {
            t_h8c = time_multi([&] { hand8c<P, OP>(a, b, c, s, words); });
            t_h8i = time_multi([&] { hand8i<P, OP>(a, b, c, s, words); });
        }
        if (core0) {
            printf("  %5u %4u  %5u  %5u ", words, tail, t_lib1, t_libR);
            if (tail) printf("     - |  %6u      -      -\n", t_lib8);
            else
                printf(" %5u |  %6u %6u %6u\n", t_hand1, t_lib8, t_h8c, t_h8i);
        }
    }
    if (core0) printf("\n");
}

template <precision_t P>
static void all_ops(const char* pname, const uint8_t* a, const uint8_t* b,
                    uint8_t* c, const void* s) {
    table<P, OP_VV>(pname, "vector + vector (add)", a, b, c, s);
    table<P, OP_VS>(pname, "vector + scalar (add)", a, b, c, s);
    table<P, OP_AXPY>(pname, "axpy", a, b, c, s);
}

int main() {
    // Three vectors SPACING words apart, then the scalar.
    uint8_t* a = (uint8_t*)snrt_l1_alloc_cluster_local(SPACING * 8, 8);
    uint8_t* b = (uint8_t*)snrt_l1_alloc_cluster_local(SPACING * 8, 8);
    uint8_t* c = (uint8_t*)snrt_l1_alloc_cluster_local(SPACING * 8, 8);
    uint8_t* s = (uint8_t*)snrt_l1_alloc_cluster_local(16, 8);

    const unsigned core = snrt_cluster_core_idx();
    const unsigned cores = snrt_cluster_compute_core_num();
    if (snrt_is_compute_core()) {
        volatile uint64_t* pa = (volatile uint64_t*)a;
        volatile uint64_t* pb = (volatile uint64_t*)b;
        for (unsigned i = core; i < SPACING; i += cores) {
            pa[i] = 0;
            pb[i] = 0;
        }
        if (core == 0) {
            ((volatile uint64_t*)s)[0] = 0;
            ((volatile uint64_t*)s)[1] = 0;
        }
    }
    snrt_cluster_hw_barrier();

    if (core == 0) {
        // The real distances (the allocator could add padding between buffers).
        const int ab = (int)(((intptr_t)b - (intptr_t)a) / 8);
        const int bc = (int)(((intptr_t)c - (intptr_t)b) / 8);
        printf("Element-wise operations on L1 data: cycles per call "
               "(mean of %d)\na at 0x%x (bank %u), b - a = %d words (mod 32: "
               "%d), c - b = %d words (mod 32: %d)\n\n",
               REPS, (unsigned)(uintptr_t)a,
               (unsigned)(((uintptr_t)a >> 3) & 31), ab, ab & 31, bc, bc & 31);
    }
    all_ops<FP64>("FP64", a, b, c, s);
    all_ops<FP32>("FP32", a, b, c, s);
    all_ops<FP16>("FP16", a, b, c, s);
    all_ops<FP8>("FP8", a, b, c, s);
    return 0;
}
