// Layout (bank conflict) sweep of axpy (library, eight cores, 2048 words): b - a
// and c - a modulo 32 in {0, 1, 2, 8, 16, 17, 24, 31}, plus c = a and c = b
// (z = y, the usual axpy, in place on the second input), for FP64 (one fused
// instruction per word), FP32 and FP16 (two instructions per word: compute
// bound, expected without a layout effect, as FP8 was). Not a pass/fail test.
// See layout_latency_common.h and paper_notes/ELEMENTWISE_LATENCY.md section 7.

#include "layout_latency_common.h"

static const unsigned D[8] = {0, 1, 2, 8, 16, 17, 24, 31};

int main() {
    const void* alpha;
    uint8_t* arena = layout_setup("axpy, layout sweep", &alpha);
    grid<FP64, F_AXPY>("FP64", "axpy (library)", arena, alpha, 2048, D, 8, D, 8,
                       true);
    grid<FP32, F_AXPY>("FP32", "axpy (library)", arena, alpha, 2048, D, 8, D, 8,
                       true);
    grid<FP16, F_AXPY>("FP16", "axpy (library)", arena, alpha, 2048, D, 8, D, 8,
                       true);
    return 0;
}
