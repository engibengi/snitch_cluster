// Layout (bank conflict) sweep of vs add (library, eight cores, two streams:
// read a, write c; the vector b is not used): every c - a = 0 ... 31 (mod 32)
// and c = a, for FP64 and FP8, at 2048 and 1000 words. The vector-scalar
// functions were only measured at eight values of c - a; the full vv sweep
// suggests that the pair a - c alone never conflicts (rows b - a = 16, 17, 19
// were flat for every c - a); this confirms it for vs itself. Not a pass/fail
// test. See layout_latency_common.h and paper_notes/ELEMENTWISE_LATENCY.md
// section 7.

#include "layout_latency_common.h"

int main() {
    const void* alpha;
    uint8_t* arena = layout_setup("vs add, layout sweep", &alpha);
    unsigned d1[1] = {0}, d2[32];
    for (unsigned j = 0; j < 32; ++j) d2[j] = j;
    grid<FP64, F_VS>("FP64", "vs add (library)", arena, alpha, 2048, d1, 1, d2,
                     32, false);
    grid<FP64, F_VS>("FP64", "vs add (library)", arena, alpha, 1000, d1, 1, d2,
                     32, false);
    grid<FP8, F_VS>("FP8", "vs add (library)", arena, alpha, 2048, d1, 1, d2,
                    32, false);
    grid<FP8, F_VS>("FP8", "vs add (library)", arena, alpha, 1000, d1, 1, d2,
                    32, false);
    return 0;
}
