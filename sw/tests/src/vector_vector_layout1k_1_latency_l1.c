// The same sweep as vector_vector_layout_1_latency_l1.c at 1000 words (not a
// multiple of 8: the cores have unequal numbers of words), to see whether the
// rule found at 2048 words holds at another length.
//
// Sweep of the layout of vv add (library, eight cores, FP64, 1000 words): every
// c - a = 0 ... 31 (mod 32), for b - a = 8 ... 15 (mod 32), plus c = a and c = b
// (in place on either input). The whole sweep is four tests (_0 ... _3, eight
// values of b - a each) so that they can run at the same time. Not a pass/fail
// test. See layout_latency_common.h and paper_notes/ELEMENTWISE_LATENCY.md
// section 7.

#include "layout_latency_common.h"

int main() {
    const void* alpha;
    uint8_t* arena = layout_setup(
        "vv add, layout sweep, 1000 words (b - a = 8 ... 15, all c - a)", &alpha);
    unsigned d1[8], d2[32];
    for (unsigned i = 0; i < 8; ++i) d1[i] = 8 + i;
    for (unsigned j = 0; j < 32; ++j) d2[j] = j;
    grid<FP64, F_VV>("FP64", "vv add (library)", arena, alpha, 1000, d1, 8, d2,
                     32, true);
    return 0;
}
