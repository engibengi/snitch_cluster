// Bank conflicts in the eight-core element-wise functions against the layout of
// the vectors (not a pass/fail test), 72 layouts: b - a and c - a modulo 32 in
// {0, 1, 2, 8, 16, 17, 24, 31}, plus c = a. FP64 and FP8; vv add (library),
// vv add with the contiguous split, axpy, and vs add (two streams, control);
// 256, 1000 and 2048 words. See layout_latency_common.h. Results:
// paper_notes/ELEMENTWISE_LATENCY.md section 7.

#include "layout_latency_common.h"

static const unsigned D[8] = {0, 1, 2, 8, 16, 17, 24, 31};

template <precision_t P>
static void all_functions(const char* pname, uint8_t* arena,
                          const void* alpha) {
    static const unsigned lengths[] = {256, 1000, 2048};
    for (unsigned l = 0; l < 3; ++l) {
        grid<P, F_VV>(pname, "vv add (library)", arena, alpha, lengths[l], D,
                      8, D, 8, false);
        grid<P, F_CONTIG>(pname, "vv add (contiguous split)", arena, alpha,
                          lengths[l], D, 8, D, 8, false);
        grid<P, F_AXPY>(pname, "axpy (library)", arena, alpha, lengths[l], D, 8,
                        D, 8, false);
        grid<P, F_VS>(pname, "vs add (library, control)", arena, alpha,
                      lengths[l], D, 8, D, 8, false);
    }
}

int main() {
    const void* alpha;
    uint8_t* arena = layout_setup(
        "Element-wise functions, eight cores, against the layout of the "
        "vectors",
        &alpha);
    all_functions<FP64>("FP64", arena, alpha);
    all_functions<FP8>("FP8", arena, alpha);
    return 0;
}
