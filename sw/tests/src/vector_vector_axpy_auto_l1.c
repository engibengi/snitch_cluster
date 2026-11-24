// Automatic one-core / eight-core axpy (snrt_vv_axpy_auto_l1 and
// snrt_vv_axpy_padded_auto_l1) against the eight-core function, on both sides of
// the threshold, on data that needs padding and data that is already padded.
// See auto_test_common.h. alpha is the first element of b, y is b one word on.

#define TEST_NAME "axpy"
#define TEST_SKIP_FP8 0
#define TEST_CLASS SNRT_VEC_COST_AXPY
#define Y(b) ((const uint8_t*)(b) + 8)
#define G_MULTI(a, b, c, n, p) snrt_vv_axpy_l1(b, a, Y(b), c, n, p)
#define A_AUTO(a, b, c, n, p) snrt_vv_axpy_auto_l1(b, a, Y(b), c, n, p)
#define PA_AUTO(a, b, c, w, p) snrt_vv_axpy_padded_auto_l1(b, a, Y(b), c, w, p)
#include "auto_test_common.h"
