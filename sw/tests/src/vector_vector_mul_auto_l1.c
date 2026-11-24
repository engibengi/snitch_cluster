// Automatic one-core / eight-core vector-vector mul (snrt_vv_mul_auto_l1 and
// snrt_vv_mul_padded_auto_l1) against the eight-core function, on both
// sides of the threshold, on data that needs padding and data that is already
// padded. See auto_test_common.h.

#define TEST_NAME "vv mul"
#define TEST_SKIP_FP8 0
#define TEST_CLASS SNRT_VEC_COST_SIMPLE
#define G_MULTI(a, b, c, n, p) snrt_vv_mul_l1(a, b, c, n, p)
#define A_AUTO(a, b, c, n, p) snrt_vv_mul_auto_l1(a, b, c, n, p)
#define PA_AUTO(a, b, c, w, p) snrt_vv_mul_padded_auto_l1(a, b, c, w, p)
#include "auto_test_common.h"
