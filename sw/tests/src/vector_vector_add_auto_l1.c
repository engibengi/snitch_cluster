// Automatic one-core / eight-core vector-vector add (snrt_vv_add_auto_l1 and
// snrt_vv_add_padded_auto_l1) against the eight-core function, on both
// sides of the threshold, on data that needs padding and data that is already
// padded. See auto_test_common.h.

#define TEST_NAME "vv add"
#define TEST_SKIP_FP8 0
#define TEST_CLASS SNRT_VEC_COST_SIMPLE
#define G_MULTI(a, b, c, n, p) snrt_vv_add_l1(a, b, c, n, p)
#define A_AUTO(a, b, c, n, p) snrt_vv_add_auto_l1(a, b, c, n, p)
#define PA_AUTO(a, b, c, w, p) snrt_vv_add_padded_auto_l1(a, b, c, w, p)
#include "auto_test_common.h"
