// Caller-padded axpy (snrt_vv_axpy_padded_core_l1 and snrt_vv_axpy_padded_l1)
// against the general functions, on data that needs padding and data that is
// already padded. See padded_test_common.h. The scalar alpha is the first
// element of b, and y is b one word further.

#define TEST_NAME "axpy"
#define TEST_SKIP_FP8 0
#define Y(b) ((const uint8_t*)(b) + 8)
#define G_CORE(a, b, c, n, p) snrt_vv_axpy_core_l1(b, a, Y(b), c, n, p)
#define G_MULTI(a, b, c, n, p) snrt_vv_axpy_l1(b, a, Y(b), c, n, p)
#define P_CORE(a, b, c, w, p) snrt_vv_axpy_padded_core_l1(b, a, Y(b), c, w, p)
#define P_MULTI(a, b, c, w, p) snrt_vv_axpy_padded_l1(b, a, Y(b), c, w, p)
#include "padded_test_common.h"
