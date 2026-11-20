// Caller-padded vector-vector div (snrt_vv_div_padded_core_l1 and
// snrt_vv_div_padded_l1) against the general functions, on data that needs
// padding and data that is already padded. See padded_test_common.h.

#define TEST_NAME "vv div"
#define TEST_SKIP_FP8 1
#define G_CORE(a, b, c, n, p) snrt_vv_div_core_l1(a, b, c, n, p)
#define G_MULTI(a, b, c, n, p) snrt_vv_div_l1(a, b, c, n, p)
#define P_CORE(a, b, c, w, p) snrt_vv_div_padded_core_l1(a, b, c, w, p)
#define P_MULTI(a, b, c, w, p) snrt_vv_div_padded_l1(a, b, c, w, p)
#include "padded_test_common.h"
