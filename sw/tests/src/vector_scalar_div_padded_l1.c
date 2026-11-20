// Caller-padded vector-scalar div (snrt_vs_div_padded_core_l1 and
// snrt_vs_div_padded_l1) against the general functions, on data that needs
// padding and data that is already padded. See padded_test_common.h.

#define TEST_NAME "vs div"
#define TEST_SKIP_FP8 1
#define G_CORE(a, b, c, n, p) snrt_vs_div_core_l1(a, b, c, n, p)
#define G_MULTI(a, b, c, n, p) snrt_vs_div_l1(a, b, c, n, p)
#define P_CORE(a, b, c, w, p) snrt_vs_div_padded_core_l1(a, b, c, w, p)
#define P_MULTI(a, b, c, w, p) snrt_vs_div_padded_l1(a, b, c, w, p)
#include "padded_test_common.h"
