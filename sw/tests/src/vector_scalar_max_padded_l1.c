// Caller-padded vector-scalar max (snrt_vs_max_padded_core_l1 and
// snrt_vs_max_padded_l1) against the general functions, on data that needs
// padding and data that is already padded. See padded_test_common.h.

#define TEST_NAME "vs max"
#define TEST_SKIP_FP8 0
#define G_CORE(a, b, c, n, p) snrt_vs_max_core_l1(a, b, c, n, p)
#define G_MULTI(a, b, c, n, p) snrt_vs_max_l1(a, b, c, n, p)
#define P_CORE(a, b, c, w, p) snrt_vs_max_padded_core_l1(a, b, c, w, p)
#define P_MULTI(a, b, c, w, p) snrt_vs_max_padded_l1(a, b, c, w, p)
#include "padded_test_common.h"
