// Cycle counts of axpy (z = alpha * x + y) on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "axpy (z = alpha * x + y)"
#define LAT_FAMILY OP_AXPY
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 ""
#define LAT_I32 ""
#define LAT_I16 ""
#define LAT_I8 ""
#define LAT_CORE(a, b, c, s, n, p) snrt_vv_axpy_core_l1(s, a, b, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vv_axpy_l1(s, a, b, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vv_axpy_padded_core_l1(s, a, b, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vv_axpy_padded_l1(s, a, b, c, n, p)
#include "latency_elementwise_common.h"
