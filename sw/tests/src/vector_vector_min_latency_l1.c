// Cycle counts of vector min vector on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector min vector"
#define LAT_FAMILY OP_VV
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fmin.d"
#define LAT_I32 "vfmin.s"
#define LAT_I16 "vfmin.h"
#define LAT_I8 "vfmin.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vv_min_core_l1(a, b, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vv_min_l1(a, b, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vv_min_padded_core_l1(a, b, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vv_min_padded_l1(a, b, c, n, p)
#include "latency_elementwise_common.h"
