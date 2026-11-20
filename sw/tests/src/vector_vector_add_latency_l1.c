// Cycle counts of vector add vector on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector add vector"
#define LAT_FAMILY OP_VV
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fadd.d"
#define LAT_I32 "vfadd.s"
#define LAT_I16 "vfadd.h"
#define LAT_I8 "vfadd.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vv_add_core_l1(a, b, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vv_add_l1(a, b, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vv_add_padded_core_l1(a, b, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vv_add_padded_l1(a, b, c, n, p)
#include "latency_elementwise_common.h"
