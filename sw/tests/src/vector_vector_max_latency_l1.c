// Cycle counts of vector max vector on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector max vector"
#define LAT_FAMILY OP_VV
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fmax.d"
#define LAT_I32 "vfmax.s"
#define LAT_I16 "vfmax.h"
#define LAT_I8 "vfmax.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vv_max_core_l1(a, b, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vv_max_l1(a, b, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vv_max_padded_core_l1(a, b, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vv_max_padded_l1(a, b, c, n, p)
#include "latency_elementwise_common.h"
