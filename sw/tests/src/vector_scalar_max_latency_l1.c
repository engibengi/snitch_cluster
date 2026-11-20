// Cycle counts of vector max scalar on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector max scalar"
#define LAT_FAMILY OP_VS
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fmax.d"
#define LAT_I32 "vfmax.r.s"
#define LAT_I16 "vfmax.r.h"
#define LAT_I8 "vfmax.r.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vs_max_core_l1(a, s, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vs_max_l1(a, s, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vs_max_padded_core_l1(a, s, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vs_max_padded_l1(a, s, c, n, p)
#include "latency_elementwise_common.h"
