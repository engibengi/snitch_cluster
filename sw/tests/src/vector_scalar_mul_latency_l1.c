// Cycle counts of vector mul scalar on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector mul scalar"
#define LAT_FAMILY OP_VS
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fmul.d"
#define LAT_I32 "vfmul.r.s"
#define LAT_I16 "vfmul.r.h"
#define LAT_I8 "vfmul.r.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vs_mul_core_l1(a, s, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vs_mul_l1(a, s, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vs_mul_padded_core_l1(a, s, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vs_mul_padded_l1(a, s, c, n, p)
#include "latency_elementwise_common.h"
