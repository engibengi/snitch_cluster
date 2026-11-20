// Cycle counts of vector sub scalar on data in L1, all precisions, all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector sub scalar"
#define LAT_FAMILY OP_VS
#define LAT_SKIP_FP8 0
#define LAT_MAX_WORDS 2048
#define LAT_I64 "fsub.d"
#define LAT_I32 "vfsub.r.s"
#define LAT_I16 "vfsub.r.h"
#define LAT_I8 "vfsub.r.b"
#define LAT_CORE(a, b, c, s, n, p) snrt_vs_sub_core_l1(a, s, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vs_sub_l1(a, s, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vs_sub_padded_core_l1(a, s, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vs_sub_padded_l1(a, s, c, n, p)
#include "latency_elementwise_common.h"
