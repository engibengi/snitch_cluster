// Cycle counts of vector div vector on data in L1, all precisions but FP8 (no FP8 division), all the
// functions: general and padded, one core and eight, run-time and constant
// precision, whole words and partial words (not a pass/fail test). See
// latency_elementwise_common.h for the columns.

#define LAT_NAME "vector div vector"
#define LAT_FAMILY OP_VV
#define LAT_SKIP_FP8 1
#define LAT_MAX_WORDS 256
#define LAT_I64 "fdiv.d"
#define LAT_I32 "vfdiv.s"
#define LAT_I16 "vfdiv.h"
#define LAT_I8 "vfdiv.h"
#define LAT_CORE(a, b, c, s, n, p) snrt_vv_div_core_l1(a, b, c, n, p)
#define LAT_MULTI(a, b, c, s, n, p) snrt_vv_div_l1(a, b, c, n, p)
#define LAT_PCORE(a, b, c, s, n, p) snrt_vv_div_padded_core_l1(a, b, c, n, p)
#define LAT_PMULTI(a, b, c, s, n, p) snrt_vv_div_padded_l1(a, b, c, n, p)
#include "latency_elementwise_common.h"
