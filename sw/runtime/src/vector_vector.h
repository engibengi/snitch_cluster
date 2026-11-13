/**
 * @file
 * @brief Vector-vector operations on vectors anywhere in memory (L3 or L1).
 * @details The functions in this file, named without a suffix, are the default
 *          ones: the caller passes pointers to the vectors wherever they are,
 *          and the function takes care of bringing them into L1 in tiles, with
 *          double buffering. They are built on the _l1 functions of
 *          vector_vector_l1.h, which work on data already in L1 and are
 *          available to the caller too.
 * @note The default (L3) versions are not written yet; for now this header only
 *       provides the _l1 functions.
 */

#pragma once

#include "vector_vector_l1.h"
