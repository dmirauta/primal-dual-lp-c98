#include "common.h"

FPN FPN_abs(FPN a) { return a > 0 ? a : -a; };

void clamp(FPN *a) {
  BYTE is_pos = *a > 0;
  FPN abs_a = is_pos ? *a : -*a;
  if (abs_a < EPSILON) {
    *a = is_pos ? EPSILON : -EPSILON;
  }
}

FPN clamped(FPN a) {
  FPN out = a;
  clamp(&out);
  return out;
}

void print_vec(FPN *vec, IDX len) {
  for (IDX k = 0; k < len; k++) {
#ifdef USE_FLOAT
    printf("%05.4f ", vec[k]);
#else
    printf("%05.4lf ", vec[k]);
#endif /* ifdef USE_FLOAT */
  }
}
