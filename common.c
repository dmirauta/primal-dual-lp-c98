#include "common.h"

FPN FPN_abs(FPN a) { return a > 0 ? a : -a; };

#ifdef ON_CPU
void print_vec(FPN *vec, IDX len) {
  for (IDX k = 0; k < len; k++) {
    printf("%05.4lf ", vec[k]);
  }
}
#endif /* ifdef ON_CPU */
