#include "gauss_jordan.h"

int main() {
  FPN data[] = {2.0, 3.0, 5.0, 4.0, 1.0, 9.0, 7.0, 6.0, 8.0, 2.0,
                7.0, 9.0, 1.0, 8.0, 3.0, 1.0, 5.0, 1.0, 2.0, 3.0};
  GJTab_t tab = {&data, 4};
  // GJTab_print(&mat);

  IdxStack_t pivots = {malloc(sizeof(IDX) * tab.N),
                       malloc(sizeof(BYTE) * tab.N), tab.N, 0};
  FPN sol[4] = {0};
  gauss_jordan(&tab, &pivots, sol);

  printf("Got: ");
  for (IDX k = 0; k < tab.N; k++) {
    printf("%05.4lf ", sol[k]);
  }
  printf("\nExpected: -0.05128205  0.76923077  0.12820513 -0.46153846\n");
}
