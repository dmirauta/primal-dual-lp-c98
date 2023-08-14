#include "stack.h"
#include <stdio.h>

// Tableau (A|b) for Gauss-Jordan elimination (for Ax=b).
// Dynamic allocation friendly, but not actually needed,
// statically sized FPN mat[N][N+1] could be fine instead.
typedef struct GJTableau {
  FPN *ptr;
  IDX N;
} GJTab_t;

#ifdef ON_CPU
void GJTab_print(GJTab_t *tab) {
  printf("\n");
  for (IDX i = 0; i < tab->N; i++) {
    for (IDX j = 0; j < tab->N + 1; j++) {
      printf("%05.2lf ", tab->ptr[i * (tab->N + 1) + j]);
    }
    printf("\n");
  }
}
#endif

typedef struct MaxRes {
  FPN val;
  IDX arg;
} MaxRes_t;

MaxRes_t col_max(GJTab_t *tab, IDX j, IdxStack_t *no_max) {
  FPN val = NEG_INF;
  IDX arg = tab->N;
  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(no_max, i) &&
        (tab->ptr[i * (tab->N + 1) + j] > val)) {
      val = tab->ptr[i * (tab->N + 1) + j];
      arg = i;
    }
  }
  return (MaxRes_t){val, arg};
}

// while we are iterating columnwise, reads/writes zigzag between two columns?
// e.g.
// m11 m12 m13 ...
//  | / | / ...
// m21 m22 ...
// although one rows is only read, while the other only written
// is this efficient in terms of memory ops?
void sub_scaled_row(GJTab_t *tab, IDX from_col, IDX to_sub, IDX sub_from,
                    FPN q) {
  for (IDX j = from_col; j < tab->N + 1; j++) {
    tab->ptr[sub_from * (tab->N + 1) + j] -=
        q * tab->ptr[to_sub * (tab->N + 1) + j];
  }
}

void eliminate(GJTab_t *tab, IdxStack_t *pivots) {
  IDX current_pivot_j = pivots->size - 1;
  IDX current_pivot_i = pivots->ptr[pivots->size - 1];
  FPN q;
  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(pivots, i)) {
      q = tab->ptr[i * (tab->N + 1) + current_pivot_j] /
          tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
      sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);
    }
  }
}

void backsub_row(GJTab_t *tab, IdxStack_t *pivots) {
  IDX current_pivot_j = pivots->size - 1;
  IDX current_pivot_i = IdxStack_pop(pivots);
  FPN q;
  IDX i;
  for (IDX k = 0; k < pivots->size; k++) {
    i = pivots->ptr[k];
    q = tab->ptr[i * (tab->N + 1) + current_pivot_j] /
        tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
    sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);
  }
}

void gauss_jordan(GJTab_t *tab, IdxStack_t *pivots, FPN *sol) {
  IdxStack_init(pivots);

  //// Make upper echelon
  MaxRes_t temp;
  for (IDX j = 0; j < tab->N - 1; j++) {
    temp = col_max(tab, j, pivots);
    // printf("\n%05.2lf %lu\n", temp.val, temp.arg);
    IdxStack_push(pivots, temp.arg);
    eliminate(tab, pivots);
    // GJTab_print(mat);
  }

  IDX final_row;
  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(pivots, i)) {
      final_row = i;
      break;
    }
  }
  IdxStack_push(pivots, final_row);
  // IdxStack_print(pivots);

  //// Make diagonal (backsubstitution)
  for (IDX j = 0; j < tab->N - 1; j++) {
    backsub_row(tab, pivots);
    // GJTab_print(mat);
  }

  // though we popped everything in the pivot stack (in backsub_row), the data
  // is still in its array, so we can use for the final variable scaling
  IDX i;
  for (IDX k = 0; k < tab->N; k++) {
    i = pivots->ptr[k];
    sol[k] =
        tab->ptr[i * (tab->N + 1) + tab->N] / tab->ptr[i * (tab->N + 1) + k];
  }
}

#ifdef ON_CPU
int main() {
  FPN data[] = {2.0, 3.0, 5.0, 4.0, 1.0, 9.0, 7.0, 6.0, 8.0, 2.0,
                7.0, 9.0, 1.0, 8.0, 3.0, 1.0, 5.0, 1.0, 2.0, 3.0};
  GJTab_t tab = {&data, 4};
  // GJTab_print(&mat);

  IdxStack_t pivots = {malloc(sizeof(IDX) * tab.N),
                       malloc(sizeof(BYTE) * tab.N), tab.N, 0};
  FPN sol[4] = {0};
  gauss_jordan(&tab, &pivots, sol);

  for (IDX k = 0; k < tab.N; k++) {
    printf("%05.4lf ", sol[k]);
  }
  printf("\n");
}
#endif
