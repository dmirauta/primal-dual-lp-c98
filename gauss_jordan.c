#include "gauss_jordan.h"

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

BYTE FPN_abs(FPN a) { return a > 0 ? a : -a; }

IDX col_maxabs(GJTab_t *tab, IDX j, IdxStack_t *no_max) {
  FPN maxabs = NEG_INF;
  IDX argmax;
  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(no_max, i) &&
        (FPN_abs(tab->ptr[i * (tab->N + 1) + j]) > maxabs)) {
      argmax = i;
    }
  }
  return argmax;
}

void sub_scaled_row(GJTab_t *tab, IDX from_col, IDX to_sub, IDX sub_from,
                    FPN q) {
  // while we are iterating columnwise, reads/writes zigzag between two rows?
  // e.g.
  // m11 m12 m13 ...
  //  | / | / ...
  // m21 m22 ...
  // although one rows is only read, while the other only written
  // is this efficient in terms of memory ops?
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
      // skip if already essentially 0
      // pointless on GPU since we have to pause for both branches?
      if (!(FPN_abs(tab->ptr[i * (tab->N + 1) + current_pivot_j]) < EPSILON)) {
        q = tab->ptr[i * (tab->N + 1) + current_pivot_j] /
            tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
        sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);
      }
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
    // skip if already essentially 0
    // pointless on GPU since we have to pause for both branches?
    if (!(FPN_abs(tab->ptr[i * (tab->N + 1) + current_pivot_j]) < EPSILON)) {
      q = tab->ptr[i * (tab->N + 1) + current_pivot_j] /
          tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
      sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);
    }
  }
}

void gauss_jordan(GJTab_t *tab, IdxStack_t *pivots, FPN *sol) {
  IdxStack_init(pivots);

  //// Make upper echelon
  for (IDX j = 0; j < tab->N - 1; j++) {
    // printf("\n%05.2lf %lu\n", temp.val, temp.arg);
    IdxStack_push(pivots, col_maxabs(tab, j, pivots));
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
