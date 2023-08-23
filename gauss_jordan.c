#include "gauss_jordan.h"

#ifdef ON_CPU
void GJTab_print(GJTab_t *tab, IdxStack_t *pivots) {
  printf("\n");
  for (IDX i = 0; i < tab->N; i++) {
    for (IDX j = 0; j < tab->N + 1; j++) {
      if (j < pivots->size) {
        if (pivots->ptr[j] == i) {
          printf(SETREDTEXT);
        }
      }
      printf("%05.2lf ", tab->ptr[i * (tab->N + 1) + j]);
      printf(RESETCOLOR);
    }
    printf(" %05lu\n", i);
  }
  printf("\n");
  for (IDX j = 0; j < tab->N + 1; j++) {
    printf("%05lu ", j);
  }
  printf("\n");
}
#endif

FPN FPN_abs(FPN a) { return a > 0 ? a : -a; }

IDX col_maxabs(GJTab_t *tab, IDX j, IdxStack_t *no_max) {
  FPN maxabs = NEG_INF;
  FPN compared;
  IDX argmax;
  for (IDX i = 0; i < tab->N; i++) {
    compared = FPN_abs(tab->ptr[i * (tab->N + 1) + j]);
    if (!IdxStack_contains(no_max, i) && (compared > maxabs)) {
      maxabs = compared;
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
  FPN q, pv, cv;
#ifdef DEBUG_GJ
  printf("\n  -- pivot col: %lu, pivot: %lu --\n", current_pivot_j,
         current_pivot_i);
  IDX rows_calcelled = 0;
#endif
  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(pivots, i)) {
      cv = tab->ptr[i * (tab->N + 1) + current_pivot_j];
      pv = tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
      q = cv / (pv + EPSILON);
      sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);

#ifdef DEBUG_GJ
      printf("\ncancelling %lu, mult %lf/%lf\n", i, cv, pv);
      if (FPN_abs(cv) > EPSILON) { // save some output space...
        GJTab_print(tab, pivots);
      }
      rows_calcelled++;
#endif
    }
  }
#ifdef DEBUG_GJ
  if (rows_calcelled == 0) {
    GJTab_print(tab, pivots);
  }
#endif
}

void backsub_row(GJTab_t *tab, IdxStack_t *pivots) {
  IDX current_pivot_j = pivots->size - 1;
  IDX current_pivot_i = IdxStack_pop(pivots);
  FPN q, cv, pv;
  IDX i;
  for (IDX k = 0; k < pivots->size; k++) {
    i = pivots->ptr[k];
    cv = tab->ptr[i * (tab->N + 1) + current_pivot_j];
    pv = tab->ptr[current_pivot_i * (tab->N + 1) + current_pivot_j];
    q = cv / (pv + EPSILON);
    sub_scaled_row(tab, current_pivot_j, current_pivot_i, i, q);

#ifdef DEBUG_GJ
    printf("\ncancelling %lu, mult %lf/%lf\n", i, cv, pv);
    if (FPN_abs(cv) > EPSILON) { // save some output space...
      GJTab_print(tab, pivots);
    }
#endif
  }
}

void gauss_jordan(GJTab_t *tab, IdxStack_t *pivots, FPN *sol) {
  IdxStack_init(pivots);

  //// Make upper echelon
  IDX pivot_row;
  for (IDX j = 0; j < tab->N - 1; j++) {
    // printf("\n%05.2lf %lu\n", temp.val, temp.arg);
    pivot_row = col_maxabs(tab, j, pivots);
    IdxStack_push(pivots, pivot_row);
    eliminate(tab, pivots);
  }

  for (IDX i = 0; i < tab->N; i++) {
    if (!IdxStack_contains(pivots, i)) {
      pivot_row = i;
      break;
    }
  }
  IdxStack_push(pivots, pivot_row);

  //// Make diagonal (backsubstitution)
  for (IDX j = 0; j < tab->N - 1; j++) {
    backsub_row(tab, pivots);
  }

  // though we popped everything in the pivot stack (in backsub_row), the data
  // is still in its array, so we can use for the final variable scaling
  IDX i;
  for (IDX k = 0; k < tab->N; k++) {
    i = pivots->ptr[k];
    sol[k] = tab->ptr[i * (tab->N + 1) + tab->N] /
             (tab->ptr[i * (tab->N + 1) + k] + EPSILON);
  }
}
