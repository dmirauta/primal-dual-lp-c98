#include "stack.h"

// Tableau (A|b) for Gauss-Jordan elimination (for Ax=b).
// Dynamic allocation friendly, but not actually needed,
// statically sized FPN mat[N][N+1] could be fine instead.
typedef struct GJTableau {
  FPN *ptr;
  IDX N;
} GJTab_t;

#ifdef ON_CPU
void GJTab_print(GJTab_t *tab, IdxStack_t *pivots);
#endif

void gauss_jordan(GJTab_t *tab, IdxStack_t *pivots, FPN *sol);
