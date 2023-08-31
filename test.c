#include "lp_primal_dual_solver.h"
#include <stdio.h>
#include <stdlib.h>

#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"

void stack_test() {
  IdxStack_t test = {malloc(sizeof(IDX) * 2), malloc(sizeof(IDX) * 2), 2, 0};
  IdxStack_init(&test);
  IdxStack_push(&test, 7);
  IdxStack_print(&test);
  IdxStack_push(&test, 2);
  IdxStack_print(&test);
  printf("pop %lu\n", IdxStack_pop(&test));
  IdxStack_print(&test);

  // should explicitly free mem...
}

void gauss_jordan_test() {
  FPN data[] = {2.0, 3.0, 5.0, 4.0, 1.0, 9.0, 7.0, 6.0, 8.0, 2.0,
                7.0, 9.0, 1.0, 8.0, 3.0, 1.0, 5.0, 1.0, 2.0, 3.0};
  GJTab_t tab = {&data, 4};

  IdxStack_t pivots = {malloc(sizeof(IDX) * tab.N),
                       malloc(sizeof(BYTE) * tab.N), tab.N, 0};
  GJTab_print(&tab, &pivots);

  FPN sol[4] = {0};
  gauss_jordan(&tab, &pivots, sol);

  printf("\nGot: ");
  print_vec(sol, tab.N);
  printf("\nExpected: -0.05128205  0.76923077  0.12820513 -0.46153846\n");

  // should explicitly free mem...
}

void dual_solve_test_general(FPN *Aptr, FPN *bptr, FPN *cptr, IDX N, IDX M) {

  LPDef_t lp = {Aptr, bptr, cptr, N, M};

  unsigned long fps = sizeof(FPN);
  IDX Ngrad = N + 2 * M;
  GJTab_t tab = {malloc(fps * Ngrad * (Ngrad + 1)), Ngrad};
  IdxStack_t pivots = {malloc(sizeof(IDX) * tab.N),
                       malloc(sizeof(BYTE) * tab.N), tab.N, 0};
  SolverVars_t sv = {
      &tab,
      &pivots,
      malloc(fps * M),    // x
      malloc(fps * M),    // u
      malloc(fps * N),    // v
      malloc(fps * Ngrad) // d_xuv
  };

  SolverOpt_t so = {0.1, 0.5, 1e-19, 1.0, 1000};

  SolverStats_t ss = solve(&lp, &sv, so);

  printf("solution: \n");
  print_vec(sv.x, M);

  printf("\nd_xuv: \n");
  print_vec(sv.d_xuv, Ngrad);

  // printf("\ngrad_neg_res: \n");
  // GJTab_print(&tab, &pivots);

  printf("\n\ngap = %05.4lf, cost = %05.4lf, iters = %lu\n", ss.gap, ss.cost,
         ss.iters);

  // should explicitly free mem...
}

void dual_solve_test_flattened() {
  FPN A[] = {1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000,
             1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000};

  FPN b[] = {0.2321, 0.1831, 0.5353, 0.8220};
  FPN c[] = {0.2803, 0.0590, 0.4413, 0.2194, 0.0000, 0.0000};

  IDX N = 4;
  IDX M = 6;

  dual_solve_test_general(&A, &b, &c, N, M);
}

void dual_solve_test_simple() {
  FPN A[] = {0.4753, 0.7483, 0.0000, 0.0000, 0.1001, 0.3721, 0.0000, 0.0000,
             0.2321, 0.1831, 1.0000, 0.0000, 0.2676, 0.4110, 0.0000, 1.0000};

  FPN b[] = {0.6550, 0.2634, 0.6905, 1.0794};

  FPN c[] = {0.2840, 0.8903, 0.0000, 0.0000};

  IDX N = 4;
  IDX M = 4;

  dual_solve_test_general(&A, &b, &c, N, M);
}

int main() {
  // printf("GJ:\n");
  // gauss_jordan_test();

  printf("\n\nLP solve:\n");
  dual_solve_test_simple();
  // dual_solve_test_flattened();

  return 0;
}
