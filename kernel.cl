// effectively just concatenating sources, theres probably a better way...
#include "common.c"
#include "gauss_jordan.c"
#include "lp_primal_dual_solver.c"
#include "stack.c"

__kernel void solve_lps(__global FPN *As, __global FPN *bs, __global FPN *cs,
                        __global FPN *sols, __global FPN *kkt_sq_gaps) {
  // kernel idx
  IDX k = get_global_id(0);

  //// Local solver memory assigment
  // working memory
  const IDX N = NDEF; // to prevent unwanted substitution
  const IDX M = MDEF;
  const IDX Ngrad = N + 2 * M;
  FPN tab_arr[Ngrad * (Ngrad + 1)];
  GJTab_t tab = {&tab_arr, Ngrad};

  IDX idxs[Ngrad];
  BYTE contained[Ngrad];
  IdxStack_t pivots = {&idxs, &contained, Ngrad, 0};

  FPN x[M];
  FPN u[M];
  FPN v[N];
  FPN d_xuv[Ngrad];
  SolverVars_t vars = {&tab, &pivots, &x, &u, &v, &d_xuv};

  // make local copies of inputs
  FPN A[N * M];
  FPN b[N];
  FPN c[M];
  for (IDX i = 0; i < N; i++) {
    for (IDX j = 0; j < M; j++) {
      A[i * M + j] = As[k * N * M + i * M + j];
    }
  }
  for (IDX j = 0; j < M; j++) {
    b[j] = bs[k * M + j];
  }
  for (IDX i = 0; i < N; i++) {
    c[i] = cs[k * N + i];
  }
  //// assignment end

  LPDef_t lp = (LPDef_t){&A, &b, &c, N, M};
  SolverOpt_t opts = {0.1, 0.5, 1e-9, 1.0, 100}; // use hardcoded opts for now

  SolverStats_t ss = solve(&lp, &vars, opts);

#ifdef SANITY_CHECK
  if (k == 7) {
    // opencl printf allways adds newline?
    printf("hello from the gpu!");
    print_vec(x, M);
#ifdef USE_FLOAT
    printf("\n\nkkt sq gap = %05.4f, cost = %05.4f, iters = %lu\n", ss.gap,
           ss.cost, ss.iters);
#else
    // does not like lf format?
    printf("\n\nkkt sq gap = %05.4lf, cost = %05.4lf, iters = %lu\n", ss.gap,
           ss.cost, ss.iters);
#endif /* ifdef USE_FLOAT */
  }
#endif

  // to send back
  for (IDX j = 0; j < M; j++) {
    sols[k * M + j] = x[j];
  }
  kkt_sq_gaps[k] = ss.gap;
}
