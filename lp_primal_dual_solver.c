#include "lp_primal_dual_solver.h"

// negative KKT equalities gap
// res size is N + 2M
void kkt_neg_res(LPDef_t *lp, SolverVars_t *vars, FPN cs_eps) {
  IDX tab_width = lp->N + 2 * lp->M + 1;
  IDX res_idx;

  // primal feasibility gap (b-Ax)
  for (IDX i = 0; i < lp->N; i++) {
    res_idx = i * tab_width + (tab_width - 1);
    vars->grad_res->ptr[res_idx] = lp->b_ptr[i];
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad_res->ptr[res_idx] -= lp->A_ptr[i * lp->M + j] * vars->x[j];
    }
  }

  // complementary slackness gap (x_i u_i = ε, u is lagrange mult for x_i >= 0)
  for (IDX i = 0; i < lp->M; i++) {
    res_idx = (lp->N + i) * tab_width + (tab_width - 1);
    vars->grad_res->ptr[res_idx] = cs_eps - vars->x[i] * vars->u[i];
  }

  // dual feasibility gap (c - A^T v - u, v is lagrange mult for Ax=b)
  for (IDX i = 0; i < lp->M; i++) {
    res_idx = (lp->N + lp->M + i) * tab_width + (tab_width - 1);
    vars->grad_res->ptr[res_idx] = lp->c_ptr[i];
    for (IDX j = 0; j < lp->N; j++) {
      vars->grad_res->ptr[res_idx] -= vars->v[j] * lp->A_ptr[j * lp->M + i];
    }
    vars->grad_res->ptr[res_idx] -= vars->u[i];
  }
}

void init_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX tab_width = lp->N + 2 * lp->M + 1;

  // init grad_res to 0
  for (IDX i = 0; i < tab_width - 1; i++) {
    for (IDX j = 0; j < tab_width; j++) {
      // clamp small values (at end) or add across the board, or neither?
      vars->grad_res->ptr[i * tab_width + j] = 0;
      // vars->grad_res->ptr[i * tab_width + j] = EPSILON;
    }
  }

  // top left is A
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad_res->ptr[i * tab_width + j] = lp->A_ptr[i * lp->M + j];
    }
  }

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * tab_width + j] = vars->u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * tab_width + (lp->M + j)] = vars->x[j];
  }

  // bot mid is I_M
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + lp->M + j) * tab_width + (lp->M + j)] = FONE;
  }

  // bot right is A^T
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad_res->ptr[(lp->N + lp->M + j) * tab_width + (2 * lp->M + i)] =
          lp->A_ptr[i * lp->M + j];
    }
  }

  // // clamp small values
  // for (IDX i = 0; i < tab_width - 1; i++) {
  //   for (IDX j = 0; j < tab_width; j++) {
  //     clamp(&vars->grad_res->ptr[i * tab_width + j]);
  //   }
  // }
}

// UNUSED
// In principle, most of the gradient is constant, we only need to update middle
// blocks, though in practice our gauss jordan method will change the tableau.
// Could reshape to put this part in lower block row and reuse cancellation in
// "upper" parts? (pivot order would also change)
void update_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX tab_width = lp->N + 2 * lp->M + 1;

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * tab_width + j] = vars->u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * tab_width + (lp->M + j)] = vars->x[j];
  }
}

FPN L2(LPDef_t *lp, SolverVars_t *vars) {
  IDX tab_width = lp->N + 2 * lp->M + 1;
  IDX res_idx;
  FPN sq_diff = FZERO;

  for (IDX i = 0; i < lp->N + 2 * lp->M; i++) {
    res_idx = i * tab_width + (tab_width - 1);
    sq_diff += vars->grad_res->ptr[res_idx] * vars->grad_res->ptr[res_idx];
  }

  return sq_diff;
}

FPN cost(LPDef_t *lp, SolverVars_t *vars) {
  FPN c = 0;
  for (IDX i = 0; i < lp->M; i++) {
    c += vars->x[i] * lp->c_ptr[i];
  }
  return c;
}

SolverStats_t solve(LPDef_t *lp, SolverVars_t *vars, SolverOpt_t opt) {

  // initialise variables
  for (IDX i = 0; i < lp->M; i++) {
    vars->x[i] = FONE;
    vars->u[i] = FONE;
  }
  for (IDX i = 0; i < lp->N; i++) {
    vars->v[i] = FONE;
  }

  FPN step = opt.init_stepsize;
  FPN old_gap = -NEG_INF;
  FPN new_gap;
  FPN cs_eps = opt.eps;
  IDX i = 0;
  BYTE aborted = 0;

  while ((old_gap > opt.tol) && (i < opt.maxiter)) {

    init_grad(lp, vars);
    kkt_neg_res(lp, vars, cs_eps);

#ifdef DEBUG_SOLVE
    // showing tab before elimination
    GJTab_print(vars->grad_res, vars->pivots);
#endif /* ifdef DEBUG_SOLVE */

    new_gap = L2(lp, vars);

    // abort if blowing up
    if (new_gap > 100 * old_gap) {
      aborted = 1;
      break;
    }

    if (new_gap > old_gap) {
      step /= 2;
    }

    gauss_jordan(vars->grad_res, vars->pivots, vars->d_xuv);

    // update variables
    for (IDX i = 0; i < lp->N; i++) {
      vars->v[i] += step * vars->d_xuv[2 * lp->M + i];
    }
    for (IDX i = 0; i < lp->M; i++) {
      vars->x[i] += step * vars->d_xuv[i];
      vars->u[i] += step * vars->d_xuv[lp->M + i];

      // project to R_+
      if (vars->x[i] < EPSILON) {
        vars->x[i] = EPSILON;
      }
    }

    old_gap = new_gap;
    cs_eps *= opt.eps_q;
    i++;

#ifdef DEBUG_SOLVE
    printf("\ndirectional grad\n");
    print_vec(vars->d_xuv, lp->N + 2 * lp->M);

    printf("\nx\n");
    print_vec(vars->x, lp->M);

    printf("\nu\n");
    print_vec(vars->u, lp->M);

    printf("\nv\n");
    print_vec(vars->v, lp->N);

    printf("\n\ni = %lu, gap = %lf\n", i, new_gap);
    printf("\n=========\n");
#endif /* ifdef DEBUG_SOLVE */
  }

  return (SolverStats_t){old_gap, cost(lp, vars), i, aborted};
}
