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
      vars->grad_res->ptr[res_idx] -= vars->u[i];
    }
  }
}

void init_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX tab_width = lp->N + 2 * lp->M + 1;

  // init all to zero for simplicity
  for (IDX i = 0; i < tab_width; i++) {
    for (IDX j = 0; j < tab_width; j++) {
      vars->grad_res->ptr[i * tab_width + j] = FZERO;
    }
  }

  // top left is A
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad_res->ptr[i * tab_width + j] -= lp->A_ptr[i * lp->M + j];
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
}

// In principle, most of the gradient is constant, we only need to update middle
// blocks, though in practice our gauss jordan method will change the tableau.
// Could reshape to put this part in lower block row and reuse cancellation in
// upper parts?
void update_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX size = lp->N + 2 * lp->M;

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * size + j] = vars->u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad_res->ptr[(lp->N + j) * size + (lp->M + j)] = vars->x[j];
  }
}

FPN L2(FPN *res, IDX size) {
  FPN sq_diff = FZERO;

  for (IDX i = 0; i < size; i++) {
    sq_diff += res[i] * res[i];
  }

  return sq_diff;
}

// iterative rootfinding in the linearised kkt residual
// solver variables to be allocated by caller
void solve(LPDef_t *lp, SolverVars_t *vars, SolverOpt_t opt) {}
