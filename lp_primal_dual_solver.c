#include "lp_primal_dual_solver.h"

// negative kkt equalities gap (changing epsilon)
// res size is N + 2M
void kkt_neg_res(LPDef_t *lp, SolverVars_t *vars, FPN eps) {
  // primal feasibility gap (b-Ax)
  for (IDX i = 0; i < lp->N; i++) {
    vars->res[i] = lp->b_ptr[i];
    for (IDX j = 0; j < lp->M; j++) {
      vars->res[i] -= lp->A_ptr[i * lp->M + j] * vars->x[j];
    }
  }

  // complementary slackness gap (x_i u_i = ε, u is lagrange mult for x_i >= 0)
  for (IDX i = 0; i < lp->M; i++) {
    vars->res[lp->N + i] = eps - vars->x[i] * vars->u[i];
  }

  // dual feasibility gap (c - A^T v - u, v is lagrange mult for Ax=b)
  for (IDX i = 0; i < lp->M; i++) {
    vars->res[lp->N + lp->M + i] = lp->c_ptr[i];
    for (IDX j = 0; j < lp->N; j++) {
      vars->res[i] -= vars->v[j] * lp->A_ptr[j * lp->M + i];
      vars->res[i] -= vars->u[i];
    }
  }
}

// gradient is N + 2M, N + 2M (shape)
void init_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX size = lp->N + 2 * lp->M;

  // init all to zero for simplicity
  for (IDX i = 0; i < size; i++) {
    for (IDX j = 0; j < size; j++) {
      vars->grad[i * size + j] = FZERO;
    }
  }

  // top left is A
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad[i * size + j] -= lp->A_ptr[i * lp->M + j];
    }
  }

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad[(lp->N + j) * size + j] = vars->u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad[(lp->N + j) * size + (lp->M + j)] = vars->x[j];
  }

  // bot mid is I_M
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad[(lp->N + lp->M + j) * size + (lp->M + j)] = FONE;
  }

  // bot right is A^T
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      vars->grad[(lp->N + lp->M + j) * size + (2 * lp->M + i)] =
          lp->A_ptr[i * lp->M + j];
    }
  }
}

// most of the gradient is constant, we only need to update middle blocks
void update_grad(LPDef_t *lp, SolverVars_t *vars) {
  IDX size = lp->N + 2 * lp->M;

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad[(lp->N + j) * size + j] = vars->u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    vars->grad[(lp->N + j) * size + (lp->M + j)] = vars->x[j];
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
// solver variables will be allocated by caller
void solve(LPDef_t *lp, SolverVars_t *vars, SolverOpt_t opt) {}
