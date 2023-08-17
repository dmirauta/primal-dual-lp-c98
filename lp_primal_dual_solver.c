#include "lp_primal_dual_solver.h"

// negative kkt equalities gap (changing epsilon)
// res size is N + 2M
void kkt_neg_res(LPDef_t *lp, FPN *x, FPN *u, FPN *v, FPN *res, FPN eps) {
  // primal feasibility gap (b-Ax)
  for (IDX i = 0; i < lp->N; i++) {
    res[i] = lp->b_ptr[i];
    for (IDX j = 0; j < lp->M; j++) {
      res[i] -= lp->A_ptr[i * lp->M + j] * x[j];
    }
  }

  // complementary slackness gap (x_i u_i = ε, u is lagrange mult for x_i >= 0)
  for (IDX i = 0; i < lp->M; i++) {
    res[lp->N + i] = eps - x[i] * u[i];
  }

  // dual feasibility gap (c - A^T v - u, v is lagrange mult for Ax=b)
  for (IDX i = 0; i < lp->M; i++) {
    res[lp->N + lp->M + i] = lp->c_ptr[i];
    for (IDX j = 0; j < lp->N; j++) {
      res[i] -= v[j] * lp->A_ptr[j * lp->M + i];
      res[i] -= u[i];
    }
  }
}

// gradient is N + 2M, N + 2M (shape)
void init_grad(LPDef_t *lp, FPN *x, FPN *u, FPN *grad) {
  IDX size = lp->N + 2 * lp->M;

  // init all to zero for simplicity
  for (IDX i = 0; i < size; i++) {
    for (IDX j = 0; j < size; j++) {
      grad[i * size + j] = FZERO;
    }
  }

  // top left is A
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      grad[i * size + j] -= lp->A_ptr[i * lp->M + j];
    }
  }

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    grad[(lp->N + j) * size + j] = u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    grad[(lp->N + j) * size + (lp->M + j)] = x[j];
  }

  // bot mid is I_M
  for (IDX j = 0; j < lp->M; j++) {
    grad[(lp->N + lp->M + j) * size + (lp->M + j)] = FONE;
  }

  // bot right is A^T
  for (IDX i = 0; i < lp->N; i++) {
    for (IDX j = 0; j < lp->M; j++) {
      grad[(lp->N + lp->M + j) * size + (2 * lp->M + i)] =
          lp->A_ptr[i * lp->M + j];
    }
  }
}

void update_grad(LPDef_t *lp, FPN *x, FPN *u, FPN *grad) {
  IDX size = lp->N + 2 * lp->M;

  // mid left is diag u
  for (IDX j = 0; j < lp->M; j++) {
    grad[(lp->N + j) * size + j] = u[j];
  }

  // mid mid is diag x
  for (IDX j = 0; j < lp->M; j++) {
    grad[(lp->N + j) * size + (lp->M + j)] = x[j];
  }
}

// iterative rootfinding in the linearised kkt residual
void solve() {
  // TODO
}
