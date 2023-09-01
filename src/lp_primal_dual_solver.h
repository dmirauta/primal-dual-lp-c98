#pragma once

#include "common.h"
#include "gauss_jordan.h"

// linear program definition
typedef struct LPDef {
  // N x M constraint matrix
  FPN *A_ptr;
  // N vec of constraint vals
  FPN *b_ptr;
  // M vec of cost coefficients
  FPN *c_ptr;
  // num constraints
  IDX N;
  // num variables
  IDX M;
} LPDef_t;

typedef struct SolverVars {
  // holds (grad | res) matrix of size (N+2M, N+2M+1)
  GJTab_t *grad_res;
  IdxStack_t *pivots;
  // size M
  FPN *x;
  // size M
  FPN *u;
  // size N
  FPN *v;
  // direction found by gauss jordan, size N+2M
  FPN *d_xuv;
} SolverVars_t;

typedef struct SolverOpt {
  // initial complementary slackness relaxation 'epsilon'
  FPN eps;
  // epsilon multiplier (per iter)
  FPN eps_q;
  // convergence tolerance (for squared residual norm)
  FPN tol;
  FPN init_stepsize;
  // maximum iterations
  IDX maxiter;
} SolverOpt_t;

typedef struct SolverStats {
  // optimality gap (from KKT equality)
  FPN gap;
  FPN cost;
  IDX iters;
  BYTE aborted;
} SolverStats_t;

// iterative rootfinding in the linearised kkt residual
// solver variables to be allocated by caller
SolverStats_t solve(LPDef_t *lp, SolverVars_t *vars, SolverOpt_t opt);
