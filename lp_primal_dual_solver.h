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
  FPN *x;
  FPN *u;
  FPN *v;
  // direction found by gauss jordan
  FPN *d_xuv;
} SolverVars_t;

typedef struct SolverOpt {
  // initial complementary slackness relaxation 'epsilon'
  FPN eps;
  // epsilon multiplier (per iter)
  FPN eps_q;
  // convergence tolerance
  FPN tol;
  // maximum iterations
  IDX maxiter;
} SolverOpt_t;
