#include "common.h"

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
