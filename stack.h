#include "common.h"

typedef struct IdxStack {
  IDX *ptr;
  BYTE *contained; // for constant time membership lookup
  IDX capacity;
  IDX size;
} IdxStack_t;

void IdxStack_init(IdxStack_t *stack);

// do not push beyond end (if size==capacity)...
void IdxStack_push(IdxStack_t *stack, IDX new_idx);

// do not pop when empty (size = 0)...
IDX IdxStack_pop(IdxStack_t *stack);

// no bounds checking on looked up value!!!
// we expect to store permutations so we expect val<capacity
BYTE IdxStack_contains(IdxStack_t *stack, IDX val);

#ifdef ON_CPU
void IdxStack_print(IdxStack_t *stack);

void stack_test();
#endif
