#include "stack.h"

void IdxStack_init(IdxStack_t *stack) {
  for (IDX k = 0; k < stack->capacity; k++) {
    stack->contained[k] = 0;
  }
  stack->size = 0;
}

void IdxStack_push(IdxStack_t *stack, IDX new) {
#ifdef DEBUG_STACK
  if (new >= stack->capacity) {
    printf("WARNING: atempt to insert large index (%lu).\n", new);
  }
  if (stack->capacity <= stack->size) {
    printf("WARNING: atempt to push to a full stack or insert large index.\n");
  }
#endif /* ifdef DEBUG_STACK */
  stack->ptr[stack->size] = new;
  stack->contained[new] = 1;
  stack->size++;
}

IDX IdxStack_pop(IdxStack_t *stack) {
#ifdef DEBUG_STACK
  if (stack->size <= 0) {
    printf("WARNING: atempt to pop empty stack.\n");
  }
#endif /* ifdef DEBUG_STACK */
  IDX old_end_element = stack->ptr[--stack->size];
  stack->contained[old_end_element] = 0;
  return old_end_element;
}

BYTE IdxStack_contains(IdxStack_t *stack, IDX val) {
  return stack->contained[val];
}

#ifdef ON_CPU
void IdxStack_print(IdxStack_t *stack) {
  for (IDX k = 0; k < stack->size; k++) {
    printf("%lu ", stack->ptr[k]);
  }
  printf("\n");
}
#endif
