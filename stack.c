#include "stack.h"

void IdxStack_init(IdxStack_t *stack) {
  for (IDX k = 0; k < stack->size; k++) {
    stack->contained[k] = 0;
  }
  stack->size = 0;
}

void IdxStack_push(IdxStack_t *stack, IDX new) {
  stack->ptr[stack->size] = new;
  stack->contained[new] = 1;
  stack->size++;
}

IDX IdxStack_pop(IdxStack_t *stack) {
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

void stack_test() {
  IdxStack_t test = {malloc(sizeof(IDX) * 2), malloc(sizeof(IDX) * 2), 2, 0};
  IdxStack_init(&test);
  IdxStack_push(&test, 7);
  IdxStack_print(&test);
  IdxStack_push(&test, 2);
  IdxStack_print(&test);
  printf("pop %lu\n", IdxStack_pop(&test));
  IdxStack_print(&test);
}
#endif
