SHELL := /bin/bash
COMMON_FLAGS = -DON_CPU -I./src src/stack.c src/common.c src/gauss_jordan.c src/lp_primal_dual_solver.c

gcctest:
	gcc $(COMMON_FLAGS) test.c -g -o test.out
	time ./test.out

wraptest:
	PYOPENCL_CTX='0' python verification/wrapper.py

lib:
	gcc $(COMMON_FLAGS) -O3 -shared -fPIC -o liblpsolve.so
