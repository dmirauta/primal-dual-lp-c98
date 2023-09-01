SHELL := /bin/bash
COMMON_FLAGS = -DON_CPU -I./src src/stack.c src/common.c src/gauss_jordan.c src/lp_primal_dual_solver.c

gcctest:
	gcc $(COMMON_FLAGS) test.c -g -o test.out
	time ./test.out

ocltest:
	python verification/opencl.py

lib:
	gcc $(COMMON_FLAGS) -shared -fPIC -o liblpsolve.so
