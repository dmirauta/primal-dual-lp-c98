# Basic c98 syntax primal dual solver for LPs

The idea is to have an LP solver that can run on a single GPU thread via OpenCL, to be able to solve many LPs in parallel.
A caveat of this is that much of the memory used is taken to be preallocated (as this is done CPU side).
May not be practical due to program complexity/ammount of branching.

Currently the simpler example is solving correctly, folded is returning a feasible but suboptimal solution.

`gcc stack.c common.c gauss_jordan.c test.c lp_primal_dual_solver.c -DON_CPU && time ./a.out`
