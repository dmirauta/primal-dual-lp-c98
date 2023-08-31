# Basic c98 syntax primal dual solver for LPs

The idea is to have an LP solver that can run on a single GPU thread via OpenCL, to be able to solve many LPs in parallel.
A caveat of this is that much of the memory used is taken to be preallocated (as this is done CPU side).
May not be practical due to program complexity/ammount of branching.

Appears to be functional according to basic tests via gcc, it is running via opencl also, but many kernels are randomly failing (on lp defs that work via gcc).

Gcc testing (single execution):
`gcc stack.c common.c gauss_jordan.c test.c lp_primal_dual_solver.c -DON_CPU && time ./a.out`

Dispatch opencl job from python test (solve many problems in parallel):
`cd verification && python opencl.py`

LP solver reference: [Primal-Dual Interior-Point Methods (Aarti Singh)](https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf)

Issues:

- Will often not get past initialisation when running directly after compilation, only even somewhat working when running with a cached kernel.
- Large kkt gaps for seemingly correct solutions.
