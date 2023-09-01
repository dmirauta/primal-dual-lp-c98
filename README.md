# Basic c98 syntax primal dual solver for LPs

The idea is to have an LP solver that can run on a single GPU thread via OpenCL, to be able to solve many LPs in parallel.
A caveat of this is that much of the memory used is taken to be preallocated (to not commit to static allocation, though working memory would have to be allocated CPU side to avoid recompilation on problem resize).
May not be practical due to program complexity/ammount of branching.

Appears to be functional according to basic tests via gcc and pyopencl tests.

See makefile for running options.

LP solver reference: [Primal-Dual Interior-Point Methods (Aarti Singh)](https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf)
