# Basic c98 syntax primal dual solver for LPs

The idea is to have an LP solver that can run on a single GPU thread via OpenCL, to be able to solve many LPs in parallel.
A caveat of this is that much of the memory used is taken to be preallocated (to not commit to static allocation, though working memory would have to be allocated CPU side to avoid recompilation on problem resize and/or stack size limitations).
May not be practical due to program complexity/ammount of branching.

See makefile for running options.

LP solver reference: [Primal-Dual Interior-Point Methods (Aarti Singh)](https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf)

## Test results

Where N is the number of variables per problem

### N=10, Nprob=500

```
Cvxpy:
        seconds elapsed: 1.5327675342559814
        problems converged: 500/500

CPU lib (single thread):
        seconds elapsed: 1.7172300815582275
        problems converged: 500/500
        matching cvxpy: 498/500

CPU lib (multi thread 8c/16t):
        seconds elapsed: 0.2414712905883789
        ...

GPU via OpenCL:
        seconds elapsed: 0.20771265029907227
        problems converged: 500/500
        matching cvxpy: 498/500
```

### N=20, Nprob=200

```
Cvxpy:
        seconds elapsed: 0.6598720550537109
        problems converged: 200/200

CPU lib (single thread):
        seconds elapsed: 4.104737281799316
        problems converged: 200/200
        matching cvxpy: 197/200

CPU lib (multi thread 8c/16t):
        seconds elapsed: 0.5677616596221924
        ...

GPU via OpenCL:
        seconds elapsed: 1.6011455059051514
        problems converged: 200/200
        matching cvxpy: 197/200
```

### N=40, Nprob=100

```
Cvxpy:
        seconds elapsed: 0.47231149673461914
        problems converged: 100/100

CPU lib (single thread):
        seconds elapsed: 14.257242918014526
        problems converged: 100/100
        matching cvxpy: 99/100

CPU lib (multi thread 8c/16t):
        seconds elapsed: 1.8579258918762207

GPU via OpenCL:
        stack frame size (523024) exceeds limit (262112) in function 'solve_lps'
        (... need to assign SolveVars on cpu here)
```
