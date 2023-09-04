# Basic c98 syntax primal dual solver for LPs

The idea is to have an LP solver that can run on a single GPU thread via OpenCL, to be able to solve many LPs in parallel.
A caveat of this is that much of the memory used is taken to be preallocated (to not commit to static allocation, though working memory would have to be allocated CPU side to avoid recompilation on problem resize and/or stack size limitations).
May not be practical due to program complexity/ammount of branching.

See makefile for running options.

LP solver reference: [Primal-Dual Interior-Point Methods (Aarti Singh)](https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf)

## Test results

Where N is the number of variables per problem. The default solver used by cvxpy appears to be ECOS.

### N=10, Nprob=500

```
Cvxpy:
        seconds elapsed: 1.5288472175598145
        problems converged: 500/500

CPU lib (single thread):
        seconds elapsed: 0.5506298542022705
        problems converged: 500/500
        matching cvxpy: 498/500

CPU lib (multi thread 8c/16t):
        seconds elapsed: 0.1230015754699707
        ...

GPU via OpenCL:
        seconds elapsed: 0.21674084663391113
        problems converged: 500/500
        matching cvxpy: 498/500
```

### N=20, Nprob=200

```
Cvxpy:
        seconds elapsed: 0.6873795986175537
        problems converged: 200/200

CPU lib (single thread):
        seconds elapsed: 0.9988245964050293
        problems converged: 200/200
        matching cvxpy: 197/200

CPU lib (multi thread 8c/16t):
        seconds elapsed: 0.17448711395263672
        ...

GPU via OpenCL:
        seconds elapsed: 1.546123743057251
        problems converged: 200/200
        matching cvxpy: 197/200
```

### N=40, Nprob=100

```
Cvxpy:
        seconds elapsed: 0.47706174850463867
        problems converged: 100/100

CPU lib (single thread):
        seconds elapsed: 2.475322723388672
        problems converged: 100/100
        matching cvxpy: 99/100

CPU lib (multi thread 8c/16t):
        seconds elapsed: 0.3572523593902588
        ...

GPU via OpenCL:
        stack frame size (523024) exceeds limit (262112) in function 'solve_lps'
        (... need to assign SolveVars on cpu here)
```
