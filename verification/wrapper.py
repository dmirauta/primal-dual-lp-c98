import os, time, ctypes, struct, sys
from sys import platform
import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt
from multiprocessing import Pool

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"

mf = cl.mem_flags
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

parent = os.getcwd()
build_options_base = f"-I {parent}/src"

use_float = False
if use_float:
    np_fpn = np.float32
    c_fpn = ctypes.c_float
    struct_float_char = "f"
    build_options_base += " -D USE_FLOAT"
else:
    np_fpn = np.float64
    c_fpn = ctypes.c_double
    struct_float_char = "d"
fps = ctypes.sizeof(c_fpn)
c_fpn_p = ctypes.POINTER(c_fpn)


class LPDef(ctypes.Structure):
    _fields_ = [
        ("A_ptr", c_fpn_p),
        ("b_ptr", c_fpn_p),
        ("c_ptr", c_fpn_p),
        ("N", ctypes.c_ulong),
        ("M", ctypes.c_ulong),
    ]


class GJTableau(ctypes.Structure):
    _fields_ = [
        ("ptr", c_fpn_p),
        ("N", ctypes.c_ulong),
    ]


class IdxStack(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_ulong)),
        ("contained", ctypes.POINTER(ctypes.c_bool)),
        ("capacity", ctypes.c_ulong),
        ("size", ctypes.c_ulong),
    ]


class SolverVars(ctypes.Structure):
    _fields_ = [
        ("grad_res", ctypes.POINTER(GJTableau)),
        ("pivots", ctypes.POINTER(IdxStack)),
        ("x", c_fpn_p),
        ("u", c_fpn_p),
        ("v", c_fpn_p),
        ("d_xuv", c_fpn_p),
    ]


class SolverOpt(ctypes.Structure):
    _fields_ = [
        ("eps", c_fpn),
        ("eps_q", c_fpn),
        ("tol", c_fpn),
        ("init_stepsize", c_fpn),
        ("maxiter", ctypes.c_ulong),
    ]


class SolverStats(ctypes.Structure):
    _fields_ = [
        ("gap", c_fpn),
        ("cost", c_fpn),
        ("iters", ctypes.c_ulong),
        ("aborted", ctypes.c_bool),
    ]


## short defs
# stuct def key:
# = <- native endianness
# P <- pointer
# L <- unsigned long/size_t/IDX
# f <- float/FPN (use float)
# d <- double/long float/FPN (no use float)
lp_def_struct_fmt = "=PPPLL"
solver_vars_struct_fmt = "=" + "P" * 6
solver_opt_struct_fmt = "=" + struct_float_char * 4 + "L"
solver_stats_struct_fmt = "=" + struct_float_char * 2 + "L"


try:
    with open("kernel.cl", "r") as f:
        kernel_cl = f.read()
except:
    print("\nIntended to be run from project root\n")


def solve_probs(probs, opts=(0.1, 0.5, 1e-9, 1.0, 100)):
    Nprobs = len(probs)

    N, M = probs[0][0].shape
    build_opts = f"{build_options_base} -D NDEF={N} -D MDEF={M}"
    build_opts += f" -D EPSILON=1e-9"
    # print("build opts: ", build_opts)

    As_cpu = np.zeros(N * M * Nprobs, dtype=np_fpn)
    bs_cpu = np.zeros(N * Nprobs, dtype=np_fpn)
    cs_cpu = np.zeros(M * Nprobs, dtype=np_fpn)
    sols_cpu = np.zeros(M * Nprobs, dtype=np_fpn)
    kkt_sq_res_cpu = np.zeros(Nprobs, dtype=np_fpn)
    # opts_c = struct.pack(solver_opt_struct_fmt, *opts)  # not currently being passed

    # make flat arrays
    for i, (A, b, c) in enumerate(probs):
        As_cpu[i * N * M : (i + 1) * N * M] = A.flatten()
        bs_cpu[i * N : (i + 1) * N] = b.flatten()
        cs_cpu[i * M : (i + 1) * M] = c.flatten()

    As_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=As_cpu)
    bs_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bs_cpu)
    cs_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cs_cpu)
    sols_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, fps * M * Nprobs)
    kkt_sq_res_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, fps * Nprobs)

    prg = cl.Program(ctx, kernel_cl).build(options=build_opts)
    prg.solve_lps(
        queue, (Nprobs,), None, As_gpu, bs_gpu, cs_gpu, sols_gpu, kkt_sq_res_gpu
    )

    cl.enqueue_copy(queue, sols_cpu, sols_gpu)
    cl.enqueue_copy(queue, kkt_sq_res_cpu, kkt_sq_res_gpu)

    for obj in [As_gpu, bs_gpu, cs_gpu, sols_gpu, kkt_sq_res_gpu]:
        obj.release()

    return sols_cpu, kkt_sq_res_cpu


if platform in ("linux", "linux2"):
    liblpsolve = ctypes.cdll.LoadLibrary(os.path.join(parent, "liblpsolve.so"))
    liblpsolve.solve.restype = SolverStats
else:
    raise Exception("Loading (or compiling) c lib is only implemented for linux")


def solve_wrap(A, b, c, opts=(0.1, 0.5, 1e-9, 1.0, 100)):
    """
    Our LP implementation running on CPU, called from python
    """

    if use_float == True:
        raise Exception("float lib loading not implemented")
    if False in [arr.dtype == np_fpn for arr in [A, b, c]]:
        raise Exception(f"expecting {np_fpn}")

    N, M = A.shape
    Ngrad = N + 2 * M
    lpdef_c = LPDef(
        A.ctypes.data_as(c_fpn_p),
        b.ctypes.data_as(c_fpn_p),
        c.ctypes.data_as(c_fpn_p),
        N,
        M,
    )

    tab_arr = np.zeros(Ngrad * (Ngrad + 1), dtype=np_fpn)
    gjtab_c = GJTableau(tab_arr.ctypes.data_as(c_fpn_p), Ngrad)

    idxs = np.zeros(Ngrad, dtype=np.uint)
    contained = np.zeros(Ngrad, dtype=np.bool_)
    pivots_c = IdxStack(
        idxs.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),
        contained.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        Ngrad,
        0,
    )

    x = np.zeros(M, dtype=np_fpn)
    u = np.zeros(M, dtype=np_fpn)
    v = np.zeros(N, dtype=np_fpn)
    d_xuv = np.zeros(Ngrad, dtype=np_fpn)
    vars_c = SolverVars(
        ctypes.pointer(gjtab_c),
        ctypes.pointer(pivots_c),
        x.ctypes.data_as(c_fpn_p),
        u.ctypes.data_as(c_fpn_p),
        v.ctypes.data_as(c_fpn_p),
        d_xuv.ctypes.data_as(c_fpn_p),
    )

    opts_c = SolverOpt(*opts)

    stats_c = liblpsolve.solve(ctypes.pointer(lpdef_c), ctypes.pointer(vars_c), opts_c)

    return x, dict(
        gap=stats_c.gap,
        cost=stats_c.cost,
        iters=stats_c.iters,
        success=not stats_c.aborted,
    )


def wrap_multisolve(probs):
    return tuple(zip(*[solve_wrap(A, b, c) for A, b, c in probs]))


def solve_wrap_mtk(args):
    A, b, c = args
    return solve_wrap(A, b, c)


def wrap_multisolve_multithread(probs):
    with Pool() as pool:
        res = tuple(zip(*pool.map(solve_wrap_mtk, probs)))
    return res


if __name__ == "__main__":
    from simple_prob import gen_lp, cvxpy_multisolve

    N = 40  # base N will not be the same as augmented...
    Nprobs = 100
    eps = 1e-9

    A, b, c = gen_lp(N=N)
    # print(solve_wrap(A, b, c))
    # summarise_cvxpy_sol(N)

    seeds = list(range(1, 1 + Nprobs))
    probs = [gen_lp(N=N, seed=s) for s in seeds]
    _N, _M = probs[0][0].shape

    print("\nCvxpy:")
    t0 = time.time()
    cvxpy_sols, cvxpy_stats = cvxpy_multisolve(probs)
    print("\tseconds elapsed:", time.time() - t0)
    cvxpy_successful = [i for i in range(Nprobs) if cvxpy_stats[i]["success"]]
    print("\tproblems converged: {}/{}".format(len(cvxpy_successful), Nprobs))

    time.sleep(0.1)
    print("\nCPU lib (single thread):")
    t0 = time.time()
    cpu_sols, cpu_stats = wrap_multisolve(probs)
    print("\tseconds elapsed:", time.time() - t0)
    cpu_successful = [i for i in range(Nprobs) if cpu_stats[i]["gap"] < eps]
    print("\tproblems converged: {}/{}".format(len(cpu_successful), Nprobs))
    cpu_sol_dists = [
        np.abs((cvxpy_sol[:N] - cpu_sol[:N]).sum())
        for cvxpy_sol, cpu_sol in zip(cvxpy_sols, cpu_sols)
    ]
    cpu_matching = [i for i in range(Nprobs) if cpu_sol_dists[i] < eps]
    print("\tmatching cvxpy: {}/{}".format(len(cpu_matching), Nprobs))

    time.sleep(0.1)
    print("\nCPU lib (multi thread 8c/16t):")
    t0 = time.time()
    cpum_sols, cpum_stats = wrap_multisolve_multithread(probs)
    print("\tseconds elapsed:", time.time() - t0)
    print("\t...")

    time.sleep(0.1)
    print("\nGPU via OpenCL:")
    t0 = time.time()
    gpu_sols_flat, kkt_sq_res = solve_probs(probs)
    gpu_sols = [gpu_sols_flat[_M * i : _M * i + N] for i in range(Nprobs)]
    print("\tseconds elapsed:", time.time() - t0)
    print("\tproblems converged: {}/{}".format((kkt_sq_res < eps).sum(), Nprobs))
    gpu_sol_dists = [
        np.abs((cvxpy_sol[:N] - gpu_sol[:N]).sum())
        for cvxpy_sol, gpu_sol in zip(cvxpy_sols, gpu_sols)
    ]
    gpu_matching = [i for i in range(Nprobs) if gpu_sol_dists[i] < eps]
    print("\tmatching cvxpy: {}/{}".format(len(gpu_matching), Nprobs))
