import os, time, ctypes, struct
import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

mf = cl.mem_flags
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

parent = os.path.dirname(os.getcwd())
build_options_base = f"-I {parent}"

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


# class LPDef(ctypes.Structure):
#     _fields_ = [
#         ("A_ptr", ctypes.POINTER(c_fpn)),
#         ("b_ptr", ctypes.POINTER(c_fpn)),
#         ("c_ptr", ctypes.POINTER(c_fpn)),
#         ("N", ctypes.c_ulong),
#         ("M", ctypes.c_ulong),
#     ]


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

with open("../kernel.cl", "r") as f:
    kernel_cl = f.read()


def solve_probs(probs, opts=(0.1, 0.5, 1e-19, 1.0, 1000)):
    Nprobs = len(probs)

    N, M = probs[0][0].shape
    build_opts = f"{build_options_base} -D NDEF={N} -D MDEF={M}"
    print("build opts: ", build_opts)

    As_cpu = np.zeros(N * M * Nprobs, dtype=np_fpn)
    bs_cpu = np.zeros(N * Nprobs, dtype=np_fpn)
    cs_cpu = np.zeros(M * Nprobs, dtype=np_fpn)
    sols_cpu = np.zeros(M * Nprobs, dtype=np_fpn)
    opts_c = struct.pack(solver_opt_struct_fmt, *opts)  # not currently being passed

    # make flat arrays
    for i, (A, b, c) in enumerate(probs):
        As_cpu[i * N * M : (i + 1) * N * M] = A.flatten()
        bs_cpu[i * N : (i + 1) * N] = b.flatten()
        cs_cpu[i * M : (i + 1) * M] = c.flatten()

    As_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=As_cpu)
    bs_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bs_cpu)
    cs_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cs_cpu)
    sols_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, fps * M * Nprobs)

    prg = cl.Program(ctx, kernel_cl).build(options=build_opts)
    prg.solve_lps(queue, (Nprobs,), None, As_gpu, bs_gpu, cs_gpu, sols_gpu)

    cl.enqueue_copy(queue, sols_cpu, sols_gpu)

    for obj in [As_gpu, bs_gpu, cs_gpu, sols_gpu]:
        obj.release()

    return sols_cpu


if __name__ == "__main__":
    from simple_prob import gen_lp

    N = 2  # base N will not be the same as augmented...
    Nprobs = 50

    probs = [gen_lp(N=N, seed=i) for i in range(Nprobs)]

    t0 = time.time()
    sols = solve_probs(probs)
    print("elapsed:", time.time() - t0)
    print(sols[:2])  # first solution, expecting ~ [0.45709073, 0.58493506]