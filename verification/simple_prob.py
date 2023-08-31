import cvxpy as cp
import numpy as np

from common import augment, gen_lin_grad, print_prob, res_vec


def _gen_lp(seed=1, N=2):
    np.random.seed(seed)
    A_ = np.random.uniform(0.1, 1, (N, N))
    G = np.random.uniform(0.1, 1, (N, N))
    sol = np.random.uniform(0.1, 1, N)
    dlt = np.random.uniform(0.1, 1, N)
    b_ = A_ @ sol
    h = G @ sol + dlt
    c_ = np.random.uniform(0.1, 1, N)
    return A_, b_, G, h, c_


def gen_lp(*args, **kwargs):
    return augment(*_gen_lp(*args, **kwargs))


def build_cvxpy_prob(nonneg, N=2, seed=1):
    """
    we don't need nonneg augmentation for our solver
    but do need it for solving with cvxpy
    """
    A_, b_, G, h, c_ = _gen_lp(N=N, seed=seed)

    # Create shaped variables and coefficients
    x = cp.Variable(shape=(N), nonneg=nonneg)

    constraints = [A_ @ x == b_, G @ x <= h]

    # Form objective.
    obj = cp.Minimize(cp.sum(cp.multiply(c_, x)))

    # Form and solve problem.
    return cp.Problem(obj, constraints), x


def summarise_cvxpy_sol(N=2, seed=1):
    prob, x = build_cvxpy_prob(True, N, seed)
    print("cost: ", prob.solve())
    print("status: ", prob.status)

    print("solution: ", x.value)


def half_step_and_print():
    G = gen_lin_grad(A, _x, u)
    r = res_vec(A, b, _x, c, u, v)
    d_xuv = np.linalg.solve(G, r)

    print("grad:")
    print(G)
    print("res:")
    print(r)
    print("d_xuv:")
    print(d_xuv)

    return d_xuv


if __name__ == "__main__":
    seed = 9
    A, b, c = gen_lp(seed=seed)

    print_prob(A, b, c)

    summarise_cvxpy_sol(seed=seed)

    # ##################################################
    # ## Double check solver implementation/debug output
    #
    # n, m = A.shape  # not the same as N...
    # _x = np.ones(m)
    # u = np.ones(m)
    # v = np.ones(n)
    #
    # print("Expected first step linear system:")
    # d_xuv = half_step_and_print()
    #
    # _x += 0.1 * d_xuv[:m]
    # u += 0.1 * d_xuv[m : 2 * m]
    # v += 0.1 * d_xuv[2 * m :]
    #
    # print("Expected second step linear system:")
    # half_step_and_print()
