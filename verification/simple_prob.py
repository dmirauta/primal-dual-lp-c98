import cvxpy as cp
import numpy as np

from common import augment, gen_lin_grad, print_prob, res_vec


def build_problem(nonneg):
    np.random.seed(1)

    # Create shaped variables and coefficients
    x = cp.Variable(shape=(N), nonneg=nonneg)

    # Create two constraints.
    constraints = [A_ @ x == b_, G @ x <= h]

    # Form objective.
    obj = cp.Minimize(cp.sum(cp.multiply(c_, x)))

    # Form and solve problem.
    return cp.Problem(obj, constraints), x


def print_step():
    print("grad:")
    print(G)
    print("res:")
    print(r)
    print("d_xuv:")
    print(d_xuv)


if __name__ == "__main__":
    N = 2
    A_ = np.random.uniform(0.1, 1, (N, N))
    G = np.random.uniform(0.1, 1, (N, N))
    sol = np.random.uniform(0.1, 1, N)
    dlt = np.random.uniform(0.1, 1, N)
    b_ = A_ @ sol
    h = G @ sol + dlt
    c_ = np.random.uniform(0.1, 1, N)

    prob, x = build_problem(False)

    # data, chain, inverse_data = prob.get_problem_data(cp.SCIPY)
    #
    # A, b, c = augment(
    #     data["A"].toarray(), data["b"], data["G"].toarray(), data["h"], data["c"]
    # )
    A, b, c = augment(A_, b_, G, h, c_)

    print("intended sol: ", sol)

    print_prob(A, b, c)

    prob, x = build_problem(True)
    print("cost: ", prob.solve())
    print("status: ", prob.status)

    print("solution: ", x.value)

    ##################################################
    ## Double check solver implementation/debug output

    n, m = A.shape  # not the same as N...
    _x = np.ones(m)
    u = np.ones(m)
    v = np.ones(n)
    G = gen_lin_grad(A, _x, u)
    r = res_vec(A, b, _x, c, u, v)
    d_xuv = np.linalg.solve(G, r)

    print("Expected first step:")
    print_step()

    _x += 0.1 * d_xuv[:m]
    u += 0.1 * d_xuv[m : 2 * m]
    v += 0.1 * d_xuv[2 * m :]

    G = gen_lin_grad(A, _x, u)
    r = res_vec(A, b, _x, c, u, v)
    d_xuv = np.linalg.solve(G, r)

    print("Second")
    print_step()
