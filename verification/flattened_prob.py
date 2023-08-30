import cvxpy as cp
import numpy as np

from common import augment, print_c_arr, print_prob


def build_problem(nonneg):
    """
    we don't need nonneg augmentation for our solver
    but do need it for solving with cvxpy
    """
    np.random.seed(1)

    N = 2
    # Create shaped variables and coefficients
    X = cp.Variable(shape=(N, N), nonneg=nonneg)
    CS = np.random.uniform(0.1, 1, (N, N))
    CS /= CS.sum()

    # Bs = np.random.uniform(0.1, 1, (N, N))
    bs = np.random.uniform(0.1, 1, N)
    hs = np.random.uniform(0.1, 1, N)

    constraints = [cp.sum(X, 0) == bs, cp.sum(X, 1) <= 2 * hs]
    # constraints += [X <= Bs]

    # Form objective.
    obj = cp.Minimize(cp.sum(cp.multiply(CS, X)))

    # Form and solve problem.
    return cp.Problem(obj, constraints), X


if __name__ == "__main__":
    prob, X = build_problem(False)

    # get flattened representation
    data, chain, inverse_data = prob.get_problem_data(cp.SCIPY)

    A, b, c = augment(
        data["A"].toarray(), data["b"], data["G"].toarray(), data["h"], data["c"]
    )

    print_prob(A, b, c)

    prob, X = build_problem(True)

    print("cost: ", prob.solve())
    print("status: ", prob.status)

    print("solution: ", X.value)
