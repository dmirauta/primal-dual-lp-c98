import cvxpy as cp
import numpy as np

from common import augment, print_c_arr


if __name__ == "__main__":
    np.random.seed(1)

    # Create shaped variables and coefficients
    X = cp.Variable(shape=(3, 3), nonneg=True)
    CS = np.random.uniform(0.1, 1, (3, 3))
    CS /= CS.sum()

    bs = np.random.uniform(0.1, 1, 3)
    hs = np.random.uniform(0.1, 1, 3)

    # Create two constraints.
    constraints = [cp.sum(X, 0) == bs, cp.sum(X, 1) <= 2 * hs]

    # Form objective.
    obj = cp.Minimize(cp.sum(cp.multiply(CS, X)))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    # get flattened representation, adds nonnegativity constraints
    #                               (already built into our solver,
    #                                so may be undesired)
    data, chain, inverse_data = prob.get_problem_data(cp.SCIPY)

    A, b, c = augment(
        data["A"].toarray(), data["b"], data["G"].toarray(), data["h"], data["c"]
    )

    print(A)
    print(A.shape)
    print_c_arr(A)
    print(b)
    print_c_arr(b)
    print(c)
    print_c_arr(c)

    print("cost: ", prob.solve())
    print("status: ", prob.status)

    print("solution: ", X.value)
