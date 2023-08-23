import cvxpy as cp
import numpy as np


def augment(A, b, G, h, c):
    """
    turn inequalities to equalities
    """

    NA, M = A.shape
    NG, M = G.shape

    NAn = NA + NG
    MAn = M + NG
    An = np.zeros((NAn, MAn))
    bn = np.zeros(NAn)
    cn = np.zeros(MAn)

    An[:NA, :M] = A
    An[NA:, :M] = G
    An[NA:, M:] = np.eye(NG)

    bn[:NA] = b
    bn[NA:] = h

    cn[:M] = c

    return An, bn, c


def print_c_arr(arr):
    print("= { ", end="")
    arr_flat = arr.flat
    for v in arr_flat[:-1]:
        print("{:.4f}, ".format(v), end="")
    print("{:.4f}".format(arr_flat[-1]) + " }")


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
