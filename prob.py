import cvxpy as cp
import numpy as np

from flattened_prob import augment, print_c_arr

np.random.seed(1)

# Create shaped variables and coefficients
N = 2
x = cp.Variable(shape=(N), nonneg=True)
A_ = np.random.uniform(0.1, 1, (N, N))
G = np.random.uniform(0.1, 1, (N, N))

sol = np.random.uniform(0.1, 1, N)
dlt = np.random.uniform(0.1, 1, N)

b_ = A_ @ sol
h = G @ sol + dlt
c_ = np.random.uniform(0.1, 1, N)

# Create two constraints.
constraints = [A_ @ x == b_, G @ x <= h]

# Form objective.
obj = cp.Minimize(cp.sum(cp.multiply(c_, x)))

# Form and solve problem.
prob = cp.Problem(obj, constraints)

# data, chain, inverse_data = prob.get_problem_data(cp.SCIPY)
#
# A, b, c = augment(
#     data["A"].toarray(), data["b"], data["G"].toarray(), data["h"], data["c"]
# )
A, b, c = augment(A_, b_, G, h, c_)

print("intended sol: ", sol)

print(A)
print(A.shape)
print_c_arr(A)
print(b)
print_c_arr(b)
print(c)
print_c_arr(c)

print("cost: ", prob.solve())
print("status: ", prob.status)

print("solution: ", x.value)
