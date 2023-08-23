import cvxpy as cp
import numpy as np

from flattened_prob import augment, print_c_arr

np.random.seed(1)

# Create shaped variables and coefficients
x = cp.Variable(shape=(3), nonneg=True)
A_ = np.random.uniform(0.1, 1, (3, 3))
b_ = np.random.uniform(0.1, 1, 3)
G = np.random.uniform(0.1, 1, (3, 3))
h = np.random.uniform(0.1, 4, 3)
c_ = np.random.uniform(0.1, 1, 3)

# Create two constraints.
constraints = [A_ @ x == b_, G @ x <= h]

# Form objective.
obj = cp.Minimize(cp.sum(cp.multiply(c_, x)))

# Form and solve problem.
prob = cp.Problem(obj, constraints)

A, b, c = augment(A_, b_, G, h, c_)

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
