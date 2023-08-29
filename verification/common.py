import numpy as np

np.set_printoptions(linewidth=300)


def augment(A, b, G, h, c):
    """
    turn inequalities to equalities by use of slack variables
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

    return An, bn, cn


def print_c_arr(arr, name="arr", max_per_row=8):
    print(f"FPN {name}[] =", end=" {")
    arr_flat = arr.flat
    for i, v in enumerate(arr_flat):
        print(f"{v:.4f}", end=", " if i + 1 < arr.size else " };\n")
        if i % max_per_row == max_per_row - 1:
            print()


def gen_lin_grad(A, x, u):
    N, M = A.shape
    K = N + 2 * M
    G = np.zeros((K, K))

    G[:N, :M] = A
    G[2 * M :, N + M :] = A.T
    G[N : N + M, :M] = np.diag(u)
    G[N : N + M, M : 2 * M] = np.diag(x)
    G[N + M :, M : 2 * M] = np.eye(M)

    return G


def res_vec(A, b, x, c, u, v, cs_eps=0.1):
    N, M = A.shape
    K = N + 2 * M
    r = np.zeros(K)
    r[:M] = b - (A @ x)
    r[M : 2 * M] = cs_eps - x * u
    r[2 * M :] = c - (A.T @ v) - u
    return r


def print_prob(A, b, c):
    print("A = ", A)
    print("b = ", b)
    print("c = ", c)
    N, M = A.shape

    print()
    print_c_arr(A, "A")
    print_c_arr(b, "b")
    print_c_arr(c, "c")
    print(
        f"""
    IDX N = {N};
    IDX M = {M};
    """
    )
