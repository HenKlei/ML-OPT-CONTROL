import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_solve


def solve_adjoint(phiT, T, nt, A):
    """Solves the adjoint equation backward in time."""
    p = np.zeros((nt + 1, phiT.shape[0]))
    p[nt] = phiT
    dt2 = T / (2. * nt)
    if isinstance(A, tuple):
        lu, piv = A[2], A[3]
        S = np.eye(A[4].shape[0]) + dt2 * A[4].T
    else:
        R = np.eye(A.shape[0]) - dt2 * A.T
        lu, piv = lu_factor(R)
        S = np.eye(A.shape[0]) + dt2 * A.T
    for k in range(nt - 1, -1, -1):
        p[k] = lu_solve((lu, piv), S @ p[k + 1])
    return p


def solve_system(x0, T, nt, A, B, u):
    """Solves the primal system forward in time."""
    x = np.zeros((nt + 1, x0.shape[0]))
    x[0] = x0
    dt2 = T / (2. * nt)
    if isinstance(A, tuple):
        lu, piv = A[0], A[1]
        S = np.eye(A[4].shape[0]) + dt2 * A[4]
    else:
        R = np.eye(A.shape[0]) - dt2 * A
        lu, piv = lu_factor(R)
        S = np.eye(A.shape[0]) + dt2 * A
    for k in range(nt):
        x[k + 1] = lu_solve((lu, piv), S @ x[k] + dt2 * B @ (u[k] + u[k + 1]))
    return x


def solve_homogeneous_system(x0, T, nt, A):
    """Solves the homogeneous primal system, i.e. the system without control, forward in time."""
    x = np.zeros((nt + 1, x0.shape[0]))
    x[0] = x0
    dt2 = T / (2. * nt)
    if isinstance(A, tuple):
        lu, piv = A[0], A[1]
        S = np.eye(A[4].shape[0]) + dt2 * A[4]
    else:
        R = np.eye(A.shape[0]) - dt2 * A
        lu, piv = lu_factor(R)
        S = np.eye(A.shape[0]) + dt2 * A
    for k in range(nt):
        x[k + 1] = lu_solve((lu, piv), S @ x[k])
    return x


def get_control_from_adjoint(phi, B, R_chol):
    """Computes the control from the trajectory of the adjoint."""
    return np.array([-cho_solve(R_chol, B.T @ p) for p in phi])


def get_state_from_adjoint(phi, x0, T, nt, A, B, R_chol):
    """Computes the state trajectory from the trajectory of the adjoint."""
    u = get_control_from_adjoint(phi, B, R_chol)
    return solve_system(x0, T, nt, A, B, u)


def get_state_from_final_time_adjoint(phiT, x0, T, nt, A, B, R_chol):
    """Computes the state trajectory from the final time adjoint."""
    phi = solve_adjoint(phiT, T, nt, A)
    return get_state_from_adjoint(phi, x0, T, nt, A, B, R_chol)


def get_control_from_final_time_adjoint(phiT, T, nt, A, B, R_chol):
    """Computes the control from the final time adjoint."""
    phi = solve_adjoint(phiT, T, nt, A)
    return get_control_from_adjoint(phi, B, R_chol)


def solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, max_iter=1000, eps=1e-12,
                                  compute_approximate_largest_eigenvalue=False):
    """Solves the optimal control problem exactly by applying the CG algorithm."""
    x0_T = solve_homogeneous_system(x0, T, nt, A)[-1]
    rhs = M @ (x0_T - xT)

    # Pre-compute LU factorisation of A
    dt2 = T / (2. * nt)
    R = np.eye(A.shape[0]) - dt2 * A
    lu, piv = lu_factor(R)
    luT, pivT = lu_factor(R.T)

    def apply_matrix(phiT):
        xT_phiT = get_state_from_final_time_adjoint(phiT, np.zeros_like(phiT), T, nt,
                                                    (lu, piv, luT, pivT, A), B, R_chol)[-1]
        assert (phiT - xT_phiT).dot(phiT) > 0. or (np.isclose((phiT - xT_phiT).dot(phiT), 0.)
                                                   and np.isclose(np.linalg.norm(phiT), 0.))
        return phiT - M @ xT_phiT

    if compute_approximate_largest_eigenvalue:
        def power_method(n_iter=100):
            p = rhs.copy()
            for i in range(n_iter):
                p = apply_matrix(p) / np.linalg.norm(p)
            return p.T.dot(apply_matrix(p)) / p.dot(p)

        approximate_largest_eigenvalue = power_method()

    phiT = phiT_init
    res = rhs - apply_matrix(phiT)
    res_norm = np.linalg.norm(res)
    d = res
    res_ = res

    k = 0
    while k < max_iter and res_norm > eps:
        z = apply_matrix(d)
        alpha = res.dot(res) / d.dot(z)
        phiT = phiT + alpha * d
        res = res - alpha * z
        beta = res.dot(res) / res_.dot(res_)
        d = res + beta * d
        res_norm = np.linalg.norm(res)
        res_ = res
        k += 1

    if compute_approximate_largest_eigenvalue:
        return phiT, approximate_largest_eigenvalue
    return phiT
