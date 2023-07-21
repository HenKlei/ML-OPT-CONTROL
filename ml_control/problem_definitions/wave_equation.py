import numpy as np
from scipy.linalg import cho_factor

from ml_control.utilities import InnerProdNumpyArray


def create_wave_equation_problem_with_two_controls(N=200, damping_force=0.):
    """Creates all required components of a wave equation problem with two controls."""
    h = 1. / (N // 2. + 1)

    def parametrized_A(mu):
        A_tilde = -2. * np.eye(N // 2) + np.diag(np.ones(N // 2 - 1), 1) + np.diag(np.ones(N // 2 - 1), -1)
        return np.block([[np.zeros((N // 2, N // 2)), np.eye(N // 2)],
                         [mu * A_tilde / h**2, -damping_force * np.eye(N // 2)]])

    def parametrized_B(mu):
        B_mat = np.zeros((N, 2))
        B_mat[N // 2, 0] = mu / h**2
        B_mat[-1, 1] = mu / h**2
        B_mat = InnerProdNumpyArray(B_mat, transpose_scaling_factor=h)
        return B_mat

    def parametrized_x0(_):
        return np.hstack([np.sin(np.linspace(h, 1.-h, N//2) * np.pi), np.zeros(N // 2)])

    def parametrized_xT(_):
        return np.zeros(N)

    nt = 10 * N
    T = 3.

    gamma_1 = 0.1
    gamma_2 = 1.
    R = np.diag([gamma_1, gamma_2])
    R_chol = cho_factor(R)
    M = np.eye(N) * 10.

    parameter_space = (3, 10)

    return T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, parameter_space


def create_wave_equation_problem(N=200, damping_force=0.):
    """Creates all required components of a damped wave equation problem."""
    h = 1. / (N // 2. + 1)

    def parametrized_A(mu):
        A_tilde = -2. * np.eye(N // 2) + np.diag(np.ones(N // 2 - 1), 1) + np.diag(np.ones(N // 2 - 1), -1)
        return np.block([[np.zeros((N // 2, N // 2)), np.eye(N // 2)],
                         [mu * A_tilde / h**2, -damping_force * np.eye(N // 2)]])

    def parametrized_B(mu):
        B_mat = np.zeros((N, 1))
        B_mat[-1] = mu / h**2
        B_mat = InnerProdNumpyArray(B_mat, transpose_scaling_factor=h)
        return B_mat

    def parametrized_x0(_):
        return np.hstack([np.sin(np.linspace(h, 1.-h, N//2) * np.pi), np.zeros(N // 2)])

    def parametrized_xT(_):
        return np.hstack([np.linspace(h, 1.-h, N//2), np.zeros(N//2)])

    nt = 10 * N
    T = 1.

    gamma = 0.1
    R = gamma * np.eye(1)
    R_chol = cho_factor(R)
    M = np.eye(N) * 10.

    parameter_space = (3, 10)

    return T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, parameter_space
