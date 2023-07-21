import numpy as np
from scipy.linalg import cho_factor

from ml_control.utilities import InnerProdNumpyArray


def create_heat_equation_problem(N=100):
    """Creates all required components of a simple heat equation problem."""
    h = 1. / (N + 1)

    def parametrized_A(mu):
        A_tilde = -2. * np.eye(N) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
        return mu * A_tilde / h**2

    def parametrized_B(mu):
        B_mat = np.zeros((N, 1))
        B_mat[-1] = mu / h**2
        B_mat = InnerProdNumpyArray(B_mat, transpose_scaling_factor=h)
        return B_mat

    def parametrized_x0(_):
        return np.sin(np.linspace(h, 1.-h, N) * np.pi)

    def parametrized_xT(_):
        return np.linspace(h, 1.-h, N)

    nt = 30 * N
    T = 0.1

    gamma = 0.1
    R = gamma * np.eye(1)
    R_chol = cho_factor(R)
    M = np.eye(N)

    parameter_space = (1, 2)

    return T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, parameter_space


def create_heat_equation_problem_with_two_parameters(N=100):
    """Creates all required components of a heat equation problem with two parameters."""
    h = 1. / (N + 1)

    def parametrized_A(mu):
        A_tilde = -2. * np.eye(N) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
        return mu[0] * A_tilde / h**2

    def parametrized_B(mu):
        B_mat = np.zeros((N, 1))
        B_mat[-1] = mu[0] / h**2
        B_mat = InnerProdNumpyArray(B_mat, transpose_scaling_factor=h)
        return B_mat

    def parametrized_x0(_):
        return np.sin(np.linspace(h, 1.-h, N) * np.pi)

    def parametrized_xT(mu):
        return mu[1] * np.linspace(h, 1.-h, N)

    nt = 30 * N
    T = 0.1

    gamma = 0.1
    R = gamma * np.eye(1)
    R_chol = cho_factor(R)
    M = np.eye(N)

    parameter_space = [(1, 2), (0.5, 1.5)]

    return T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, parameter_space


def create_heat_equation_problem_complex(N=100):
    """Creates all required components of a heat equation problem with two parameters and two controls."""
    h = 1. / (N + 1)

    def parametrized_A(mu):
        A_tilde = -2. * np.eye(N) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
        return mu[0] * A_tilde / h**2

    def parametrized_B(mu):
        B_mat = np.zeros((N, 2))
        B_mat[0, 0] = mu[0] / h**2
        B_mat[-1, 1] = mu[0] / h**2
        B_mat = InnerProdNumpyArray(B_mat, transpose_scaling_factor=h)
        return B_mat

    def parametrized_x0(_):
        return np.sin(np.linspace(h, 1.-h, N) * np.pi)

    def parametrized_xT(mu):
        return mu[1] * np.linspace(h, 1.-h, N)

    nt = 30 * N
    T = 0.1

    gamma_1 = 0.125
    gamma_2 = 0.25
    R = np.diag([gamma_1, gamma_2])
    R_chol = cho_factor(R)
    M = np.eye(N)

    parameter_space = [(1, 2), (0.5, 1.5)]

    return T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, parameter_space
