import numpy as np

from ml_control.systems import solve_homogeneous_system, get_state_from_final_time_adjoint, get_control_from_final_time_adjoint


class ReducedModel:
    def __init__(self, reduced_basis, N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT,
                 R_chol, M, spatial_norm=lambda x: np.linalg.norm(x)):
        self.reduced_basis = reduced_basis
        self.N = N
        self.T = T
        self.nt = nt
        self.parametrized_A = parametrized_A
        self.parametrized_B = parametrized_B
        self.parametrized_x0 = parametrized_x0
        self.parametrized_xT = parametrized_xT
        self.R_chol = R_chol
        self.M = M
        self.spatial_norm = spatial_norm

    def solve(self, mu, return_adjoint=True, return_adjoint_coefficients=False):
        """Solves the reduced basis reduced model for the given parameter."""
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        x0_T_mu = solve_homogeneous_system(x0, self.T, self.nt, A)[-1]

        mat = np.array([phi - self.M @ get_state_from_final_time_adjoint(phi, np.zeros(self.N), self.T, self.nt,
                                                                         A, B, self.R_chol)[-1]
                        for phi in self.reduced_basis]).T
        phi_reduced_coefficients = np.linalg.solve(mat.T @ mat, mat.T @ self.M @ (x0_T_mu - xT))
        if return_adjoint_coefficients:
            return phi_reduced_coefficients

        phi_reduced = self.reduced_basis.T @ phi_reduced_coefficients
        u = get_control_from_final_time_adjoint(phi_reduced, self.T, self.nt, A, B, self.R_chol)
        if return_adjoint:
            return u, phi_reduced
        else:
            return u

    def estimate_error(self, mu):
        """Estimates the error in the final time adjoint."""
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        x0_T_mu = solve_homogeneous_system(x0, self.T, self.nt, A)[-1]

        mat = np.array([phi - self.M @ get_state_from_final_time_adjoint(phi, np.zeros(self.N), self.T, self.nt,
                                                                         A, B, self.R_chol)[-1]
                        for phi in self.reduced_basis]).T
        phi_reduced_coefficients = np.linalg.solve(mat.T @ mat, mat.T @ self.M @ (x0_T_mu - xT))
        projection = mat @ phi_reduced_coefficients

        estimated_error_mu = self.spatial_norm(projection - self.M @ (x0_T_mu - xT))
        return estimated_error_mu
