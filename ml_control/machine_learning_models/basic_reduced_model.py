import numpy as np

from ml_control.logger import getLogger
from ml_control.systems import get_control_from_final_time_adjoint, get_state_from_final_time_adjoint,\
    solve_homogeneous_system


class BasicReducedMachineLearningModel:
    def __init__(self, reduced_model, training_data, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                 parametrized_xT, R_chol, M, logger_name, spatial_norm=lambda x: np.linalg.norm(x), zero_padding=True):
        self.training_data = training_data
        self.reduced_model = reduced_model
        self.N = self.reduced_model.N
        self.T = T
        self.nt = nt
        self.parametrized_A = parametrized_A
        self.parametrized_B = parametrized_B
        self.parametrized_x0 = parametrized_x0
        self.parametrized_xT = parametrized_xT
        self.R_chol = R_chol
        self.M = M
        self.spatial_norm = spatial_norm

        self.zero_padding = zero_padding

        self.logger = getLogger(logger_name, level='INFO')

    def train(self):
        """Trains the machine learning surrogate."""
        raise NotImplementedError

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the machine learning surrogate."""
        raise NotImplementedError

    def extend_model(self):
        if self.zero_padding:
            # Add zero padding to machine learning training data
            for i, (mu, coeffs) in enumerate(self.training_data):
                self.training_data[i] = (mu, np.hstack([coeffs, 0.]))
        else:
            # Train a machine learning surrogate for every component of the coefficients individually
            pass

    def solve(self, mu, reduced_coeffs=None, return_adjoint=True, return_adjoint_coefficients=False):
        """Solves the machine learning reduced model for the given parameter."""
        if reduced_coeffs is None:
            if self.reduced_model.reduced_basis is not None:
                phi_reduced_coefficients = self.get_coefficients(mu)
                if return_adjoint_coefficients:
                    return phi_reduced_coefficients
                phi_reduced = self.reduced_model.reduced_basis.T @ phi_reduced_coefficients
            else:
                if return_adjoint_coefficients:
                    return None
                phi_reduced = np.zeros(self.reduced_model.N)
        else:
            phi_reduced = self.reduced_model.reduced_basis.T @ reduced_coeffs

        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        u = get_control_from_final_time_adjoint(phi_reduced, self.T, self.nt, A, B, self.R_chol)
        if return_adjoint:
            return u, phi_reduced
        else:
            return u

    def estimate_error(self, mu, return_reduced_coefficients=False):
        """Estimates the error in the final time adjoint."""
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        x0_T_mu = solve_homogeneous_system(x0, self.T, self.nt, A)[-1]

        if self.reduced_model.reduced_basis is not None:
            phi_reduced_coefficients = self.get_coefficients(mu)
            phi_reduced = self.reduced_model.reduced_basis.T @ phi_reduced_coefficients
            projection = phi_reduced - self.M @ get_state_from_final_time_adjoint(phi_reduced, np.zeros(self.N), self.T,
                                                                                  self.nt, A, B, self.R_chol)[-1]
        else:
            phi_reduced_coefficients = None
            projection = np.zeros(self.reduced_model.N)

        estimated_error_mu = self.spatial_norm(projection - self.M @ (x0_T_mu - xT))
        if return_reduced_coefficients:
            return estimated_error_mu, phi_reduced_coefficients
        return estimated_error_mu
