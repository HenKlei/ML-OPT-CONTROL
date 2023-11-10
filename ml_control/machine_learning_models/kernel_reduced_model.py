import numpy as np
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian

from ml_control.machine_learning_models.basic_reduced_model import BasicReducedMachineLearningModel


class KernelReducedModel(BasicReducedMachineLearningModel):
    def __init__(self, reduced_model, training_data, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                 parametrized_xT, R_chol, M, spatial_norm=lambda x: np.linalg.norm(x), zero_padding=True):
        super().__init__(reduced_model, training_data, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                         parametrized_xT, R_chol, M, 'KernelReducedModel', spatial_norm=spatial_norm,
                         zero_padding=zero_padding)

    def train(self, kernel=Gaussian(0.1), greedy_type='p_greedy', tol_p=1e-10, max_iter=None):
        """Trains the VKOGA surrogate."""
        training_data_x = []
        training_data_y = []
        for mu, coeffs in self.training_data:
            training_data_x.append(mu)
            training_data_y.append(coeffs)
        training_data_x = np.array(training_data_x)
        training_data_y = np.array(training_data_y)
        if training_data_x.ndim == 1:
            training_data_x = training_data_x[..., np.newaxis]

        self.kernel_model = VKOGA(kernel=kernel, greedy_type=greedy_type, tol_p=tol_p)
        self.kernel_model = self.kernel_model.fit(training_data_x, training_data_y,
                                                  maxIter=max_iter or len(training_data_x))

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the VKOGA surrogate."""
        converted_input = mu.reshape(1, -1)
        if hasattr(self, "kernel_model"):
            return self.kernel_model.predict(converted_input).flatten()
        else:
            return np.zeros(len(self.reduced_model.reduced_basis))
