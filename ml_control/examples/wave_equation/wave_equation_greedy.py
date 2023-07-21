import numpy as np
import time
import pathlib

from ml_control.analysis import run_greedy_and_analysis, write_results_to_file
from ml_control.problem_definitions.wave_equation import create_wave_equation_problem
from ml_control.logger import getLogger


T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, \
    R_chol, M, parameter_space = create_wave_equation_problem()

k_train = 50
training_parameters = np.linspace(*parameter_space, k_train)
k_test_plotting = 5
test_parameters_plotting = np.random.uniform(*parameter_space, k_test_plotting)
k_test_analysis = 100
test_parameters_analysis = np.random.uniform(*parameter_space, k_test_analysis)
tol = 1e-1
max_basis_size = min(30, k_train)
cg_params = {}
logger = getLogger('wave_equation', level='INFO')

results_greedy, results_analysis = run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                                                           test_parameters_analysis, N, T, nt, parametrized_A,
                                                           parametrized_B, parametrized_x0, parametrized_xT, R_chol, M,
                                                           tol, max_basis_size, cg_params, logger,
                                                           spatial_norm=lambda x: np.linalg.norm(h * x),
                                                           temporal_norm=lambda u: np.linalg.norm(u * (T / nt)))

write_results = True
if write_results:
    filepath_prefix = 'results/results_wave_equation_' + time.strftime('%Y%m%d-%H%M%S') + '/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    write_results_to_file(results_greedy, results_analysis, filepath_prefix)
