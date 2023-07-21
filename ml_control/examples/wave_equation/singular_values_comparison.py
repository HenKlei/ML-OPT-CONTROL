import numpy as np
import pathlib
import matplotlib.pyplot as plt

from ml_control.problem_definitions.wave_equation import create_wave_equation_problem
from ml_control.systems import solve_optimal_control_problem


k_train = 50
damping_forces = [0, 1, 5, 10, 50, 100]

cg_params = {}
svs = []

for damping in damping_forces:
    T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, \
        R_chol, M, parameter_space = create_wave_equation_problem(damping_force=damping)
    training_parameters = np.linspace(*parameter_space, k_train)
    phiT_init = np.zeros(N)

    optimal_adjoints = []
    for mu in training_parameters:
        A = parametrized_A(mu)
        B = parametrized_B(mu)
        x0 = parametrized_x0(mu)
        xT = parametrized_xT(mu)
        phi_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)
        optimal_adjoints.append(phi_opt)
    _, singular_values, _ = np.linalg.svd(np.array(optimal_adjoints))
    plt.semilogy(singular_values, label=f"Damping force={damping}")
    svs.append(singular_values)

plt.legend()
plt.show()

write_results = True
if write_results:
    filepath_prefix = 'plot_data_undamped/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    with open(filepath_prefix + 'singular_values_optimal_adjoints_comparison.txt', 'w') as f:
        f.write("Damping force:\t")
        for damping in damping_forces:
            f.write(f"{damping}\t")
        f.write("\n")
        for i in range(len(svs[0])):
            f.write(f"{i + 1}\t")
            for j in range(len(damping_forces)):
                f.write(f"{svs[j][i]}\t")
            f.write("\n")
