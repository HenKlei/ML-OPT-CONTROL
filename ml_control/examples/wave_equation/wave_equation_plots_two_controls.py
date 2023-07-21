import numpy as np
import matplotlib.pyplot as plt

from ml_control.problem_definitions.wave_equation import create_wave_equation_problem_with_two_controls
from ml_control.systems import solve_optimal_control_problem, get_control_from_final_time_adjoint, \
    get_state_from_final_time_adjoint, solve_system
from ml_control.visualization import plot_final_time_adjoints, plot_controls, plot_final_time_solutions, \
    animate_solution


T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, \
    R_chol, M, parameter_space = create_wave_equation_problem_with_two_controls()

mu = 5.
A = parametrized_A(mu)
B = parametrized_B(mu)
x0 = parametrized_x0(mu)
xT = parametrized_xT(mu)
phiT_init = np.zeros_like(x0)
phiT_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init)
u_opt = get_control_from_final_time_adjoint(phiT_opt, T, nt, A, B, R_chol)
x_opt = get_state_from_final_time_adjoint(phiT_opt, x0, T, nt, A, B, R_chol)
# Just for testing purposes:
x_opt_2 = solve_system(x0, T, nt, A, B, u_opt)
assert np.allclose(x_opt, x_opt_2)

plot_final_time_adjoints([phiT_opt], title="Final time optimal adjoint")
plot_controls([u_opt], T, title="Optimal control")
plot_final_time_solutions([x_opt[-1]], title="Final time optimal solution")
animate_solution(x_opt, ylim=(np.min(x_opt) * 1.1, np.max(x_opt) * 1.1), title="Optimal solution")

spatial_norm = lambda x: np.linalg.norm(h * x)
print(f"Deviation in final time state: {spatial_norm(x_opt[-1] - xT)}")

fig, axs = plt.subplots(3)
plot_final_time_adjoints([phiT_opt], show_plot=False, ax=axs[0])
axs[0].set_title("Optimal final time adjoint")
plot_controls([u_opt], T, show_plot=False, ax=axs[1])
axs[1].set_title("Optimal control")
plot_final_time_solutions([x_opt[-1], xT], labels=["Optimal state", "Target state"], show_plot=False, ax=axs[2])
axs[2].set_title("Final time states")
axs[2].legend()
fig.suptitle(f"Results for parameter {mu}")
plt.show()
